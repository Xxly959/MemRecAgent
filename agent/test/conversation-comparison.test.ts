// 实际持续Python桥接+真实Pi循环；仅外部模型为固定回调，不计模型效果。
import assert from 'node:assert/strict';
import {test} from 'node:test';
import {mkdtempSync,writeFileSync,readFileSync,readdirSync,rmSync,existsSync} from 'node:fs';
import {tmpdir} from 'node:os';
import {resolve} from 'node:path';
import {createAssistantMessageEventStream,type AssistantMessage} from '@earendil-works/pi-ai';
import type {StreamFn} from '@earendil-works/pi-agent-core';
import {ConversationClient,comparisonConnection} from '../src/conversation-client.js';
import {runConversationTurn} from '../src/conversation-agent.js';
import {python,root} from './helpers.js';

function setup(quantityOverride?:number,attributeOverride?:number,withAliases=false) {
  const folder=mkdtempSync(resolve(tmpdir(),'memrec-comparison-'));
  const worker=resolve(folder,'worker.py'),trace=resolve(folder,'analysis.jsonl');
  writeFileSync(worker,`#!${python}
import sys,json
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0,${JSON.stringify(resolve(root,'tests'))})
from memrecagent.entity_aliases import install_entity_aliases
from test_attribute_language import durable
from test_request_attributes import host,payload
from test_memory_interaction import value
from test_memory_analyst import reply
from memrecagent.conversation_bridge import ConversationBridge
from memrecagent.agent_bridge import serve_bridge
def complete(messages):
 data=json.loads(messages[-1]['content']);row=data['turns'][-1]
 t=SimpleNamespace(turn_id=row['turn_id'],content=row['content'])
 with Path(${JSON.stringify(trace)}).open('a') as stream:stream.write('{}\\n')
 raw=durable(t,recommend=True) if '软预算' in t.content else value(t,entity='BrandC' if 'BrandC' in t.content else '品牌C',relation='prefer') if '喜欢' in t.content else payload(t,recommend=True) if '预算' in t.content else value(t,'recommend')
 if '软预算' in t.content:
  raw['attribute_changes'][0]['is_hard']=False
  if '这次' in t.content:raw['attribute_changes'][0]['scope']='request'
 return reply(raw)
if ${withAliases?'True':'False'}:
 install_entity_aliases(Path(${JSON.stringify(folder)})/'memory',dict(schema_version=1,entries=[dict(category='phone',canonical='品牌C',aliases=['BrandC','C牌'],source='synthetic:catalog:v1')]))
bridge=ConversationBridge(host(Path(${JSON.stringify(folder)})),complete=complete)
attribute_override=${attributeOverride===undefined?'None':String(attributeOverride)}
quantity_override=${quantityOverride===undefined?'None':String(quantityOverride)}
class Wire:
 def handle(self,row):
  result=bridge.handle(row)
  if quantity_override is not None and result.get('ok') and result.get('result',{}).get('ranking'):
   result['result']['ranking']['requested_count']=quantity_override
  if attribute_override is not None and result.get('ok') and result.get('result',{}).get('ranking',{}).get('items'):
   result['result']['ranking']['items'][0]['attribute_preference_signal']=attribute_override
  return result
serve_bridge(Wire(),sys.stdin.buffer,sys.stdout.buffer)
`,{mode:0o700});
  const owner=new ConversationClient({python:worker,bundle:'/unused',facts:'/unused',memoryRoot:'/unused',
    userId:'u',sessionId:'s',apiKey:'synthetic-placeholder'});
  return {owner,folder,trace};
}

function bytes(folder:string) {
  if(!existsSync(folder)) return [];
  return readdirSync(folder,{recursive:true,withFileTypes:true}).filter(e=>e.isFile())
    .map(e=>{const path=resolve(e.parentPath,e.name);return [path.slice(folder.length),readFileSync(path).toString('base64')];})
    .sort(([a],[b])=>a.localeCompare(b));
}

function model(reversed=false):StreamFn {
  let count=0;
  return (_model,context)=>{
    const stream=createAssistantMessageEventStream();
    let content:AssistantMessage['content'];
    if(count++===0) content=[{type:'toolCall',id:'read-branch',name:'conversation_step',arguments:{}}];
    else {
      const tool=context.messages.findLast(m=>m.role==='toolResult');
      assert.ok(tool?.role==='toolResult');
      const data=JSON.parse(tool.content.filter(c=>c.type==='text').map(c=>c.text).join(''));
      assert.ok(data.conversation_context.some((t:{role:string})=>t.role==='assistant'));
      assert.deepEqual(data.new_memory_ids,[]);
      const ids=data.ranking.items.map((p:{item_id:string})=>p.item_id);
      content=[{type:'text',text:JSON.stringify({reply:'',item_ids:reversed?[...ids].reverse():ids})}];
    }
    stream.end({role:'assistant',content,stopReason:count===1?'toolUse':'stop',api:'openai-completions',
      provider:'fixed',model:'fixed',timestamp:0,usage:{input:0,output:0,cacheRead:0,cacheWrite:0,totalTokens:0,
      cost:{input:0,output:0,cacheRead:0,cacheWrite:0,total:0}}});
    return stream;
  };
}

async function prefix(owner:ConversationClient) {
  const signal=new AbortController().signal;
  await owner.step('我长期喜欢品牌C。',signal);
  await owner.observe('已记录这条偏好。');
  await owner.step('这次预算三千，给我推荐手机。',signal);
  return owner.comparisonBinding(signal);
}

test('Pi四分支复用同一实际预算与历史，只读排名不重复分析和提交',async()=>{
  const {owner,folder,trace}=setup();
  try {
    const binding=await prefix(owner),before=bytes(resolve(folder,'memory'));
    for(const [memory,strategy] of [[false,'popularity'],[true,'popularity'],[false,'bpr_cold_popularity'],[true,'bpr_cold_popularity']] as const) {
      const connection=comparisonConnection(owner,binding,{useLongTermMemory:memory,strategy});
      const data=await connection.step(binding.prompt,new AbortController().signal);
      assert.equal(data.analysis_calls.length,0);assert.equal(data.new_memory_ids.length,0);
      assert.ok(data.ranking?.items.every(p=>p.item_id!=='A'));
      if(strategy==='popularity') assert.deepEqual(data.ranking?.items.map(p=>p.item_id),memory?['C','B']:['B','C']);
      const result=await runConversationTurn({client:connection,prompt:binding.prompt,streamFn:model()});
      assert.equal(result.status,'ok');assert.equal(result.modelCalls,2);assert.equal(result.toolCalls,1);
      assert.deepEqual(result.analysisCalls,[]);assert.deepEqual(result.memoryIds,[]);
      assert.equal(result.analysisStatus,'reported');
      await assert.rejects(connection.observe('这是反事实，不能伪造交付'),/comparison_read_only/);
    }
    assert.equal(readFileSync(trace,'utf8').trim().split('\n').length,4);
    assert.deepEqual(bytes(resolve(folder,'memory')),before);
  } finally {await owner.close();rmSync(folder,{recursive:true,force:true});}
});

test('单个分支输出重排被拒绝，不能关闭共享会话或污染其他分支',async()=>{
  const {owner,folder,trace}=setup();
  try {
    const binding=await prefix(owner);
    const bad=comparisonConnection(owner,binding,{useLongTermMemory:false,strategy:'popularity'});
    const failed=await runConversationTurn({client:bad,prompt:binding.prompt,streamFn:model(true)});
    assert.equal(failed.status,'error');assert.equal(failed.code,'final_guard');
    const good=comparisonConnection(owner,binding,{useLongTermMemory:true,strategy:'popularity'});
    const result=await runConversationTurn({client:good,prompt:binding.prompt,streamFn:model()});
    assert.equal(result.status,'ok');assert.equal(readFileSync(trace,'utf8').trim().split('\n').length,4);
  } finally {await owner.close();rmSync(folder,{recursive:true,force:true});}
});

test('排名后同状态的新问题使旧绑定失效，不发送第二次模型',async()=>{
  const {owner,folder}=setup();
  try {
    const binding=await prefix(owner);
    const base=comparisonConnection(owner,binding,{useLongTermMemory:true,strategy:'popularity'});
    const changed={...base,step:async(text:string,signal:AbortSignal)=>{
      const data=await base.step(text,signal);
      await owner.step('再次给我推荐手机。',signal);
      return data;
    }};
    const result=await runConversationTurn({client:changed,prompt:binding.prompt,streamFn:model()});
    assert.equal(result.status,'error');assert.equal(result.code,'conversation_state');assert.equal(result.modelCalls,1);
    const fresh=await owner.comparisonBinding(new AbortController().signal);
    assert.equal(fresh.stateToken,binding.stateToken);assert.notEqual(fresh.requestToken,binding.requestToken);
  } finally {await owner.close();rmSync(folder,{recursive:true,force:true});}
});

test('实际Pi四组沿用本次数量，预算后不足时如实提示不补造商品',async()=>{
  const {owner,folder,trace}=setup();
  try {
    const signal=new AbortController().signal;
    await owner.step('我长期喜欢品牌C。',signal);
    await owner.observe('已记录这条偏好。');
    const prompt='这次预算三千，给我推荐三款手机。';
    await owner.step(prompt,signal);
    const binding=await owner.comparisonBinding(signal),before=bytes(resolve(folder,'memory'));
    for(const [memory,strategy] of [[false,'popularity'],[true,'popularity'],[false,'bpr_cold_popularity'],[true,'bpr_cold_popularity']] as const) {
      const connection=comparisonConnection(owner,binding,{useLongTermMemory:memory,strategy});
      const data=await connection.step(prompt,signal);
      assert.equal((data.ranking as unknown as {requested_count:number}).requested_count,3);
      assert.equal(data.ranking?.items.length,2);
      const result=await runConversationTurn({client:connection,prompt,streamFn:model()});
      assert.equal(result.status,'ok');assert.equal(result.modelCalls,2);
      assert.match(result.finalText??'',/仅找到2款/);
      assert.match(result.finalText??'',/希望3款/);
      assert.doesNotMatch(result.finalText??'',/3\. /);
    }
    assert.equal(readFileSync(trace,'utf8').trim().split('\n').length,4);
    assert.deepEqual(bytes(resolve(folder,'memory')),before);
  } finally {await owner.close();rmSync(folder,{recursive:true,force:true});}
});

for(const count of [1,11]) test(`客户端拒绝不合法或小于实际返回数量的回执 ${count}`,async()=>{
  const {owner,folder}=setup(count);
  try {
    await assert.rejects(owner.step('预算三千，推荐两款手机。',new AbortController().signal),/conversation_protocol/);
  } finally {await owner.close();rmSync(folder,{recursive:true,force:true});}
});


test('真实Python展示后临时屏蔽与撤销不调用分析，下一轮不继承',async()=>{
  const {owner,folder,trace}=setup();
  const signal=new AbortController().signal;
  try {
    await prefix(owner);
    await assert.rejects(owner.registerFeedbackDisplay(signal));
    await owner.observe('实际向用户显示C、B。');
    await owner.registerFeedbackDisplay(signal);
    const before=bytes(resolve(folder,'memory'));
    const blocked=await owner.blockDisplayed(1,'product','request',signal);
    assert.deepEqual(blocked.ranking.items.map(p=>p.item_id),['B']);
    assert.equal(blocked.scope,'request');
    assert.deepEqual(bytes(resolve(folder,'memory')),before);
    const undone=await owner.undoFeedback(blocked.eventId,signal);
    assert.deepEqual(undone.ranking.items.map(p=>p.item_id),['C','B']);
    await owner.observe('撤销后实际展示C、B。');
    await owner.registerFeedbackDisplay(signal);
    const permanent=await owner.blockDisplayed(1,'brand','persistent',signal);
    assert.deepEqual(permanent.ranking.items.map(p=>p.item_id),['B']);
    const next=await owner.step('再次给我推荐手机。',signal);
    assert.deepEqual(next.ranking?.items.map(p=>p.item_id),['B']);
    assert.equal(readFileSync(trace,'utf8').trim().split('\n').length,6);
  } finally {await owner.close();rmSync(folder,{recursive:true,force:true});}
});


test('实际Pi显示独立软属性信号，不把综合分变化误说成品牌偏好',async()=>{
  const {owner,folder,trace}=setup();
  let calls=0;
  const streamFn:StreamFn=(_model,context)=>{
    const stream=createAssistantMessageEventStream();
    const content:AssistantMessage['content']=calls++===0?
      [{type:'toolCall',id:'soft-attr',name:'conversation_step',arguments:{}}]:
      [{type:'text',text:JSON.stringify({reply:'',item_ids:JSON.parse(context.messages.findLast(m=>m.role==='toolResult')!.content.filter(c=>c.type==='text').map(c=>c.text).join('')).ranking.items.map((p:{item_id:string})=>p.item_id)})}];
    stream.end({role:'assistant',content,stopReason:calls===1?'toolUse':'stop',api:'openai-completions',provider:'fixed',model:'fixed',timestamp:0,
      usage:{input:0,output:0,cacheRead:0,cacheWrite:0,totalTokens:0,cost:{input:0,output:0,cacheRead:0,cacheWrite:0,total:0}}});
    return stream;
  };
  try {
    const result=await runConversationTurn({client:owner,prompt:'以后手机软预算三千，长期保存并推荐手机。',streamFn});
    assert.equal(result.status,'ok');assert.match(result.finalText??'',/软属性信号1\.000/);
    assert.equal(readFileSync(trace,'utf8').trim().split('\n').length,2);
  } finally {await owner.close();rmSync(folder,{recursive:true,force:true});}
});

test('客户端拒绝越界的软属性信号',async()=>{
  const {owner,folder}=setup(undefined,2);
  try {await assert.rejects(owner.step('给我推荐手机。',new AbortController().signal),/conversation_protocol/);}
  finally {await owner.close();rmSync(folder,{recursive:true,force:true});}
});


test('真实Pi四组使用可信别名规范键，保留原文且只读分支不写盘',async()=>{
  const {owner,folder,trace}=setup(undefined,undefined,true);
  const signal=new AbortController().signal;
  try {
    await owner.step('我长期喜欢BrandC。',signal);await owner.observe('已记录这条偏好。');
    await owner.step('这次预算三千，给我推荐手机。',signal);
    const binding=await owner.comparisonBinding(signal),before=bytes(resolve(folder,'memory'));
    for(const memory of [false,true]) for(const strategy of ['popularity','bpr_cold_popularity']) {
      const connection=comparisonConnection(owner,binding,{useLongTermMemory:memory,strategy});
      const data=await connection.step(binding.prompt,signal);
      assert.deepEqual(data.ranking?.items.map(p=>p.item_id),memory?['C','B']:['B','C']);
      const result=await runConversationTurn({client:connection,prompt:binding.prompt,streamFn:model()});
      assert.equal(result.status,'ok');
    }
    assert.deepEqual(bytes(resolve(folder,'memory')),before);
    assert.equal(readFileSync(trace,'utf8').trim().split('\n').length,4);
    const stored=readdirSync(resolve(folder,'memory')).filter(n=>n.endsWith('.jsonl')).map(n=>readFileSync(resolve(folder,'memory',n),'utf8')).join('');
    assert.match(stored,/BrandC/);assert.match(stored,/品牌C/);assert.match(stored,/entity_registry/);
  } finally {await owner.close();rmSync(folder,{recursive:true,force:true});}
});


test('实际Pi四组共享仅本次软偏好且不写长期记忆',async()=>{
  const {owner,folder,trace}=setup();
  const signal=new AbortController().signal;
  try {
    const first=await owner.step('这次手机软预算三千，别长期保存并推荐手机。',signal);
    assert.deepEqual(first.new_memory_ids,[]);
    await owner.observe('按本次偏好排序。');
    await owner.step('给我推荐手机。',signal);
    const binding=await owner.comparisonBinding(signal),before=bytes(resolve(folder,'memory'));
    for(const memory of [false,true]) for(const strategy of ['popularity','bpr_cold_popularity']) {
      const connection=comparisonConnection(owner,binding,{useLongTermMemory:memory,strategy});
      const data=await connection.step(binding.prompt,signal);
      assert.deepEqual(data.ranking?.items.map(p=>p.item_id),['B','C','A']);
      assert.equal(data.analysis_calls.length,0);assert.deepEqual(data.new_memory_ids,[]);
      const result=await runConversationTurn({client:connection,prompt:binding.prompt,streamFn:model()});
      assert.equal(result.status,'ok');assert.match(result.finalText??'',/软属性信号1\.000/);
    }
    assert.deepEqual(bytes(resolve(folder,'memory')),before);
    assert.equal(readFileSync(trace,'utf8').trim().split('\n').length,4);
    assert.ok(!bytes(resolve(folder,'memory')).some(([n])=>n.endsWith('.jsonl')));
  } finally {await owner.close();rmSync(folder,{recursive:true,force:true});}
});
