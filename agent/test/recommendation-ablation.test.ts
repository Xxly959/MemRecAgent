// 固定模型只验证端到端接线：分析→实际磁盘→新会话Pi四组，不计语义收益。
import assert from 'node:assert/strict';
import {test} from 'node:test';
import {execFileSync} from 'node:child_process';
import {existsSync,mkdtempSync,readFileSync,readdirSync,rmSync,writeFileSync} from 'node:fs';
import {tmpdir} from 'node:os';
import {resolve} from 'node:path';
import {createAssistantMessageEventStream,type AssistantMessage} from '@earendil-works/pi-ai';
import type {StreamFn} from '@earendil-works/pi-agent-core';
import {runRecommendationTurn} from '../src/recommendation-agent.js';
import {python,root} from './helpers.js';

function memoryBytes(path:string):Record<string,string>{
  if(!existsSync(path)) return {};
  return Object.fromEntries(readdirSync(path,{recursive:true,withFileTypes:true})
    .filter(entry=>entry.isFile())
    .map(entry=>{
      const file=resolve(entry.parentPath,entry.name);
      return [file.slice(path.length),readFileSync(file).toString('base64')];
    }).sort(([a],[b])=>a.localeCompare(b)));
}

function scripted():StreamFn {
  let calls=0;
  return (_m,context)=>{
    const stream=createAssistantMessageEventStream();
    let content:AssistantMessage['content'];
    if(calls++===0) content=[{type:'toolCall',id:'rank',name:'recommend_products',arguments:{}}];
    else{
      const receipt=context.messages.findLast(m=>m.role==='toolResult');
      assert.ok(receipt?.role==='toolResult');
      const raw=JSON.parse(receipt.content.filter(c=>c.type==='text').map(c=>c.text).join(''));
      content=[{type:'text',text:JSON.stringify({item_ids:raw.items.map((i:{item_id:string})=>i.item_id)})}];
    }
    stream.end({role:'assistant',content,stopReason:calls===1?'toolUse':'stop',api:'openai-completions',
      provider:'fixed',model:'fixed',timestamp:0,usage:{input:0,output:0,cacheRead:0,cacheWrite:0,totalTokens:0,
      cost:{input:0,output:0,cacheRead:0,cacheWrite:0,total:0}}});
    return stream;
  };
}

for(const invalid of [false,true]){
  test(`真实Pi四组消费实际${invalid?'失败':'成功'}分析留下的磁盘状态`,async()=>{
    const path=mkdtempSync(resolve(tmpdir(),'memrec-auto-four-'));
    try{
      execFileSync(python,[resolve(root,'tests/ablation_story_helper.py'),path,...(invalid?['--invalid']:[])]);
      const history=JSON.parse(readFileSync(resolve(path,'history-trace.json'),'utf8'));
      assert.equal(history.actual_api_calls,0);
      assert.equal(history.stored_records,invalid?0:2);
      assert.equal(history.calls.length,2);
      const beforeMemory=memoryBytes(resolve(path,'memory'));
      if(!invalid) assert.ok(Object.keys(beforeMemory).some(name=>name.endsWith('.jsonl')));
      const expected=invalid?{G00:['A','B','C'],G10:['A','B','C'],G01:['A','C','B'],G11:['A','C','B']}:
        {G00:['A','B','C'],G10:['B','C'],G01:['A','C','B'],G11:['B','C']};
      for(const group of ['G00','G10','G01','G11'] as const){
        const result=await runRecommendationTurn({python,bundle:resolve(path,'bundle'),
          memoryRoot:resolve(path,'memory'),facts:resolve(path,'facts.json'),request:resolve(path,`request-${group}.json`),
          prompt:'新会话为绑定的当前用户推荐手机。',streamFn:scripted()});
        assert.equal(result.status,'ok');
        assert.deepEqual(result.items?.map(i=>i.item_id),expected[group]);
        assert.equal(result.modelCalls,2);assert.equal(result.toolCalls,1);assert.equal(result.validationCalls,1);
        if(invalid||group==='G00'||group==='G01') assert.ok(result.items?.every(i=>i.signals.length===0));
      }
      // 推荐阶段保持只读，不能为了满足对照预期补入标准答案。
      assert.deepEqual(memoryBytes(resolve(path,'memory')),beforeMemory);
    }finally{rmSync(path,{recursive:true,force:true});}
  });
}

test('真实Pi工具在喜欢品牌B时仍以本次预算过滤，只返回C',async()=>{
  const path=mkdtempSync(resolve(tmpdir(),'memrec-budget-pi-'));
  try{
    execFileSync(python,[resolve(root,'tests/ablation_story_helper.py'),path]);
    const factsPath=resolve(path,'facts.json');
    const facts=JSON.parse(readFileSync(factsPath,'utf8'));
    for(const row of facts.facts) row.attributes=[{attribute:'price_cny',
      value:row.item_id==='C'?2000:4000,unit:'CNY',evidence_ref:'synthetic:price',available_at:row.available_at}];
    writeFileSync(factsPath,JSON.stringify(facts));
    const requestPath=resolve(path,'request-G11.json');
    const request=JSON.parse(readFileSync(requestPath,'utf8'));
    request.request_id='budget-request';
    request.constraints=[{user_id:'u',session_id:'S2',beneficiary_id:'self',request_id:'budget-request',
      attribute:'price_cny',operator:'<=',value:3000,unit:'CNY',evidence_ref:'synthetic:budget',
      available_at:facts.facts[0].available_at}];
    writeFileSync(requestPath,JSON.stringify(request));
    const before=memoryBytes(resolve(path,'memory'));
    const result=await runRecommendationTurn({python,bundle:resolve(path,'bundle'),
      memoryRoot:resolve(path,'memory'),facts:factsPath,request:requestPath,
      prompt:'按宿主绑定的本次三千预算推荐手机。',streamFn:scripted()});
    assert.equal(result.status,'ok');
    assert.deepEqual(result.items?.map(i=>i.item_id),['C']);
    assert.equal(result.modelCalls,2);assert.equal(result.toolCalls,1);assert.equal(result.validationCalls,1);
    assert.deepEqual(memoryBytes(resolve(path,'memory')),before);
  }finally{rmSync(path,{recursive:true,force:true});}
});
