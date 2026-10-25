import assert from 'node:assert/strict';
import {test} from 'node:test';
import {execFileSync} from 'node:child_process';
import {mkdtempSync,rmSync,writeFileSync} from 'node:fs';
import {tmpdir} from 'node:os';
import {resolve} from 'node:path';
import {createAssistantMessageEventStream,type AssistantMessage} from '@earendil-works/pi-ai';
import type {StreamFn} from '@earendil-works/pi-agent-core';
import {runRecommendationTurn} from '../src/recommendation-agent.js';
import {python,root} from './helpers.js';

function fixture() {
  const path=mkdtempSync(resolve(tmpdir(),'memrec-rank-'));
  execFileSync(python,[resolve(root,'scripts/create_memory_recommendation_fixture.py'),path]);
  return {path,python,bundle:resolve(path,'bundle'),memoryRoot:resolve(path,'memory'),
    facts:resolve(path,'facts.json'),request:resolve(path,'request.json'),
    cleanup:()=>rmSync(path,{recursive:true,force:true})};
}

function syntheticModel(change='none',onFinal?:()=>void):StreamFn {
  let calls=0;
  return (_m,context)=>{
    const stream=createAssistantMessageEventStream();
    let content:AssistantMessage['content'];
    if (calls++===0) {
      content=change==='no-tool'?[{type:'text',text:'直接回答'}]:
        [{type:'toolCall',id:'rank-1',name:'recommend_products',arguments:change==='injected-user'?{user_id:'someone-else'}:{}}];
      if (change==='extra-tool') content.push({type:'toolCall',id:'rank-2',name:'recommend_products',arguments:{}});
    } else {
      const tool=context.messages.findLast(m=>m.role==='toolResult');
      assert.ok(tool?.role==='toolResult');
      const data=JSON.parse(tool.content.filter(c=>c.type==='text').map(c=>c.text).join(''));
      assert.equal(data.delivery_token,undefined);
      assert.equal(data.state_token,undefined);
      const ids=data.items.map((p:{item_id:string})=>p.item_id);
      if (change==='reorder') ids.reverse();
      if (change==='invent-item') ids.push('unavailable');
      onFinal?.();
      content=[{type:'text',text:JSON.stringify({item_ids:ids})}];
    }
    stream.end({role:'assistant',content,stopReason:content.some(c=>c.type==='toolCall')?'toolUse':'stop',
      api:'openai-completions',provider:'synthetic',model:'synthetic',timestamp:0,
      usage:{input:0,output:0,cacheRead:0,cacheWrite:0,totalTokens:0,
        cost:{input:0,output:0,cacheRead:0,cacheWrite:0,total:0}}});
    return stream;
  };
}

test('真实Pi工具读取磁盘偏好并忠实输出C/B，不泄露原始证据',async()=>{
  const f=fixture();
  try {
    const result=await runRecommendationTurn({...f,prompt:'合成：给我推荐手机',streamFn:syntheticModel()});
    assert.equal(result.status,'ok');
    assert.equal(result.modelCalls,2);assert.equal(result.toolCalls,1);assert.equal(result.validationCalls,1);
    assert.deepEqual(result.items?.map(p=>p.item_id),['C','B']);
    assert.match(result.finalText??'',/品牌C/);
    assert.match(result.finalText??'',/0\.750/);
    assert.ok(!JSON.stringify(result).includes('合成：我喜欢品牌C'));
  } finally {f.cleanup();}
});

for (const change of ['reorder','invent-item','injected-user','no-tool','extra-tool']) {
  test(`越权或无依据最终输出被拒绝 ${change}`,async()=>{
    const f=fixture();
    try {
      const result=await runRecommendationTurn({...f,prompt:'x',streamFn:syntheticModel(change)});
      assert.equal(result.status,'error');assert.equal(result.finalText,undefined);
      assert.equal(result.items,undefined);
      if (['injected-user','extra-tool'].includes(change)) assert.equal(result.toolCalls,0);
    } finally {f.cleanup();}
  });
}

test('第二次模型返回期间修改可信请求使旧结果禁止发布',async()=>{
  const f=fixture();
  try {
    const result=await runRecommendationTurn({...f,prompt:'x',streamFn:syntheticModel('none',()=>{
      writeFileSync(f.request,'{}');
    })});
    assert.equal(result.status,'error');assert.equal(result.code,'state_changed');
    assert.equal(result.finalText,undefined);assert.equal(result.items,undefined);
  } finally {f.cleanup();}
});

test('不响应AbortSignal的模型仍受整轮期限约束',async()=>{
  const f=fixture();
  try {
    const streamFn:StreamFn=()=>createAssistantMessageEventStream();
    const start=performance.now();
    const result=await runRecommendationTurn({...f,prompt:'x',streamFn,totalMs:20});
    assert.equal(result.code,'turn_timeout');assert.ok(performance.now()-start<1000);
  } finally {f.cleanup();}
});
