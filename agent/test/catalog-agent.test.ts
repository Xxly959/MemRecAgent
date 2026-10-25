import assert from 'node:assert/strict';
import { test } from 'node:test';
import { mkdtempSync, writeFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import { createAssistantMessageEventStream, type AssistantMessage } from '@earendil-works/pi-ai';
import type { StreamFn } from '@earendil-works/pi-agent-core';
import { runCatalogTurn } from '../src/catalog-agent.js';
import { fixtureBundle, python, root } from './helpers.js';

function response(content: AssistantMessage['content'], stopReason: AssistantMessage['stopReason']='stop') {
  const stream = createAssistantMessageEventStream();
  const message: AssistantMessage = {role:'assistant', content, api:'openai-completions',
    provider:'offline', model:'offline', timestamp:0, stopReason,
    usage:{input:0, output:0, cacheRead:0, cacheWrite:0, totalTokens:0,
      cost:{input:0,output:0,cacheRead:0,cacheWrite:0,total:0}}};
  stream.end(message);
  return stream;
}
const tool = (id='t1', name='catalog_status', args={}) =>
  ({type:'toolCall' as const, id, name, arguments:args});

test('真实Pi两轮消费真实Python工具回执', async () => {
  const f=fixtureBundle();
  try {
    const result = await runCatalogTurn({python,bundle:f.bundle,prompt:'目录可用吗'});
    assert.equal(result.status,'ok');
    assert.equal(result.modelCalls,2); assert.equal(result.toolCalls,1);
    assert.match(result.finalText!, /目录就绪.*3件商品/);
    const events = result.events.map(e=>e.type);
    assert.ok(events.indexOf('tool_execution_start') < events.indexOf('bridge_receipt'));
    assert.ok(events.indexOf('bridge_receipt') < events.lastIndexOf('model_request'));
    assert.equal(events.at(-1),'delivered');
  } finally {f.cleanup();}
});

for (const [name, calls, code] of [
  ['三工具批次',[tool('a'),tool('b'),tool('c')],'tool_budget'],
  ['未知工具',[tool('a','shell')],'invalid_tool'],
  ['额外参数',[tool('a','catalog_status',{path:'private'})],'invalid_tool'],
  ['重复工具ID',[tool('a'),tool('a')],'invalid_tool'],
] as const) {
  test(`完整批次执行前拒绝${name}`, async () => {
    const result = await runCatalogTurn({python,bundle:'unused',prompt:'x',
      streamFn:()=>response([...calls],'toolUse')});
    assert.equal(result.status,'error'); assert.equal(result.code,code);
    assert.equal(result.toolCalls,0); assert.equal(result.modelCalls,1);
    assert.equal(result.finalText,undefined);
  });
}

test('工具失败停止，不再请求模型', async () => {
  const result = await runCatalogTurn({python,bundle:'/missing/bundle',prompt:'x'});
  assert.equal(result.status,'error'); assert.equal(result.modelCalls,1);
  assert.equal(result.toolCalls,1); assert.equal(result.finalText,undefined);
});

test('第三模型前停止且不交付未完成答案', async () => {
  const f=fixtureBundle();let count=0;
  const streamFn: StreamFn=()=>response([tool(`t${++count}`)],'toolUse');
  try {
    const result=await runCatalogTurn({python,bundle:f.bundle,prompt:'x',streamFn});
    assert.equal(count,2); assert.equal(result.code,'model_budget');
    assert.equal(result.toolCalls,2); assert.equal(result.finalText,undefined);
  } finally {f.cleanup();}
});

test('累计第三工具在执行前停止', async () => {
  const f=fixtureBundle();let count=0;
  try {
    const result=await runCatalogTurn({python,bundle:f.bundle,prompt:'x',streamFn:()=>
      response(count++===0 ? [tool('a')] : [tool('b'),tool('c')],'toolUse')});
    assert.equal(result.code,'tool_budget');assert.equal(result.toolCalls,1);
  } finally {f.cleanup();}
});

for (const reason of ['length','error','aborted'] as const) {
  test(`助手${reason}不能当最终成功`, async () => {
    const result=await runCatalogTurn({python,bundle:'unused',prompt:'x',
      streamFn:()=>response([{type:'text',text:'伪成功'}],reason)});
    assert.equal(result.status,'error');assert.equal(result.finalText,undefined);
  });
}

test('模型不配合abort也由总截止结算', async () => {
  const start=performance.now();
  const result=await runCatalogTurn({python,bundle:'unused',prompt:'x',limits:{totalMs:40},
    streamFn:()=>new Promise(()=>{})});
  assert.equal(result.code,'turn_timeout');assert.ok(performance.now()-start<1000);
});

test('取消后的迟到工具提议不执行，下一轮同ID仍独立', async () => {
  const controller=new AbortController(); let late=false;
  const first=runCatalogTurn({python,bundle:'unused',prompt:'x',signal:controller.signal,
    streamFn:async()=>{await new Promise(r=>setTimeout(r,70));late=true;return response([tool()],'toolUse');}});
  setTimeout(()=>controller.abort(),10);
  assert.equal((await first).code,'cancelled');
  await new Promise(r=>setTimeout(r,100)); assert.equal(late,true);
  const f=fixtureBundle();
  try {assert.equal((await runCatalogTurn({python,bundle:f.bundle,prompt:'x'})).status,'ok');}
  finally {f.cleanup();}
});

test('预取消零调用，超出默认上限配置拒绝', async () => {
  const c=new AbortController();c.abort();
  const result=await runCatalogTurn({python,bundle:'unused',prompt:'x',signal:c.signal});
  assert.equal(result.modelCalls,0);assert.equal(result.toolCalls,0);
  assert.equal(result.code,'cancelled');
  assert.equal((await runCatalogTurn({python,bundle:'unused',prompt:'x',
    limits:{totalMs:30001}})).code,'invalid_config');
});

test('工具期限覆盖Python启动，失败后不发第二模型', async () => {
  const result=await runCatalogTurn({python,bundle:'unused',prompt:'x',limits:{toolMs:1}});
  assert.equal(result.code,'tool_timeout');assert.equal(result.modelCalls,1);
  assert.equal(result.finalText,undefined);
});

for (const text of ['', '无工具依据的成功声明']) {
  test('没有实际回执不能交付助手文本', async () => {
    const result=await runCatalogTurn({python,bundle:'unused',prompt:'x',
      streamFn:()=>response([{type:'text',text}])});
    assert.equal(result.code,'final_guard');assert.equal(result.finalText,undefined);
  });
}

for (const startupDelay of [0,750]) {
for (const [mode,code] of [['late-duplicate','bridge_protocol'],['late-eof','bridge_eof']]) {
  test(`首回执成功后的${mode}立即停止第二模型${startupDelay?'（慢启动）':''}`, async () => {
    const dir=mkdtempSync(resolve(tmpdir(),'memrec-late-'));
    const executable=resolve(dir,'child.mjs');
    // 用可信测试解释器注入真实坏管道，不改变宿主或客户端私有字段。
    const target=pathToFileURL(resolve(root,'agent/test/fixtures/bridge-child.mjs')).href;
    writeFileSync(executable,`#!${process.execPath}\nprocess.argv[2]=${JSON.stringify(mode)};\n${startupDelay?`await new Promise(r=>setTimeout(r,${startupDelay}));\n`:''}await import(${JSON.stringify(target)});\n`,{mode:0o700});
    let calls=0,aborted=false,secondStarted:number|undefined;
    try {
      const result=await runCatalogTurn({python:executable,bundle:'unused',prompt:'x',
        limits:{totalMs:startupDelay?2000:1000},streamFn:(_m,_c,options)=>{
          if (calls++===0) return response([tool()],'toolUse');
          secondStarted=performance.now();
          return new Promise(()=>{options?.signal?.addEventListener('abort',()=>{aborted=true;},{once:true});});
        }});
      assert.equal(result.code,code);assert.equal(aborted,true);
      assert.equal(calls,2);
      assert.notEqual(secondStarted,undefined);
      // 已知故障的停止耗时从第二模型开始计，不把解释器启动混入。
      // 原用例整轮1000ms与故障响应700ms保持；慢启动另用2000ms整轮预算。
      assert.ok(performance.now()-secondStarted!<700,'已知故障不应等总超时');
      assert.equal(result.finalText,undefined);
    } finally {rmSync(dir,{recursive:true,force:true});}
  });
}
}
