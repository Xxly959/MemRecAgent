import assert from 'node:assert/strict';
import { test } from 'node:test';
import { createDeepSeekModel } from '../src/deepseek-model.js';
import { runCatalogTurn } from '../src/catalog-agent.js';
import { fixtureBundle, python } from './helpers.js';
import { normalizeContext } from '@earendil-works/pi-ai';

function sse(content: object, finish: string) {
  const chunks=[{choices:[{index:0,delta:{role:'assistant',...content},finish_reason:null}]},
    {choices:[{index:0,delta:{},finish_reason:finish}],usage:{prompt_tokens:100,
      completion_tokens:12,total_tokens:112,prompt_cache_hit_tokens:60,prompt_cache_miss_tokens:40}}];
  return new Response(chunks.map(c=>'data: '+JSON.stringify({id:'synthetic',object:'chat.completion.chunk',
    created:0,model:'deepseek-flash',...c})+'\n\n').join('')+'data: [DONE]\n\n',
    {headers:{'Content-Type':'text/event-stream'}});
}
test('真实Pi及SDK处理两次假HTTP，第二请求含真实工具回执', async () => {
  const f=fixtureBundle();let attempts=0;
  const transport:typeof fetch=async(input,init)=>{
    assert.equal(String(input),'https://api.deepseek.com/chat/completions');
    assert.equal(init?.method,'POST');assert.equal(init?.redirect,'error');
    const body=JSON.parse(String(init?.body));assert.equal(body.model,'deepseek-flash');
    assert.equal(body.temperature,0);assert.equal(body.max_tokens,512);
    if (++attempts===1) return sse({tool_calls:[{index:0,id:'live-t1',type:'function',
      function:{name:'catalog_status',arguments:'{}'}}]},'tool_calls');
    const message=body.messages.find((m:{role:string})=>m.role==='tool');
    assert.equal(message.tool_call_id,'live-t1');
    const health=JSON.parse(message.content);assert.equal(health.n_catalog_items,3);
    return sse({content:JSON.stringify({bundle_id:health.bundle_id,as_of_ms:health.as_of_ms,
      n_catalog_items:health.n_catalog_items})},'stop');
  };
  const provider=createDeepSeekModel({apiKey:'synthetic-placeholder',transport});
  try {
    const result=await runCatalogTurn({python,bundle:f.bundle,prompt:'查目录',
      streamFn:provider.streamFn,responseFormat:'catalog_json'});
    assert.equal(result.status,'ok');assert.equal(result.modelCalls,2);
    assert.equal(result.toolCalls,1);assert.match(result.finalText!,/3件商品/);
    const stats=provider.snapshot();assert.equal(stats.httpAttempts,2);
    assert.equal(stats.responses.length,2);assert.equal(stats.responses[1].usage.cacheRead,60);
    assert.ok(!JSON.stringify(stats).includes('synthetic-placeholder'));
  } finally {f.cleanup();}
});

test('429不自动重试，任意供应商错误不进入回执/日志', async () => {
  let calls=0;
  const provider=createDeepSeekModel({apiKey:'synthetic-placeholder',transport:async()=>{
    calls++;return new Response(JSON.stringify({error:{message:'private credential-like text'}}),
      {status:429,headers:{'Content-Type':'application/json'}});
  }});
  const result=await runCatalogTurn({python,bundle:'unused',prompt:'x',
    streamFn:provider.streamFn,responseFormat:'catalog_json'});
  assert.equal(result.status,'error');assert.equal(calls,1);
  assert.equal(provider.snapshot().httpAttempts,1);
  assert.ok(!JSON.stringify([result,provider.snapshot()]).includes('private credential'));
});

test('实际HTTP守卫阻止第三请求，缺Key不发起', async () => {
  assert.throws(()=>createDeepSeekModel({apiKey:''}), /missing_key/);
  let calls=0;
  const provider=createDeepSeekModel({apiKey:'synthetic-placeholder',transport:async()=>{
    calls++;return sse({content:'{}'},'stop');
  }});
  for (let i=0;i<3;i++) {
    const stream=await provider.streamFn({} as never,normalizeContext({messages:[]}),{});
    await stream.result();
  }
  assert.equal(calls,2);assert.equal(provider.snapshot().httpAttempts,2);
});

test('非思考配置必须进入实际wire，不能只设置本地reasoning元信息', async () => {
  let body:Record<string,unknown>={};
  const provider=createDeepSeekModel({apiKey:'synthetic-placeholder',transport:async(_input,init)=>{
    body=JSON.parse(String(init?.body));return sse({content:'{}'},'stop');
  }});
  const stream=await provider.streamFn({} as never,normalizeContext({messages:[]}),{});
  await stream.result();
  assert.deepEqual(body.thinking,{type:'disabled'});
  assert.equal(Object.hasOwn(body,'reasoning_effort'),false);
});

test('推荐增量具名工具选择必须进入实际wire，目录默认保持',async()=>{
  let selected='';
  const provider=createDeepSeekModel({apiKey:'synthetic-placeholder',
    toolName:'recommend_products',transport:async(_input,init)=>{
      selected=JSON.parse(String(init?.body)).tool_choice.function.name;
      return sse({content:'{}'},'stop');
    }});
  const stream=await provider.streamFn({} as never,normalizeContext({messages:[]}),{});
  await stream.result();assert.equal(selected,'recommend_products');
});

test('完整对话使用conversation_step而非绕过独立记忆分析',async()=>{
  let selected='';
  const provider=createDeepSeekModel({apiKey:'synthetic-placeholder',toolName:'conversation_step',
    transport:async(_input,init)=>{
      selected=JSON.parse(String(init?.body)).tool_choice.function.name;
      return sse({content:'{}'},'stop');
    }});
  const stream=await provider.streamFn({} as never,normalizeContext({messages:[]}),{});
  await stream.result();assert.equal(selected,'conversation_step');
});
