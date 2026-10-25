import { randomUUID } from 'node:crypto';
import { Agent, type StreamFn } from '@earendil-works/pi-agent-core';
import { Type, type AssistantMessage, type Model } from '@earendil-works/pi-ai';
import { BridgeClient, BridgeError, type CatalogHealth } from './bridge-client.js';
import { offlineModel, catalogSummary } from './offline-model.js';
import { strictJson } from './strict-json.js';

interface Limits { totalMs?: number; toolMs?: number; modelCalls?: number; toolCalls?: number }
interface Options { python:string; bundle:string; prompt:string; signal?:AbortSignal;
  limits?:Limits; streamFn?:StreamFn; responseFormat?:'text'|'catalog_json';
  model?:Model<'openai-completions'> }
export interface TurnResult {
  status:'ok'|'error'; code?:string; finalText?:string; modelCalls:number; toolCalls:number;
  events:{type:string; toolCallId?:string}[];
}
const model: Model<'openai-completions'> = {id:'offline',name:'offline',api:'openai-completions',
  provider:'offline',baseUrl:'',reasoning:false,input:['text'],contextWindow:65536,maxTokens:512,
  cost:{input:0,output:0,cacheRead:0,cacheWrite:0}};

export async function runCatalogTurn(options:Options):Promise<TurnResult> {
  const result:TurnResult = {status:'error',modelCalls:0,toolCalls:0,events:[]};
  if (options.responseFormat!==undefined && !['text','catalog_json'].includes(options.responseFormat)) {
    result.code='invalid_config';return result;
  }
  const limits = {totalMs:30000,toolMs:5000,modelCalls:2,toolCalls:2,...options.limits};
  for (const [key, max] of Object.entries({totalMs:30000,toolMs:5000,modelCalls:2,toolCalls:2})) {
    const value = limits[key as keyof typeof limits];
    if (!Number.isSafeInteger(value) || value <= 0 || value > max) {
      result.code='invalid_config'; return result;
    }
  }
  if (options.signal?.aborted) {result.code='cancelled';return result;}
  const controller = new AbortController();
  const client = new BridgeClient({python:options.python,bundle:options.bundle,
    requestId:randomUUID(),toolTimeoutMs:limits.toolMs});
  let agent: Agent;
  let stopped = false;
  let delivered = false;
  let health:CatalogHealth|undefined;
  let last:AssistantMessage|undefined;
  let resolveStop!:()=>void;
  const stop = new Promise<void>(resolve=>{resolveStop=resolve;});
  const fail = (code:string) => {
    if (stopped || delivered) return;
    stopped=true;result.code=code;
    controller.abort();agent?.abort();resolveStop();
  };
  const underlying = options.streamFn ?? offlineModel();
  const streamFn:StreamFn = (...args) => {
    if (stopped) throw new BridgeError(result.code!);
    if (result.modelCalls >= limits.modelCalls) {
      fail('model_budget');throw new BridgeError('model_budget');
    }
    result.modelCalls++;
    result.events.push({type:'model_request'});
    return underlying(...args);
  };
  const toolIds = new Set<string>();
  const systemPrompt=options.responseFormat==='catalog_json' ?
    '必须先调用catalog_status查询绑定目录。取得工具结果后，最终仅输出JSON对象，恰含bundle_id、as_of_ms、n_catalog_items三个字段，值逐项来自工具回执；不加Markdown或其他文字。不推荐商品、不猜实时库存。' :
    '仅查询绑定目录状态，依据真实工具回执回答。';
  agent = new Agent({initialState:{model:options.model ?? model,systemPrompt,
    tools:[{name:'catalog_status',label:'目录状态',description:'查询冻结目录状态，不推荐商品',
      parameters:Type.Object({}, {additionalProperties:false}),
      execute:async(id) => {
        if (stopped) return {content:[],details:{},isError:true};
        if (result.toolCalls >= limits.toolCalls) {
          fail('tool_budget');return {content:[],details:{},isError:true};
        }
        result.toolCalls++;
        try {
          const value=await client.request(id,controller.signal);
          if (stopped) return {content:[],details:{},isError:true};
          health=value;result.events.push({type:'bridge_receipt',toolCallId:id});
          return {content:[{type:'text' as const,text:JSON.stringify(value)}],details:{},
            structuredContent:{...value}};
        } catch (error) {
          fail(error instanceof BridgeError ? error.code : 'bridge_error');
          return {content:[{type:'text' as const,text:'tool_failed'}],details:{},isError:true};
        }
      }}]},streamFn,toolExecution:'sequential',identity:{attribution:null}});
  const unsubscribe=agent.subscribe(event=>{
    if (stopped) return;
    if (event.type==='tool_execution_start' || event.type==='tool_execution_end') {
      result.events.push({type:event.type,toolCallId:event.toolCallId});
    }
    if (event.type==='message_end' && event.message.role==='toolResult') {
      result.events.push({type:'tool_result',toolCallId:event.message.toolCallId});
    }
    if (event.type!=='message_end' || event.message.role!=='assistant') return;
    last=event.message;
    if (['error','aborted','length'].includes(last.stopReason)) {fail('model_failed');return;}
    const calls=last.content.filter(c=>c.type==='toolCall');
    if (calls.length > limits.toolCalls-result.toolCalls) {fail('tool_budget');return;}
    for (const call of calls) {
      if (call.name!=='catalog_status' || !call.arguments || Array.isArray(call.arguments)
        || typeof call.arguments!=='object' || Object.keys(call.arguments).length
        || !call.id || toolIds.has(call.id)) {fail('invalid_tool');return;}
      toolIds.add(call.id);
    }
  });
  const cancel=()=>fail('cancelled');
  const unsubscribeFailure=client.subscribeFailure(fail);
  options.signal?.addEventListener('abort',cancel,{once:true});
  const timer=setTimeout(()=>fail('turn_timeout'),limits.totalMs);
  try {
    const work=agent.prompt(options.prompt).catch(()=>fail('agent_failed'));
    await Promise.race([work,stop]);
    if (!stopped) {
      const text=last?.content.filter(c=>c.type==='text').map(c=>c.text).join('');
      let answerMatches=!!health && text===catalogSummary(health);
      if (options.responseFormat==='catalog_json') {
        answerMatches=false;
        try {
          const value=strictJson(text ?? '') as Record<string,unknown>;
          answerMatches=!!health && !!value && typeof value==='object' && !Array.isArray(value)
            && Object.keys(value).sort().join()==='as_of_ms,bundle_id,n_catalog_items'
            && value.bundle_id===health.bundle_id && value.as_of_ms===health.as_of_ms
            && value.n_catalog_items===health.n_catalog_items;
        } catch { /* 非法/重复字段不修补，保持最终拒绝。 */ }
      }
      if (!health || !client.healthy || !last || last.stopReason!=='stop'
        || last.content.some(c=>c.type==='toolCall') || !answerMatches) fail('final_guard');
      else {
        delivered=true;result.status='ok';result.finalText=options.responseFormat==='catalog_json' ?
          catalogSummary(health) : text;
        result.events.push({type:'delivered'});
      }
    }
  } finally {
    clearTimeout(timer);options.signal?.removeEventListener('abort',cancel);
    unsubscribe();unsubscribeFailure();controller.abort();agent.abort();await client.close();
  }
  // 返回深快照，已经超时的依赖即便迟到也不能更改已交付结果。
  return structuredClone(result);
}
