// 供应商独立于记忆/推荐：固定官方端点、禁重试、按实际fetch计数。
import { createHash } from 'node:crypto';
import type { Model } from '@earendil-works/pi-ai';
import { stream as streamCompletions } from '@earendil-works/pi-ai/api/openai-completions';
import type { StreamFn } from '@earendil-works/pi-agent-core';

const model:Model<'openai-completions'>={id:'deepseek-flash',name:'deepseek-flash',
  api:'openai-completions',provider:'deepseek',baseUrl:'https://api.deepseek.com',reasoning:false,
  input:['text'],contextWindow:65536,maxTokens:512,
  cost:{input:0,output:0,cacheRead:0,cacheWrite:0},
  compat:{supportsStore:false,supportsDeveloperRole:false,maxTokensField:'max_tokens'}};
interface RequestRecord {attempt:number; bodySha256:string; elapsedMs:number; status?:number}
interface ResponseRecord {stopReason:string; usageReported:boolean; usage:{input:number;output:number;cacheRead:number;
  cacheWrite:number;totalTokens:number}; content?:unknown}

export function createDeepSeekModel(config:{apiKey:string;transport?:typeof fetch;captureSynthetic?:boolean;
  toolName?:'catalog_status'|'recommend_products'|'conversation_step'}) {
  if (!config.apiKey.trim()) throw new Error('missing_key');
  const toolName=config.toolName??'catalog_status';
  if (!['catalog_status','recommend_products','conversation_step'].includes(toolName)) throw new Error('invalid_tool');
  const transport=config.transport ?? globalThis.fetch;
  let httpAttempts=0;
  const requests:RequestRecord[]=[],responses:ResponseRecord[]=[];
  const guardedFetch:typeof fetch=async(input,init)=>{
    const url=input instanceof Request ? input.url : String(input);
    if (url!=='https://api.deepseek.com/chat/completions' || init?.method!=='POST')
      throw new Error('endpoint_rejected');
    if (init.signal?.aborted) throw new Error('cancelled');
    if (httpAttempts>=2) throw new Error('http_budget');
    const record:RequestRecord={attempt:++httpAttempts,elapsedMs:0,
      bodySha256:createHash('sha256').update(typeof init.body==='string'?init.body:'').digest('hex')};
    requests.push(record);
    const start=performance.now();
    try {
      const response=await transport(input,{...init,redirect:'error'});
      record.status=response.status;
      return response;
    } finally {record.elapsedMs=performance.now()-start;}
  };
  const streamFn:StreamFn=(_requestedModel,context,options)=>{
    const stream=streamCompletions(model,context,{...options,apiKey:config.apiKey,fetch:guardedFetch,
      maxRetries:0,temperature:0,maxTokens:512,timeoutMs:30000,
      onPayload:payload=>{
        // 本地reasoning元信息不等于DeepSeek wire开关，必须显式发送。
        const {reasoning_effort:_effort,...body}=payload as Record<string,unknown>;
        return {...body,thinking:{type:'disabled'}};
      },
      toolChoice:context.messages.some(m=>m.role==='toolResult') ? 'none' :
        {type:'function',function:{name:toolName}}});
    void stream.result().then(message=>{
      const u=message.usage;
      const record:ResponseRecord={stopReason:message.stopReason,usageReported:u.totalTokens>0,usage:{input:u.input,
        output:u.output,cacheRead:u.cacheRead,cacheWrite:u.cacheWrite,totalTokens:u.totalTokens}};
      // 只在受控合成实验显式保存模型内容；错误消息可能含供应商原文，不保存。
      if (config.captureSynthetic && ['stop','toolUse'].includes(message.stopReason))
        record.content=structuredClone(message.content);
      responses.push(record);
    }).catch(()=>{ /* 不复制任意错误对象到实验或默认日志。 */ });
    return stream;
  };
  return {streamFn,model:structuredClone(model),snapshot:()=>structuredClone({httpAttempts,requests,responses})};
}
