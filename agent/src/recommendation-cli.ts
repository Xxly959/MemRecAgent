// 默认固定假模型演示；只有显式--live才读取进程Key并允许最多两次真实HTTP。
import {isAbsolute} from 'node:path';
import {createAssistantMessageEventStream,type AssistantMessage} from '@earendil-works/pi-ai';
import type {StreamFn} from '@earendil-works/pi-agent-core';
import {createDeepSeekModel} from './deepseek-model.js';
import {runRecommendationTurn} from './recommendation-agent.js';

function offline():StreamFn {
  let calls=0;
  return (_model,context)=>{
    const stream=createAssistantMessageEventStream();
    let content:AssistantMessage['content'];
    if (calls++===0) content=[{type:'toolCall',id:'offline-rank-1',name:'recommend_products',arguments:{}}];
    else {
      const tool=context.messages.findLast(m=>m.role==='toolResult');
      if (!tool||tool.role!=='toolResult'||tool.isError) throw new Error('receipt_missing');
      const value=JSON.parse(tool.content.filter(c=>c.type==='text').map(c=>c.text).join(''));
      content=[{type:'text',text:JSON.stringify({item_ids:value.items.map((p:{item_id:string})=>p.item_id)})}];
    }
    stream.end({role:'assistant',content,stopReason:calls===1?'toolUse':'stop',
      api:'openai-completions',provider:'offline',model:'offline',timestamp:0,
      usage:{input:0,output:0,cacheRead:0,cacheWrite:0,totalTokens:0,
        cost:{input:0,output:0,cacheRead:0,cacheWrite:0,total:0}}});
    return stream;
  };
}

const args=process.argv.slice(2),live=args[0]==='--live';
if (live) args.shift();
const values=new Map<string,string>();
let valid=args.length===10;
for (let i=0;i<args.length;i+=2) {
  const [key,value]=[args[i],args[i+1]];
  if (!['--python','--bundle','--memory-root','--facts','--request'].includes(key)
    || !value || !isAbsolute(value) || values.has(key)) valid=false;
  values.set(key,value);
}
const key=live?process.env.DEEPSEEK_API_KEY:undefined;
if (!valid || (live&&!key?.trim())) {
  process.stdout.write(JSON.stringify({status:'error',code:!valid?'invalid_config':'missing_key',httpAttempts:0})+'\n');
  process.exitCode=2;
} else {
  const provider=live?createDeepSeekModel({apiKey:key!,toolName:'recommend_products'}):undefined;
  const controller=new AbortController(),cancel=()=>controller.abort();
  process.once('SIGINT',cancel);process.once('SIGTERM',cancel);
  try {
    const result=await runRecommendationTurn({python:values.get('--python')!,bundle:values.get('--bundle')!,
      memoryRoot:values.get('--memory-root')!,facts:values.get('--facts')!,request:values.get('--request')!,
      prompt:'请按已绑定的当前人物和请求推荐商品，只依据工具结果回答。',signal:controller.signal,
      streamFn:provider?.streamFn??offline(),model:provider?.model});
    process.stdout.write(JSON.stringify({status:result.status,result,
      provider:provider?.snapshot()??{httpAttempts:0}})+'\n');
    process.exitCode=result.status==='ok'?0:2;
  } catch {
    process.stdout.write(JSON.stringify({status:'error',code:'host_failed',httpAttempts:provider?.snapshot().httpAttempts??0})+'\n');
    process.exitCode=2;
  } finally {process.off('SIGINT',cancel);process.off('SIGTERM',cancel);}
}
