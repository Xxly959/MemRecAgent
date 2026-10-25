import assert from 'node:assert/strict';
import {test} from 'node:test';
import {createAssistantMessageEventStream,type AssistantMessage} from '@earendil-works/pi-ai';
import type {StreamFn} from '@earendil-works/pi-agent-core';
import {runConversationTurn} from '../src/conversation-agent.js';

test('状态撤销后不把旧工具历史发送给第二次模型',async()=>{
  let requests=0,exposed=false;
  const fixed=model({reply:'自然答复。',item_ids:[]});
  const client={step:async()=>({response_action:'answer',clarification:null,new_memory_ids:[],
    context_memories:[],analysis_calls:[],state_token:'snapshot',ranking:null,
    conversation_context:[{turn_id:'old',role:'user',content:'PRIVATE_HISTORY_SENTINEL'}]}),
    validate:async()=>{throw new Error('revoked');},observe:async()=>{},close:async()=>{}};
  const streamFn:StreamFn=(...args)=>{
    requests++;exposed ||= JSON.stringify(args[1]).includes('PRIVATE_HISTORY_SENTINEL');
    return fixed(...args);
  };
  const result=await runConversationTurn({client,prompt:'继续说说手机。',streamFn});
  assert.equal(result.status,'error');
  assert.equal(requests,1);assert.equal(result.modelCalls,1);assert.equal(exposed,false);
  assert.equal(result.code,'conversation_state');
});

test('等待新鲜度验证超时后，迟到验证不得重新启动模型',async()=>{
  let release!:()=>void,requests=0;
  const pending=new Promise<void>(resolve=>{release=resolve;});
  const fixed=model({reply:'自然答复。',item_ids:[]});
  const client={step:async()=>({response_action:'answer',clarification:null,new_memory_ids:[],
    context_memories:[],analysis_calls:[],state_token:'snapshot',ranking:null}),
    validate:async()=>pending,observe:async()=>{},close:async()=>{}};
  const streamFn:StreamFn=(...args)=>{requests++;return fixed(...args);};
  const result=await runConversationTurn({client,prompt:'继续说说手机。',streamFn,totalMs:20});
  assert.equal(result.status,'error');assert.equal(result.code,'turn_timeout');
  assert.equal(requests,1);
  release();await new Promise<void>(resolve=>setImmediate(resolve));
  assert.equal(requests,1);
});

function model(response:unknown):StreamFn {
  let n=0;
  return ()=>{
    const stream=createAssistantMessageEventStream();
    const content:AssistantMessage['content']=n++===0?
      [{type:'toolCall',id:'memory-tool',name:'conversation_step',arguments:{}}]:
      [{type:'text',text:JSON.stringify(response)}];
    stream.end({role:'assistant',content,stopReason:n===1?'toolUse':'stop',api:'openai-completions',
      provider:'synthetic',model:'synthetic',timestamp:0,usage:{input:0,output:0,cacheRead:0,cacheWrite:0,
        totalTokens:0,cost:{input:0,output:0,cacheRead:0,cacheWrite:0,total:0}}});
    return stream;
  };
}

test('主模型获知本次属性不能冒充长期保存，品牌ID不表示预算已保存',async()=>{
  const client={step:async()=>({response_action:'answer',clarification:null,new_memory_ids:['brand-only'],
    context_memories:[],analysis_calls:[],state_token:'snapshot',ranking:null}),
    validate:async()=>{},observe:async()=>{},close:async()=>{}};
  let seen='';
  const fixed=model({reply:'本次按三千预算筛选。',item_ids:[]});
  const streamFn:StreamFn=(...args)=>{
    seen=args[1].messages.filter(m=>m.role==='system').map(m=>typeof m.content==='string'?m.content:JSON.stringify(m.content)).join('\n');
    return fixed(...args);
  };
  const result=await runConversationTurn({client,prompt:'这次预算三千',streamFn});
  assert.equal(result.status,'ok');
  assert.match(seen,/本次属性.*不能声称已长期保存预算/);
  assert.match(seen,/attribute:.*长期属性/);
  assert.match(seen,/memory:.*品牌/);
});

test('Pi实际循环消费独立记忆子任务与程序排名，不能调换顺序',async()=>{
  for (const ids of [['C','B'],['B','C']]) {
    const events:string[]=[];
    const client={
      step:async(text:string)=>{
        assert.equal(text,'给我推荐手机');events.push('step');
        return {response_action:'recommend',clarification:null,new_memory_ids:[],context_memories:[],
          analysis_calls:[],state_token:'snapshot',ranking:{items:[
            {item_id:'C',entity:'品牌C',title:null,score:.75,base_score:8,preference_signal:1,signals:[]},
            {item_id:'B',entity:'品牌B',title:'商品B',score:.5,base_score:9,preference_signal:0,signals:[]}]}};
      },
      validate:async()=>{events.push('validate');},
      observe:async()=>{events.push('observe');},
      close:async()=>{events.push('close');},
    };
    const result=await runConversationTurn({client,prompt:'给我推荐手机',streamFn:model({reply:'',item_ids:ids})});
    assert.equal(result.status,ids[0]==='C'?'ok':'error');
    assert.deepEqual(events,ids[0]==='C'?['step','validate','validate']:['step','validate','close']);
    if (result.status==='ok') await client.observe();
    if (result.status==='ok') assert.match(result.finalText??'',/品牌C/);
    else assert.equal(result.finalText,undefined);
  }
});

test('歧义必须呈现宿主原问题并登记实际助手回复，不能被模型改写',async()=>{
  const question='这条喜好属于谁？请明确人物对象。';
  const client={step:async()=>({response_action:'clarify',clarification:question,new_memory_ids:[],
    context_memories:[],analysis_calls:[],state_token:'snapshot',ranking:null}),
    validate:async()=>{},observe:async(text:string)=>{assert.equal(text,question);},close:async()=>{}};
  const result=await runConversationTurn({client,prompt:'她喜欢品牌C',streamFn:model({reply:question,item_ids:[]})});
  assert.equal(result.status,'ok');assert.equal(result.finalText,question);
  await client.observe(result.finalText!);
});

test('普通咨询可以自然回答；没有推荐意图不能私自输出商品排名',async()=>{
  const client={step:async()=>({response_action:'answer',clarification:null,new_memory_ids:[],
    context_memories:[],analysis_calls:[],state_token:'snapshot',ranking:null}),
    validate:async()=>{},observe:async()=>{},close:async()=>{}};
  const result=await runConversationTurn({client,prompt:'你怎么看品牌C',
    streamFn:model({reply:'你更关注它的哪些体验？我目前没有实时商品资料。',item_ids:[]})});
  assert.equal(result.status,'ok');assert.match(result.finalText??'',/关注/);
});

test('子任务失败仍报告尝试与已保存ID，不把付费失败当零成本',async()=>{
  const client={step:async()=>{throw Object.assign(new Error('safe'),{code:'analysis_failed',
    metadata:{analysis_calls:[{stage:'draft',status:'provider_error'}],new_memory_ids:['already-saved']}});},
    validate:async()=>{},observe:async()=>{},close:async()=>{}};
  const result=await runConversationTurn({client,prompt:'x',streamFn:model({reply:'',item_ids:[]})});
  assert.equal(result.code,'analysis_failed');assert.equal(result.analysisCalls.length,1);
  assert.deepEqual(result.memoryIds,['already-saved']);assert.equal(result.finalText,undefined);
});

test('持续连接迟到故障立即取消等待模型，返回快照不被迟到回复修改',async()=>{
  let listener:(code:string)=>void=()=>{},calls=0;
  const client={step:async()=>({response_action:'answer',clarification:null,new_memory_ids:[],
    context_memories:[],analysis_calls:[],state_token:'snapshot',ranking:null}),
    validate:async()=>{},observe:async()=>{},close:async()=>{},
    subscribeFailure:(fn:(code:string)=>void)=>{listener=fn;return ()=>{};}};
  const first=model({reply:'正常回答',item_ids:[]});
  const streamFn:StreamFn=(...args)=>{
    if (calls++===0) return first(...args);
    setTimeout(()=>listener('conversation_eof'),10);
    return createAssistantMessageEventStream();
  };
  const start=performance.now();
  const result=await runConversationTurn({client,prompt:'x',streamFn,totalMs:500});
  assert.equal(result.code,'conversation_eof');assert.ok(performance.now()-start<300);
});

test('进程中断无回执时保存与子任务用量必须标未知',async()=>{
  const client={step:async()=>{throw Object.assign(new Error('safe'),{code:'conversation_eof'});},
    validate:async()=>{},observe:async()=>{},close:async()=>{}};
  const result=await runConversationTurn({client,prompt:'x',streamFn:model({reply:'',item_ids:[]})});
  assert.equal(result.code,'conversation_eof');
  assert.equal(result.analysisStatus,'unknown');assert.equal(result.memoryCommitStatus,'unknown');
});
