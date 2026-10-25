import assert from 'node:assert/strict';
import {test} from 'node:test';
import {mkdtempSync,writeFileSync,rmSync} from 'node:fs';
import {resolve} from 'node:path';
import {tmpdir} from 'node:os';
import {ConversationClient} from '../src/conversation-client.js';
import {runConversationTurn} from '../src/conversation-agent.js';
import {createAssistantMessageEventStream,type AssistantMessage} from '@earendil-works/pi-ai';
import type {StreamFn} from '@earendil-works/pi-agent-core';
import {python,root} from './helpers.js';

test('真实持续Python实例完成两用户轮及助手登记，失败回执不复试',async()=>{
  const folder=mkdtempSync(resolve(tmpdir(),'memrec-conversation-'));
  const executable=resolve(folder,'worker.py');
  writeFileSync(executable,`#!${python}\nimport sys,json\nfrom pathlib import Path\nsys.path.insert(0,${JSON.stringify(resolve(root,'tests'))})\nfrom test_conversation_products import session\nfrom test_conversation_bridge import request\nfrom test_memory_analyst_host import FixedStages\nfrom test_memory_interaction import turn,value\nfrom memrecagent.conversation_bridge import ConversationBridge\nfrom memrecagent.agent_bridge import serve_bridge\nhost=session(Path(${JSON.stringify(folder)}))\nresponses=[]\ndef complete(messages):\n data=json.loads(messages[-1]['content'])\n turns=data['turns']\n current=turns[-1]\n from types import SimpleNamespace\n t=SimpleNamespace(turn_id=current['turn_id'],content=current['content'])\n raw=value(t,entity='品牌C',relation='prefer') if '喜欢' in t.content else value(t,'recommend')\n from test_memory_analyst import reply\n return reply(raw)\nserve_bridge(ConversationBridge(host,complete=complete),sys.stdin.buffer,sys.stdout.buffer)\n`,{mode:0o700});
  const client=new ConversationClient({python:executable,bundle:'/unused',memoryRoot:'/unused',facts:'/unused',
    userId:'u',sessionId:'s',apiKey:'synthetic-placeholder'});
  try {
    const first=await client.step('我长期喜欢品牌C',new AbortController().signal);
    assert.equal(first.new_memory_ids.length,1);
    await client.observe('已记住你明确表达的本人偏好。');
    const second=await client.step('给我推荐手机',new AbortController().signal);
    assert.deepEqual(second.ranking?.items.map(p=>p.item_id),['A','C','B']);
    await client.validate(second.state_token,new AbortController().signal);
    let calls=0;
    const streamFn:StreamFn=(_model,context)=>{
      const stream=createAssistantMessageEventStream();
      let content:AssistantMessage['content'];
      if (calls++===0) content=[{type:'toolCall',id:'real-pipeline',name:'conversation_step',arguments:{}}];
      else {
        const tool=context.messages.findLast(m=>m.role==='toolResult');
        assert.ok(tool?.role==='toolResult');
        const data=JSON.parse(tool.content.filter(c=>c.type==='text').map(c=>c.text).join(''));
        assert.ok(data.conversation_context.some((t:{role:string})=>t.role==='assistant'));
        content=[{type:'text',text:JSON.stringify({reply:'',item_ids:data.ranking.items.map((p:{item_id:string})=>p.item_id)})}];
      }
      stream.end({role:'assistant',content,stopReason:calls===1?'toolUse':'stop',api:'openai-completions',provider:'synthetic',
        model:'synthetic',timestamp:0,usage:{input:0,output:0,cacheRead:0,cacheWrite:0,totalTokens:0,
          cost:{input:0,output:0,cacheRead:0,cacheWrite:0,total:0}}});
      return stream;
    };
    const result=await runConversationTurn({client,prompt:'给我推荐手机',streamFn});
    assert.equal(result.status,'ok');assert.equal(result.analysisCalls.length,2);
    assert.equal(result.modelCalls,2);assert.equal(result.toolCalls,1);
    await client.observe(result.finalText!);
  } finally {await client.close();rmSync(folder,{recursive:true,force:true});}
});

test('缺Key或相对路径在启动前拒绝',()=>{
  const base={python,bundle:'/unused',memoryRoot:'/unused',facts:'/unused',userId:'u',sessionId:'s',apiKey:''};
  assert.throws(()=>new ConversationClient(base),/invalid_config/);
  assert.throws(()=>new ConversationClient({...base,apiKey:'synthetic-placeholder',python:'python'}),/invalid_config/);
  assert.throws(()=>new ConversationClient({...base,apiKey:'synthetic-placeholder',category:''}),/invalid_config/);
});

test('宿主品类以独立argv传入持续进程',async()=>{
  const folder=mkdtempSync(resolve(tmpdir(),'memrec-domain-'));
  const executable=resolve(folder,'worker.py');
  writeFileSync(executable,`#!${python}\nimport sys,json\nassert sys.argv[-2:]==['--category','musical_instruments']\nfor line in sys.stdin:\n r=json.loads(line)\n print(json.dumps({'version':1,'id':r['id'],'ok':True,'result':{'valid':True}}),flush=True)\n`,{mode:0o700});
  const client=new ConversationClient({python:executable,bundle:'/unused',memoryRoot:'/unused',facts:'/unused',
    userId:'u',sessionId:'s',apiKey:'synthetic-placeholder',category:'musical_instruments'});
  try {await client.validate('snapshot',new AbortController().signal);}
  finally {await client.close();rmSync(folder,{recursive:true,force:true});}
});
