import assert from 'node:assert/strict';
import { test } from 'node:test';
import { createAssistantMessageEventStream, type AssistantMessage } from '@earendil-works/pi-ai';
import type { StreamFn } from '@earendil-works/pi-agent-core';
import { runCatalogTurn } from '../src/catalog-agent.js';
import { fixtureBundle, python } from './helpers.js';

for (const change of ['none','wrong-count','extra','duplicate','code-fence']) {
  test(`目录JSON必须与真实回执一致 ${change}`,async()=>{
    const f=fixtureBundle();let call=0;
    const streamFn:StreamFn=(_m,context)=>{
      const stream=createAssistantMessageEventStream();
      let content:AssistantMessage['content'];
      if (call++===0) content=[{type:'toolCall',id:'t',name:'catalog_status',arguments:{}}];
      else {
        const receipt=context.messages.findLast(m=>m.role==='toolResult');
        assert.ok(receipt?.role==='toolResult');
        const h=JSON.parse(receipt.content.filter(c=>c.type==='text').map(c=>c.text).join(''));
        const object={bundle_id:h.bundle_id,as_of_ms:h.as_of_ms,n_catalog_items:h.n_catalog_items};
        if (change==='wrong-count') object.n_catalog_items++;
        let text=JSON.stringify(change==='extra'?{...object,price:999}:object);
        if (change==='duplicate') text=text.replace('"as_of_ms":','"as_of_ms":0,"as_of_ms":');
        if (change==='code-fence') text='```json\n'+text+'\n```';
        content=[{type:'text',text}];
      }
      stream.end({role:'assistant',content,stopReason:call===1?'toolUse':'stop',
        api:'openai-completions',provider:'synthetic',model:'synthetic',timestamp:0,
        usage:{input:0,output:0,cacheRead:0,cacheWrite:0,totalTokens:0,
          cost:{input:0,output:0,cacheRead:0,cacheWrite:0,total:0}}});
      return stream;
    };
    try {
      const result=await runCatalogTurn({python,bundle:f.bundle,prompt:'x',streamFn,
        responseFormat:'catalog_json'});
      assert.equal(result.status,change==='none'?'ok':'error');
      if (change!=='none') {assert.equal(result.code,'final_guard');assert.equal(result.finalText,undefined);}
    } finally {f.cleanup();}
  });
}
