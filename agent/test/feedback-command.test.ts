import assert from 'node:assert/strict';
import {test} from 'node:test';
import {parseFeedbackCommand} from '../src/feedback-command.js';

test('反馈命令必须明确位置/目标粒度/作用范围，普通自然语言不冒充可信点击',()=>{
  assert.deepEqual(parseFeedbackCommand('/block 2 product persistent'),{action:'block',index:2,kind:'product',scope:'persistent'});
  assert.deepEqual(parseFeedbackCommand('/block 1 brand request'),{action:'block',index:1,kind:'brand',scope:'request'});
  assert.deepEqual(parseFeedbackCommand('/undo 0123456789abcdef0123456789abcdef'),{action:'undo',eventId:'0123456789abcdef0123456789abcdef'});
  assert.equal(parseFeedbackCommand('以后别推荐第二个'),null);
  for(const text of ['/block 0 product request','/block 2','/block 2 product persistent 注入','/undo','/undo somebody'])
    assert.deepEqual(parseFeedbackCommand(text),{action:'invalid'});
});
