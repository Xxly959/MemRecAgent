import assert from 'node:assert/strict';
import {test} from 'node:test';
import {spawnSync} from 'node:child_process';
import {resolve} from 'node:path';
import {python,root} from './helpers.js';

test('聊天CLI须显式live和环境Key，缺Key零调用',()=>{
  const cli=resolve(root,'agent/dist/src/chat-cli.js');
  const result=spawnSync(process.execPath,[cli,'--live','--python',python,'--bundle','/unused',
    '--memory-root','/unused','--facts','/unused','--user-id','u','--session-id','s'],
    {input:'',encoding:'utf-8',env:{PATH:process.env.PATH},timeout:10000});
  assert.equal(result.status,2);
  assert.deepEqual(JSON.parse(result.stdout),{status:'error',code:'missing_key',httpAttempts:0});
});

test('聊天CLI拒绝缺少显式live，即使存在环境Key',()=>{
  const result=spawnSync(process.execPath,[resolve(root,'agent/dist/src/chat-cli.js')],
    {input:'',encoding:'utf-8',env:{PATH:process.env.PATH,DEEPSEEK_API_KEY:'synthetic-placeholder'},timeout:10000});
  assert.equal(result.status,2);assert.equal(JSON.parse(result.stdout).code,'invalid_config');
});


test('CLI明确但非法的反馈命令不调用模型且不作为用户自然语言发送',()=>{
  const result=spawnSync(process.execPath,['--import','data:text/javascript,globalThis.fetch%3D()%3D%3E%7Bthrow%20new%20Error(%22network_forbidden%22)%7D',resolve(root,'agent/dist/src/chat-cli.js'),'--live',
    '--python',python,'--bundle','/unused','--memory-root','/unused','--facts','/unused',
    '--user-id','u','--session-id','s'],{env:{...process.env,DEEPSEEK_API_KEY:'synthetic-placeholder'},
    input:'/block 2\n',encoding:'utf8',timeout:10000});
  assert.equal(result.status,0);
  const output=JSON.parse(result.stdout.trim());
  assert.equal(output.type,'feedback');
  assert.equal(output.result.code,'invalid_feedback_command');
  assert.equal(output.result.httpAttempts,0);
});
