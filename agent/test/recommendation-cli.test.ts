import assert from 'node:assert/strict';
import {test} from 'node:test';
import {execFileSync,spawnSync} from 'node:child_process';
import {mkdtempSync,rmSync} from 'node:fs';
import {tmpdir} from 'node:os';
import {resolve} from 'node:path';
import {python,root} from './helpers.js';

test('默认推荐CLI完全离线，真实Pi和磁盘记忆给出C/B',()=>{
  const path=mkdtempSync(resolve(tmpdir(),'memrec-rank-cli-'));
  try {
    execFileSync(python,[resolve(root,'scripts/create_memory_recommendation_fixture.py'),path]);
    const result=spawnSync(process.execPath,[resolve(root,'agent/dist/src/recommendation-cli.js'),
      '--python',python,'--bundle',resolve(path,'bundle'),'--memory-root',resolve(path,'memory'),
      '--facts',resolve(path,'facts.json'),'--request',resolve(path,'request.json')],
      {encoding:'utf-8',env:{PATH:process.env.PATH},timeout:15000});
    assert.equal(result.status,0,result.stderr);
    const value=JSON.parse(result.stdout);
    assert.equal(value.provider.httpAttempts,0);
    assert.deepEqual(value.result.items.map((p:{item_id:string})=>p.item_id),['C','B']);
  } finally {rmSync(path,{recursive:true,force:true});}
});

test('显式live没有环境Key应在任何工具或HTTP前停止',()=>{
  const result=spawnSync(process.execPath,[resolve(root,'agent/dist/src/recommendation-cli.js'),
    '--live','--python',python,'--bundle','/unused','--memory-root','/unused',
    '--facts','/unused','--request','/unused'],{encoding:'utf-8',env:{PATH:process.env.PATH},timeout:10000});
  assert.equal(result.status,2);
  assert.deepEqual(JSON.parse(result.stdout),{status:'error',code:'missing_key',httpAttempts:0});
});
