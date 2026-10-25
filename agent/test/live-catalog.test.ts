import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { resolve } from 'node:path';
import { test } from 'node:test';
import { python, root } from './helpers.js';
for (const args of [[],['--live','--python',python,'--bundle','unused'],
  ['--live','--python',python,'--bundle','unused','--case','arbitrary-text']]) {
  test('真实入口显式live/Key/合成任务缺一不可，拒绝时无网络',()=>{
    const result=spawnSync(process.execPath,[resolve(root,'agent/dist/src/live-catalog.js'),...args],
      {encoding:'utf8',timeout:5000,env:{PATH:process.env.PATH}});
    assert.equal(result.status,2);assert.equal(result.stderr,'');
    const output=JSON.parse(result.stdout);assert.equal(output.status,'error');
    assert.ok(['invalid_config','missing_key'].includes(output.code));
    assert.equal(output.httpAttempts,0);
  });
}
