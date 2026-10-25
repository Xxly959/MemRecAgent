import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { resolve } from 'node:path';
import { test } from 'node:test';
import { fixtureBundle, python, root } from './helpers.js';
const cli=resolve(root,'agent/dist/src/cli.js');

test('正式CLI完成真实Pi和Python两轮，无Key要求', () => {
  const f=fixtureBundle();
  try {
    const result=spawnSync(process.execPath,[cli,'--python',python,'--bundle',f.bundle],
      {encoding:'utf8',timeout:15000,env:{PATH:process.env.PATH}});
    assert.equal(result.status,0);assert.equal(result.stderr,'');
    const output=JSON.parse(result.stdout);
    assert.equal(output.status,'ok'); assert.equal(output.modelCalls,2);
    assert.equal(output.toolCalls,1);assert.match(output.finalText,/3件商品/);
    assert.equal(output.events,undefined);
  } finally {f.cleanup();}
});
for (const args of [[],['--live'],['--python','python','--bundle','unused'],
  ['--python',python,'--bundle','/sensitive/missing-path']]) {
  test(`CLI非法配置或坏目录固定失败 ${args.length}`, () => {
    const result=spawnSync(process.execPath,[cli,...args],{encoding:'utf8',timeout:10000});
    assert.equal(result.status,2);assert.equal(result.stderr,'');
    const output=JSON.parse(result.stdout);
    assert.equal(output.status,'error');assert.equal(typeof output.code,'string');
    assert.equal(output.finalText,undefined);
    assert.ok(!result.stdout.includes('sensitive'));
  });
}
