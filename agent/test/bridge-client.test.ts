import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { resolve } from 'node:path';
import { test } from 'node:test';
import { BridgeClient } from '../src/bridge-client.js';
import { fixtureBundle, python, root } from './helpers.js';

test('bridge真实Python服务回执且能连续请求', async () => {
  const f = fixtureBundle();
  const client = new BridgeClient({python, bundle:f.bundle, requestId:'request-1'});
  try {
    const first = await client.request('tool-1');
    assert.equal(first.n_catalog_items, 3);
    assert.equal(first.status, 'ready');
    assert.deepEqual(await client.request('tool-2'), first);
  } finally { await client.close(); f.cleanup(); }
});

for (const mode of ['wrong-id','duplicate','bad-json','duplicate-key','bad-health','unsafe','oversize','eof','silent','utf8','nan']) {
  test(`bridge固定故障码且拒绝下一请求 ${mode}`, async () => {
    const client = new BridgeClient({python, bundle:'unused', requestId:'r', toolTimeoutMs:150},
      () => spawn(process.execPath, [resolve(root,'agent/test/fixtures/bridge-child.mjs'),mode]));
    try {
      // 重复帧不得在第一条成功后静默放过；下一请求也不可继续。
      if (mode === 'duplicate') {
        await client.request('t').catch(()=>undefined);
        await new Promise(resolve=>setTimeout(resolve,20));
        await assert.rejects(client.request('t2'), /bridge_protocol/);
      } else await assert.rejects(client.request('t'),
        mode === 'silent' ? /tool_timeout/ : /bridge_protocol|bridge_eof/);
      await assert.rejects(client.request('later'));
    } finally { await client.close(); }
  });
}
for (const mode of ['split','stderr','max-frame']) {
  test(`bridge合法分片和stderr不污染协议 ${mode}`, async () => {
    const client = new BridgeClient({python,bundle:'unused',requestId:'r'},
      () => spawn(process.execPath, [resolve(root,'agent/test/fixtures/bridge-child.mjs'),mode]));
    try { assert.equal((await client.request('t')).n_catalog_items,3); }
    finally {await client.close();}
  });
}
test('bridge取消和启动失败均有界', async () => {
  const controller = new AbortController(); controller.abort();
  const client = new BridgeClient({python,bundle:'unused',requestId:'r'});
  await assert.rejects(client.request('t',controller.signal), /cancelled/);
  await client.close();
  const missing = new BridgeClient({python:'/missing/interpreter',bundle:'unused',requestId:'r'});
  await assert.rejects(missing.request('t'), /bridge_start_failed/);
  await missing.close();
});
