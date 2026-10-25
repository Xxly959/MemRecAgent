// 测试只构建固定合成推理包，不训练、不调用模型。
import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { execFileSync } from 'node:child_process';
export const root = resolve(dirname(fileURLToPath(import.meta.url)), '../../..');
export const python = resolve(root, '.venv/bin/python');
export function fixtureBundle() {
  const temp = mkdtempSync(resolve(tmpdir(), 'memrec-pi-'));
  const bundle = resolve(temp, 'bundle');
  execFileSync(python, [resolve(root, 'scripts/create_bridge_fixture.py'), bundle], { cwd: root });
  return { bundle, cleanup: () => rmSync(temp, { recursive: true, force: true }) };
}
