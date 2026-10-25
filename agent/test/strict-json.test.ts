// 重复键必须从原始文本识别，reviver看不到被覆盖的旧键。
import assert from 'node:assert/strict';
import { test } from 'node:test';
import { strictJson } from '../src/strict-json.js';

for (const text of ['{"a":1,"a":2}', '{"a":1,"\\u0061":2}',
  '{"x":[{"b":1,"b":2}]}', '{"a":NaN}', '{"a":1e999}', 'null trailing']) {
  test(`严格JSON拒绝 ${text}`, () => assert.throws(() => strictJson(text)));
}
test('嵌套不同对象同名与字符串内结构字符不误判', () => {
  const text = '{"a":"\\\"},:{","x":[{"a":1},{"a":2}],"v":{"x":null}}';
  assert.deepEqual(strictJson(text), JSON.parse(text));
});
