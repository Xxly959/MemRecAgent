// JSON.parse先验证语法；原文词法遍历保留会被覆盖的重复键信息。
export function strictJson(text: string): unknown {
  const value: unknown = JSON.parse(text);
  const stack: { object: boolean; key: boolean; keys: Set<string> }[] = [];
  for (let i = 0; i < text.length; i++) {
    const ch = text[i];
    if (ch === '"') {
      const start = i++;
      for (; i < text.length; i++) {
        if (text[i] === '\\') i++;
        else if (text[i] === '"') break;
      }
      const top = stack.at(-1);
      if (top?.object && top.key) {
        const key: string = JSON.parse(text.slice(start, i + 1));
        if (top.keys.has(key)) throw new Error('invalid_json');
        top.keys.add(key);
        top.key = false;
      }
    } else if (ch === '{' || ch === '[') {
      stack.push({ object: ch === '{', key: true, keys: new Set() });
    } else if (ch === '}' || ch === ']') stack.pop();
    else if (ch === ',') {
      const top = stack.at(-1);
      if (top?.object) top.key = true;
    }
  }
  const pending: unknown[] = [value];
  while (pending.length) {
    const item = pending.pop();
    if (typeof item === 'number' && !Number.isFinite(item)) throw new Error('invalid_json');
    if (item && typeof item === 'object') pending.push(...Object.values(item));
  }
  return value;
}
