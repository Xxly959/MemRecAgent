import { createAssistantMessageEventStream, type AssistantMessage } from '@earendil-works/pi-ai';
import type { StreamFn } from '@earendil-works/pi-agent-core';
import { validHealth, type CatalogHealth } from './bridge-client.js';

export function catalogSummary(health: CatalogHealth): string {
  return `目录就绪：${health.bundle_id}，${health.n_catalog_items}件商品；历史截止${health.as_of_ms}。`;
}

export function offlineModel(): StreamFn {
  let calls = 0;
  return (_model, context) => {
    const stream = createAssistantMessageEventStream();
    let content: AssistantMessage['content'];
    let stopReason: AssistantMessage['stopReason'];
    if (calls++ === 0) {
      content = [{type:'toolCall', id:'catalog-1', name:'catalog_status', arguments:{}}];
      stopReason = 'toolUse';
    } else {
      const receipt = context.messages.findLast(m => m.role === 'toolResult');
      if (!receipt || receipt.role !== 'toolResult' || receipt.toolCallId !== 'catalog-1'
        || receipt.isError) throw new Error('offline_receipt_missing');
      const text = receipt.content.filter(c=>c.type==='text').map(c=>c.text).join('');
      const health: unknown = JSON.parse(text);
      if (!validHealth(health)) throw new Error('offline_receipt_invalid');
      content = [{type:'text', text:catalogSummary(health)}];
      stopReason = 'stop';
    }
    // 固定模型替代供应商；Pi循环和Python领域执行均是真实调用。
    stream.end({role:'assistant', content, stopReason, api:'openai-completions',
      provider:'offline', model:'offline', timestamp:Date.now(), usage:{input:0, output:0,
        cacheRead:0, cacheWrite:0, totalTokens:0,
        cost:{input:0,output:0,cacheRead:0,cacheWrite:0,total:0}}});
    return stream;
  };
}
