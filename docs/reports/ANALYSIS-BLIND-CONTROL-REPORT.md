# 草案可见性控制接口

> 公开整理版：保留工程方法、实验结果与失败记录；本机路径已泛化，教学和内部协作记录留在本地。原始数据、模型和运行日志不随仓库发布。

后续进展：真正共享一次实际草案和view的首测已完成，54调用/18模板例，
direct18/18、visible18/18、blind17/18，唯一差为空条件start；并非推荐损失证明。
详见SHARED-DRAFT-CONTROL-REPORT.md。下文保留最初接口交付时的验证范围。

2026-10-24新增显式 `analyst_blind` 模式。实际草案一次、独立最终角色一次，
只隐藏最终角色收到的草案内容。当前是离线工程交付，没有新真实模型成绩。
默认入口和各自默认调用模式未切换，Pi聊天仍沿用原配置。

## 接口与处理

可信Python宿主可在 `ConversationMemorySession.prepare_user` 或
`ConversationProductSession.process_user` 传 `analysis_mode="analyst_blind"`。
普通analyst和blind调用相同的实际草案、相关记忆选择与最终角色构造路径；
`AnalystTask.expose_draft` 是严格bool，0/1、None和字符串均拒绝。

第二阶段仍验证草案状态与视图归属/时序，blind payload中draft=null，
不含untrusted_draft、原始坏草案和草案错误码。有效草案选择的相关memory_view保持原样；
无效草案仍只给空view。真实对话、实际问题、人物模板、最终角色system prompt相同，
最后的严格候选才有机会提交，草案不写盘。

第一阶段供应商、元数据、过大响应等故障仍立即停止，不为凑两次调用绕过安全边界。
可恢复解码错误可继续最终角色，首段processing_error及成本仍保留。
最终失败没有草案回退、重试或额外模型调用；删除、用途与快照守卫走同一正式路径。

## 归因边界

blind并不完全独立于草案：相关记忆view仍由草案选择。
接口支持的是固定view条件下“直接看到草案”的控制，不证明角色、总计算量或检索独立性的收益。
角色prompt两种双阶段模式相同，均新增了草案被隐藏时按实际turns分析的条件说明；
原模型成绩对应此前冻结源码，不能套到当前prompt。

若要逐例严格归因，应共享同一次真实草案、同一view和历史，再仅改变expose_draft；
独立多轮运行还会产生不同草案、后续状态及view，应作为系统策略比较另报。
调用上限同为两次，不等于实际token或金额相同；隐藏正文减少输入，应如实计量。
当前没有为了配齐token而加填充文本。

## 评测入口

`evaluate_memory_quality.py` 和 `evaluate_external_replay.py` 均接受新mode，
默认计划零调用；blind按整轮两次预留预算，剩一次时不能启动一轮，错误和not_run保留分母。
外部重放的阶段仍为draft/analyst，不能把blind的首调用标成direct而漏算。

```bash
uv run --offline --locked --extra ml --extra api python scripts/evaluate_memory_quality.py \
  --analysis-mode analyst_blind --output-dir artifacts/analysis_blind/new-run
```

以上只预览，不消费Key或建输出目录。live仍需显式授权范围、环境Key、调用预算和新目录。
旧 `memory_interaction_evaluation` 历史入口保持两模式，没有自动把新模式灌入所有旧实验。
现有预留test不运行，旧dev不重复追分。

## 工程证据

新12项检查最初全部失败于未支持mode/字段，接线后12/12；相关角色、宿主和评测160项通过。
完整Python1344项通过，44.08秒；Ruff181文件格式/检查、Python和TS编译通过。
Node完整串行87项通过，37.859秒。离线包审计在交付账本记录，尚无本增量独立最终审查。

实际双模式固定客户端产物位于 `artifacts/analysis_blind/20261024-offline/`：
同一两步/两会话程序探针，各4次固定回调、2/2严格检查、HTTP0，源码输入摘要一致。
逐消息对比第一调用完全相同、第二角色system相同、第二payload除draft外完全相同，
包含相同旧记忆；只有最终候选可发布。此处2/2是已知固定响应的工程检查，不是模型准确率。

未来仍需新冻结数据、共享草案的真实配对控制、完整Pi四组回答、系统基线与敏感度。
本增量不取代这些工作，也不改变旧267调用或先前统一写入开发集的负面结果。
