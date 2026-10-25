# 下一版请求数量评分协议

> 公开整理版：保留工程方法、实验结果与失败记录；本机路径已泛化，教学和内部协作记录留在本地。原始数据、模型和运行日志不随仓库发布。

2026-10-25，ISS088工程增量。新增模型调用0，不使用上一实验剩余10次，不重跑旧dev或test。
已关闭260调用报告的NDCG@3主结果与@1事后诊断均保持原样。

## 接口与选择

`evaluate_external_replay(..., scoring_mode="fixed"|"request", k=3)`；CLI显式`--scoring-mode request`。
默认fixed继续旧固定K语义，标记fixed_k_v1，不能将它当作自动按请求数量评估。
新request_count_v1只从task.runtime.request_text按现有有限数量解析器取明确1至10款；没有明确数量才用fallback k。
范围/多个冲突数量/超限在整个运行预检阶段拒绝，不创建输出、不发模型；不是静默丢掉困难任务改善分数。
该模式适用于明确评测推荐结果的任务，任意自然语言数量识别仍不保证。

不使用reference/gold、商品数量或实际返回列表长度选择K。实际只返回1件、原请求要2件时Precision分母仍为2；
模型/供应商失败的空结果仍按预定K评分，保持失败分母。同一任务四组共享K，不能给某组单独缩小分母。
生产提取、提示、提交、排序和实际输出数量不改；变化只在评测协议和本地审计元数据。

## 可复查元数据

- manifest：scoring_mode、scoring_protocol、每episode的scoring_cutoffs。
- cases：scoring_cutoff.k、source=fixed/fallback/explicit_request、evidence_spans。
- report/CLI终态摘要：scoring_protocol与计划task数的scoring_k_distribution。
- CLI零调用plan用相同helper显示各任务K/来源/原文位置，研究者应在首调前固定该模式。

evidence_spans对应该task原始请求字符串，不重复存原文。当前源码版本与旧109文件快照不同属有意新版本，不能把新源混进旧实验。
未来汇总按协议命名为NDCG@requestK，与fixed@3分开；不同协议/K分布不能混报提升。

## 验证

新增12项：首9项实际因缺scoring_mode接口TypeError RED→GREEN；真实CLI缺--scoring-mode参数RED→GREEN。
不冒称每个业务断言分别先红；另外的失败/参考字段隔离补充检查使用真实宿主的既有保护。
新单独测试用两件都符合预算且同样相关的商品，避免“只有一个正例”本来就使旧@3=1而无法揭示问题。
这是新的作者工程fixture，未改既有模型测试标签。

完整Python1606/52.69秒，随后严格串行Node102/47.023秒；Ruff、215文件格式、TS/Python编译通过。
新独立合成探针4固定回调、网络禁止/真实HTTP0：同一商品与分析消息，fixed K3=.613147、request K1=1；
这表示分母不同，**不是推荐算法效果提升**。歧义数量零调用拒绝，实际CLI无Key plan通过且不创建运行目录。

原封存/控制/合并215文件SHA不变、原109白名单快照逐文件匹配；原始与合并调用/步骤/案例470个对象核对吻合。
工作树只有external_replay.py和evaluate_external_replay.py这两个旧白名单文件合法变更，旧源已归档并未重写。
所有260模型尝试与原主成绩保留，不为“修正指标”再评已看test。

## 运行（默认零调用）

新作者合成fixture的本机预检示例：

```bash
rtk proxy env UV_CACHE_DIR=.uv-cache uv run --offline --locked --extra ml --extra api python scripts/evaluate_external_replay.py --suite artifacts/evaluation_cutoff/20261025-request-v1/new-synthetic-suite.json --output-dir artifacts/evaluation_cutoff/new-run --scoring-mode request
```

只有在新的输入/协议/有限调用授权均固定后，才可另用live；旧270额度不自动授权新实验。
固定K研究仍可显式--scoring-mode fixed，不悄悄重标历史结果。默认保持fixed是兼容选择，未来面向实际用户数量的研究应明确选request。

本增量由作者测试/复核，未追加独立二审，不当新供应商语义/真实画像或用户效用证据；整体项目结项仍须按完整清单判断。


离线构建成功：wheel77/sdist271文件，新模式入包，五公开技术文档字节与当前吻合；指定私有路径/凭证及绝对私有路径候选0。
248源码/测试/脚本摘要从探针到打包保持，215封存/控制/合并文件SHA仍同，package-audit.json可查。
旧260调用不是在新模式下重新评出的成绩，当前源与旧实验源之间的版本区别已保留。
