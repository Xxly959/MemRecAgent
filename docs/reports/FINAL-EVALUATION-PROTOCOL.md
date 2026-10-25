# 最终预留集执行协议（待有限调用授权）

> 公开整理版：保留工程方法、实验结果与失败记录；本机路径已泛化，教学和内部协作记录留在本地。原始数据、模型和运行日志不随仓库发布。

2026-10-25。只使用已有用户确认的AI合成数据，不新增真实个人资料；当前未发起调用。
这是原外部派生预留test的一次运行，不重跑已看开发集；原开发失败和负结果保持。

## 固定输入与执行

数据：`artifacts/external_derived/20261024-v1/derived-suite.json`，SHA256
`b03a1054b7ead64fef1b8a4e3def6774140f55f1d629d813a3fdc95506b098bb`，按原场景族划分split=test。
只评有推荐任务关联的15段/15任务、最多90步，未关联的另外15段不发送，分母与选择规则事前确定。
CLI已执行零调用plan预检：direct最多90次、analyst最多180次，总物理HTTP上限270。
不包括新的系统LLM对照、Pi完整回答、任何重试或开发重跑；上述后续若执行需另准备并明确额度。

模型配置沿正式DeepSeekClient：deepseek-flash、thinking disabled、temperature0、max_tokens4096、JSON输出，固定官方端点。
只读取本机忽略环境Key，密钥不进入报告/消息/冻结清单。单次/双阶段顺序固定direct先、analyst后；
两组各故事空库起步，各自真实上下文传播，不将两组不同演化的前缀声称逐例只差一次模型调用。
共享草案可见性研究沿原独立54调用控制，不在这里混入第三组。

每个推荐轮的G00/G10/G01/G11共享该组真实接受的分析/人物/当前约束/候选，程序排名，参考标签只在评分端。
所有算法组仍无匹配行为画像，实际冷回退，算法/协同估计必须null，不把这个测试包装为有效CF收益。
这不是Pi第二主模型完整回答验收，也不是用户满意度实验。

运行前固定当前源码/脚本/锁文件摘要，运行期间变动即停止；不根据test结果改prompt/标签/权重或重跑。
各阶段限额90/180，不借未用额度给其他实验，不启动一半双阶段调用；API/transport错误按现有runner终止并记剩余not_run。
过程错误、未实际询问的条件答复、空列表和候选外正例均保留。任何中断先查实际终态/调用账本，不自动重启。

预定独立输出目录（目前不存在）：

- `artifacts/final_evaluation/20261025-holdout-direct`
- `artifacts/final_evaluation/20261025-holdout-analyst`

执行命令在有限授权后才加live，当前只做plan：

```bash
rtk proxy env UV_CACHE_DIR=.uv-cache uv run --offline --locked --extra ml --extra api python scripts/evaluate_external_replay.py --suite artifacts/external_derived/20261024-v1/derived-suite.json --output-dir artifacts/final_evaluation/20261025-holdout-direct --split test --analysis-mode direct --max-calls 90 --live
rtk proxy env UV_CACHE_DIR=.uv-cache uv run --offline --locked --extra ml --extra api python scripts/evaluate_external_replay.py --suite artifacts/external_derived/20261024-v1/derived-suite.json --output-dir artifacts/final_evaluation/20261025-holdout-analyst --split test --analysis-mode analyst --max-calls 180 --live
```

以上环境命令不自动加载dotenv；授权后的可信launcher只在内存读取已配置Key并传环境，禁止打印或把Key当shell参数。
若当前Key失效，记录并停止，不把认证失败自动算作新重试额度。

## 事前分析口径

按原provisional/review_required分层：主分析只用原暂定参考，复核标签另报；不按test结果调整政策或挑主样本。
写入/状态按实际ok/processing/provider/host/not_run/unasked分母；state_match与status=ok的联合值另列，
不把因报错恰好未变的状态称成功。长期新记录、错误内容/归属、修订与澄清按既有指标合同。
排名报NDCG@3（eligible-only，gain=2^relevance−1）、Recall/HR/MRR、约束违反、空列表/错误和全部案例分母。
按独立场景族等权及成簇配对bootstrap，报告单/双状态与成本和G10−G00；算法未生效则效应null。
已报total_tokens与缺total计数分开，各字段已报合计可查；未知不能当0真实总量，不按缓存/耗时作因果收益结论。

所有消息/原始响应、逐步状态、四组工具结果、调用成本、输入源码摘要及终态报告保留本机忽略产物。
不导出或推送原始数据，不修改既有污染证据/历史实验以改善本次成绩。
新增严格证据守卫可能降低合法输出率，失败必须保留，不补手工正确记忆伪装自动效果。

最终结论允许无收益或倒退；本测试的完成不自动完成“真实活跃画像+同品类自然语言”、新软信号敏感度、
纯无记忆解释模型或真实用户满意度要求，缺口仍按FINAL-ACCEPTANCE-PREFLIGHT报告。
