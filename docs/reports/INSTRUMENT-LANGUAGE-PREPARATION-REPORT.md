# 乐器同域真实语言实验：执行包已准备，尚未调用

> 公开整理版：保留工程方法、实验结果与失败记录；本机路径已泛化，教学和内部协作记录留在本地。原始数据、模型和运行日志不随仓库发布。

2026-10-25。当前API0，未读取Key；需要新增有限调用授权后才运行。
完整协议见INSTRUMENT-LANGUAGE-EVALUATION-PROTOCOL.md，准备目录artifacts/instrument_language/20261025-prepared。

## 已具备的具体输入与执行能力

6段42步新作者合成对话，4个既有活跃BPR画像+2个合成冷用户、本机原生ID绑定、现有5000商品包与4871完整标题表。
每个画像的两个商品按固定哈希从未见商品选择，不看模型分/未来购买；不用原模型首位制造记忆提升。
6个最后查询明确三款，共享实际分析作四组排名；其他轮也保留实际result和受益人/指定商品校验。
计划direct最多42次、analyst最多84次，总最多126，所有失败/未运行/未问确认保留分母。

模型只收到公开目录标题/ID与合成言语、真实已发生前缀、相关记忆及草案；画像ID/向量/行为/参考不发送。
引用source是商品实体，不称品牌；实测算法臂准确标识BPR或冷回退。
没有独立效用目标，relevance=0明确表示未提供此标签，因此NDCG等效用指标输出null/N/A，不能解释成0分。
这次既不是盲测，也不是完整Pi回答/系统无记忆或向量记忆基线；不以通过验收声称推荐更好。

## 回放器改造

新增NativeReplayEnvironment，加载正式已校验bundle和显式facts/aliases/本机bindings，冻结所有实际环境文件。
external_replay可选native_environment，默认旧合成手机路径与旧评分协议保持；CLI增加--native-environment，默认仍零调用plan。
原生实体评分类型product、实际活跃算法臂、真实目录先验来源和局限明确，不再用brand/全部cold描述该环境。
native消息守卫拒绝用户ID/评分字段，调用前后仍检查源码/输入；供应商transport只停当前故事，下一全新故事可继续。
非transport供应商错误、host错误、冻结漂移停止campaign；零重试，无自动resume或重复输出目录。

## 预检发现与修复

首轮固定候选两模式各40/42处理、35/42状态命中：

1. NI02T05固定响应scope误写request，实际合法枚举current。仅修正固定客户端规格，言语与状态参考不改。
2. NI03T02确认unknown主体时，MemoryCommitSession先用规范ID重建CandidateAssertion，而原文只有标题别名，严格entity_evidence拒绝。

第二项为真实产品缺陷ISS089。手机BrandA与乐器Concert Harp两项实际RED→GREEN：规范键仅用于同目标守卫，
确认重建仍使用票据中原始实体别名，通过原严格证据复核后，propose再规范化存储键。未弱化decoder或改用户证据。
首次预检、初版脚本/规格、原始suite均保留。后续只增参考受益人/指定商品字段，以及NI06T06咨询/购买建议路由的事前争议标记。

新增受益人回归两项实际RED→GREEN：推荐对象错为self即使长期状态仍正确，也不能算严格成功。
本轮共10项新Python用例，含原生目录/标题通路、冷与活跃路由、传输故事边界、输入漂移、CLI plan、上述证据/人物回归。

## 实际验证

- 完整Python1636 passed / 69.46s；随后串行Node103/103 / 54.04s。Ruff/224文件格式、TS/Python编译通过。
- 最终固定响应direct42、analyst84回调，各42/42处理/状态/严格、6/6整段；API0、实际BPR case-arm各8。
- 总42步、无争议41步均保留；summarize.py从逐行calls/steps/cases独立重算状态、动作、主体/指定商品、用户前缀并核对全分母。
- 全部效用NDCG保持null；固定响应的全通过不是模型理解成绩。
- 启动器8项隔离检查通过：plan、正常两组、transport、HTTP停止、缺终态、坏额度、冻结漂移、重复启动；子进程均替换，真实API0。
- 可信启动器显式live后才加载本机Key，使用umask077保存付费运行目录/日志，不把Key放进参数或输出。

分析器预固定6段/族配对状态差值、2000次bootstrap/seed20261025，只作描述性系统对比。
所有token未知单列，不靠report.json一个文件算物理成本；四组效用/协同仍无独立目标，不能补出收益结论。

## 范围

本轮没有新增真实API、Key读取、训练下载、新依赖、提交或推送；没有新增独立审查，作者验证不称独立复审PASS。
旧手机开发/预留/系统基线及260调用全部保持，不用本轮固定结果覆盖旧失败。
目前只差本次有限调用与发送范围授权才能启动已准备实验；预算属性、短名、独立效用及最新Pi完整回答仍保留缺口，整体目标未结项。

交付审计另发现ISS090：首个sdist690文件含410个.tmp_pytest临时文件，原审计路径集合未覆盖该目录；首包与verification.json保留为不合格记录。
已补.gitignore及显式Hatch sdist排除，经实际首包断言RED→dist-clean重建GREEN：wheel79/sdist280，源码/五公开文档匹配、顶层白名单通过、临时目录为0，详见package-verification-clean.json。
此后38项相关回归通过；未改推荐/记忆运行源码、未增依赖或改锁文件，不为纯打包排除重复模型或语言验收。
冻结清单与私有source-before-calls在同准备目录，授权前仅运行默认零调用plan。
