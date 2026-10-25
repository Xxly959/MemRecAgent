# Pi桥接路线：有限DeepSeek真实目录工具流

> 公开整理版：保留工程方法、实验结果与失败记录；本机路径已泛化，教学和内部协作记录留在本地。原始数据、模型和运行日志不随仓库发布。

2026-10-23。用户明确同意继续桥接。Pi循环保持固定1.4.0，Python记忆与推荐保留；当前只接真实供应商目录工具，未将克隆源码变成fork、未接记忆或排序，也未执行四组收益消融。

## 实现与原理

新增agent/src/deepseek-model.ts、live-catalog.ts、catalog-probes.ts及三测试文件；仅修改既有catalog-agent.ts，加入可信model及catalog_json格式。默认离线text/CLI保持。

固定官方端点deepseek-flash，temperature0/max_tokens512；SDK内外重试0，自定义fetch实际发送前检查URL/POST/取消/2次物理上限，redirect error。第一模型具名强制catalog_status，第二none；最终JSON恰含bundle_id/as_of_ms/n_catalog_items，逐项核真实health后程序渲染，不修错误/额外/重复字段或代码块。不是自由工具选择/意图评测。

Key只在显式live从环境读取；实验宿主沿既有授权，从本机忽略0600配置注入，不写源或产物。默认不保存模型内容，受控固定合成入口才留正常/工具内容；任意供应商错误原文不记。length截断仅记原因/用量，无正文，是审查Minor延后。

## 工程与独立证据

- 新目录字段/假HTTP/入口先RED→11新增GREEN；供应商接口类型纠正见ISS-058。wire缺口另1RED→GREEN；最终新12/全Node59通过，6.276s，TS编译通过。
- Python最终1037通过/19.89s，本地练习14/1.10s，Ruff/141格式通过；旧Python代码/模型/记忆/测试/依赖未改。
- 单次独立首调前58Node/5.840s、tsc noEmit及6假传输探针通过，0C/0I/1Minor。length内容缺口保留，未二次复审；后续wire作者修复不是已独立审查。
- 168既有摘要仅计划内catalog-agent.ts改变，其他167不变；npm/Python锁均不变。离线sdist175/wheel48成功，指定私有路径/凭证模式/实际Key匹配均0；147个新产物文件实际Key匹配0。

## 两份真实结果分别保留

| 运行 | 物理HTTP | 工具 | 结果 |
| --- | --- | --- | --- |
| 原冻结三例首测 | 1，HTTP400 | 0 | 0/3成功，执行1/未执行2，provider失败停止，无重试 |
| 独立wire修复协议探针，新输入 | 2，均200 | 1 | 1个有限工具链探针通过，非原三例回归 |

原目录artifacts/pi_deepseek_catalog/20261023T014102303422Z，report/runs/manifest/source保留。执行期间源码一致，没有为了400当场改请求或继续余例。原错误正文未存，usage未报告，不能将SDK默认0算零消耗。

ISS-059：本地model.reasoning=false没有把thinking disabled送入wire；新假fetch测试实证undefined RED，onPayload明确关闭并移除reasoning_effort后GREEN。原Python客户端及旧spike已显式关闭，此供应商条件迁移时漏掉。[官方思考模式](https://api-docs.deepseek.com/guides/thinking_mode/)默认开启，[Chat API约束](https://api-docs.deepseek.com/api/create-chat-completion/)说明思考模式不支持具名工具选择并返回400；与本次相符，但缺原message不能宣称唯一400因果已证实。

随后另冻新输入/源码/构建/同一合成bundle，最多2HTTP，不重跑原集。目录artifacts/pi_deepseek_wire_probe/20261023T014425062880Z，result含完整正常合成回答与真实事件链。model_request→tool_execution_start→bridge_receipt→tool_execution_end→tool_result→model_request→delivered，toolCallId一致，最终JSON与真实health一致，运行期源码不变、stderr空。

该探针墙时3290.139ms；第一响应input354/output17/total371，第二219/63/282，合计已报告input573/output80/total653，cacheRead均0。原400用量未知，所以653不是本轮全部3请求的总消耗；未核价格，不报金额。201/270ms是fetch到响应头耗时，不能当完整生成时延。

## 限制、决策与接续

桥接保持Pi固定、把记忆与推荐作为明确实验变量；不必先改上游循环来优化记忆。已真实接通一条有限目录工具链，原三例验收仍失败/停止，不能混成3/3或任意对话已可靠。首工具强制，未验证模型自主挑工具；记忆Hook/排名尚未接入，旧Phase3模型分数不改。

延后Minor：length截断不留正文；当前保留stopReason/用量但该情形无法核原始截断文本。原首调失败不能当做通过；本轮没有记忆/推荐效果结论。

下一复用已有ConversationMemorySession和ProductService.recommend，先一个跨会话偏好→实际筛选/评分→DeepSeek回答的完整例子，再冻结基础/仅记忆/仅算法/组合四组。继续在同一Pi/模型/候选/预算下保留全部回答与负结果，不承诺组合每次获胜。不训练/下载数据/自动写真实记忆/提交推送；无用户新增独立实现证据。
