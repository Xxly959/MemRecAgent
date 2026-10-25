# Phase3原范围结项审计

> 公开整理版：保留工程方法、结果和失败；本机路径已泛化，内部记录及原始产物留在本地。

2026-10-22，Asia/Shanghai。独立只读审计；本轮仅新增本文，不修改产品、参考、旧结果或目标状态，不调用模型、不读取凭证、不推送。审计对象是当前本地软件及冻结合成验收证据，不是生产部署认证。

## 结项建议

**建议按原已批准范围结项：Phase3软件交付完成，显式analyst路径在最新有限合成生命周期验收通过；保留direct基线失败和真实用户泛化风险。** 本轮逐要求复核没有发现仍缺的必需接口，亦未复现需阻止上述有限范围结项的当前安全缺口。此判断不是对全部代码的无缺陷保证，不将模型语义风险改称已彻底修复。

原范围要求独立记忆评测与失败案例，不要求所有保留模式、所有历史困难集或无限用户表达满分。当前明确双阶段路径已经有完整新冻结主线证据，不能因为direct的两个已披露失败，再临时增加“基线也必须满分”的阶段门槛。反之，analyst的18/18只解决有限主线验收，不能代替下面的软件接口、安全边界、管理能力和评测完整性核查。

**允许的结项表述**：带证据的本人长期记忆、他人会话记忆、修订/时效/删除、受控提交/确认、渐进交互与相关激活、两阶段分析、可信目标反馈和用途控制已交付本地接口；新合成集analyst完整闭环通过。**不能表述**：默认direct已经全部通过、任意真实对话自动写入可靠、完整自主Agent已完成、记忆改善商品推荐、广告决策或生产服务已验证。

## 范围依据与证据等级

权威按原需求读取，而非把滚动接续中的历史“下一步”累加为新门槛：

- `docs/ROADMAP.md`原Phase3表：episodic/semantic/working memory的提取、整合、检索、更新、遗忘；证据/时间/置信度/范围；负向、比较、排除、冲突、时效；查看/纠正/删除；独立记忆评测及失败案例；合成标识。
- `docs/PHASE3-SCOPE-AND-TASKS.md`原范围与四项队列：本地管理、受控提交/确认、完整多轮检查、必要的专职角色；明确不以六组636调用、Pi、排序或生产部署作Phase3门槛。
- 已批准 `docs/superpowers/specs/2026-10-20-phase3-memory-records-design.md` 与 `2026-10-21-phase3-progressive-preference-design.md`：本人长期/他人会话、不可变证据与显式修订、当前请求优先、真实发生的双方上下文、线索不自动晋升、可信精确目标反馈。
- `AGENTS.md`及完成清单明确的个性化/未来广告用途控制，不能因广告未开发而省略最小许可接口。

本审计同时读取README、PROGRESS、DECISIONS、完成清单、阶段范围与相关交付报告，检查Git状态后保留全部已有改动。以下“已交付”来自当前源代码与测试核查；工程全量、独立增量审查、真实模型有限样本和推荐效果分别标注，不互相代替。

## 逐要求证据矩阵

| 原要求/完成清单项 | 当前树实现与可定位测试 | 模型/报告证据及限制 | 审计结论 |
| --- | --- | --- | --- |
| 本人长期，他人仅当前user/session | `memory.py:PreferenceMemory.__post_init__`拒绝非self；`memory_commit.py:_accept_context`仅成功接受后发布他人稳定会话态度；`test_memory_session_context.py`覆盖换人物/用户/会话/重建/临时范围；`test_memory_analyst_host.py`覆盖渐进他人路径 | 最新V组他人态度及后续激活analyst通过；direct V5仍错引用新锚点，不能凭同显示名或0长期错写宣称正确 | 本地边界已交付；人物语义风险保留 |
| 原始证据、时间、置信度、范围、不可变修订 | `memory.py:PreferenceMemory`为frozen，保存evidence/ref与双时间；`memory_store.py:_validate_history/list_memories/active_memories`验证链头、按发生/可用时间严格筛选、neutral取消旧态度；`test_memory_store.py`覆盖晚到/未来/分叉/原子失败 | 置信度可空，不虚构已校准概率；证据快照与当前状态分离 | 已交付 |
| 整合、检索、遗忘/时效 | 同对象显式supersedes链；按用户/类别/实体/as_of取有效头，`max_memory_age`可选；`test_memory_freshness.py`覆盖先解修订再失效、不复活旧exclude | 默认不自动过期；失效不等于物理删除，不实现未经约定的自动衰减打分/向量检索 | 满足本阶段最小实现 |
| 查看/纠正/删除与派生重放防护 | `memory_store.py:correct/delete_entity/evidence_is_deleted`；`scripts/manage_memories.py`查看当前/历史；删除完整链并保留摘要水位；`test_memory_management.py`、`test_memory_management_cli.py`、`test_memory_proposals.py`及interaction删除来源回归 | 外部备份/已导出副本由调用方管理；当前无向量索引；本地存储不承诺多进程事务 | 已交付；未来新增索引须纳入删除 |
| 请求/偏好主体/范围分离，本次例外不永久修订，冲突澄清 | `memory_policy.py:resolve_recommendation`、`memory_proposals.py`、`memory_commit.py`；`test_memory_policy.py`、`test_memory_subject_gate.py`覆盖未知受益人与已知态度不可被短答改派 | 最新U/W含选项查看、保留旧限制、明确取消、范围确认，analyst与direct动作/状态均通过；历史ISS-039仍开放 | 守卫与有限主线已交付；不声称任意人物理解 |
| 受控prepare/confirm/commit，失败不部分发布 | `memory_commit.py`私有票据绑定root/用户/会话/快照/许可revision，`commit`在clarify时拒绝且`add_many`原子写入；`memory_confirmation.py`绑定实际问题与回复 | `test_memory_commit.py`、`test_memory_confirmation.py`、`test_memory_analyst_host.py`覆盖重复/过期/删除/范围确认；最新各2/2完整确认 | 已交付；语言CLI仍只读、真实自动写入未开放 |
| 普通双方对话，线索→明确稳定本人表达→修订→相关激活 | `memory_interaction.py:observe_assistant/prepare_user/commit/_background`；普通助手轮真实进入历史，成功接受才发布线索；`test_memory_interaction.py`覆盖助手自夸、重复询问、唯一问题短答、含混多目标、取消与删除线索 | 最新3组合成故事含脚本普通助手语境、会话线索、本人稳定表达、隔话题激活与修订；脚本助手不是自主回答质量证明 | Hook与有限闭环已交付；无弱信号任意自动晋升 |
| 专职MemoryAnalyst真实两阶段、仅最终候选提交 | `prepare_user(analysis_mode="analyst")`实际draft→受限相关只读view→独立final，final才交原host.prepare/confirm；阶段间与返回后检查状态；`test_memory_analyst_host.py`覆盖删除/纠正/失败/确认/重绑定 | 最新analyst40物理请求=20draft+20final；V5草案T5→final T4实际纠正，后续激活通过。不是仅创建角色名字或模拟tool call | 已交付；direct仍为默认，analyst须显式选用 |
| 无效draft不能阻断独立重读，也不能放宽最终守卫 | `memory_analyst.py:inspect_draft`只将合法wire/metadata且≤64KiB的真实decode失败作为不可信raw交final，empty view；provider/metadata/oversize停止；final严格拒绝、无重试或回退；`test_memory_invalid_draft.py` | ISS-050独立167+12无C/I/M；最新真实集draft_processing_errors=0，未遇坏draft，不能称本次模型坏草案纠错已证实 | 必需入口已交付；模型收益证据仍空缺，不阻塞软件结项 |
| 可信精确商品/品牌“不感兴趣”、本次/持久、撤销/删除 | `memory_feedback.py:HostFeedbackSession`与`memory_feedback_store.py`独立事件存储；只接受本宿主display，product不扩brand；`start_request`清临时状态，undo精确事件，delete有水位；`test_memory_feedback.py` | TRUSTED-FEEDBACK报告38+38独立检查通过、1 Minor规格澄清；无按钮/HTTP/语言自动绑定/实际过滤排名；临时收据不跨实例持久 | 已交付批准的本地结构化接口；宿主接入义务须保留 |
| 用户用途控制与广告默认拒绝 | `memory_use_policy.py`持久三开关，默认advertising=False；`memory_policy/proposals/commit/interaction`共用许可；`set_use_policy`清历史工作状态，调用起点revision防同实例/ABA；`test_memory_use_policy.py` | 原独立Important由作者RED→GREEN修复并重放12探针，无第二轮独立复审；本审计再次实跑同实例关/关开回归。精确点击反馈是另一显式操作，不由自动长期更新开关推断许可；管理仍可用 | ISS-049缺接口已补；未发现现有回归可复现的许可绕过 |
| 独立评测、失败分析、分母、成本与完整链路 | `memory_interaction_evaluation.py`与冻结runner；最新manifest/input/source/calls/analysis-records/steps/report/author audit留存；错误不丢分母，条件回复仅在真实问后进入 | 最新60物理请求、direct16/18与analyst18/18；旧18/24、5/18、10/18/12/18、15/18/16/18及定向失败保留；本轮没有重跑追分 | 满足独立记忆评测及失败案例要求，不作不同集因果对比 |
| 工程/审查/包与隐私、文档教学分离 | 最新request-intent ledger记录默认979+练习14=993、Ruff/137格式/离线构建；包审计sdist151/wheel47，private_members为空、指定credential pattern候选0；增量独立77+6无C/I/M；ISSUES保留负结果 | 这里全量/构建是读取已有实际日志，未冒充本审计重跑；模式扫描不是隐私完备证明；发布前仍须另审白名单，未授权推送 |

## 最新冻结结果独立核对

产物：`artifacts/phase3_request_lifecycle/20261022T121900Z/`。读取两模式manifest、comparison-report、author-results-audit及原始calls；当前源码/fixture/依赖等与冻结`source/`逐文件字节比对 **23/23一致、差异0**。没有只依赖`source_unchanged: true`的自报。

| 维度 | direct | analyst |
| --- | ---: | ---: |
| 物理模型请求 | 20 | 40（draft20/final20） |
| 基础严格步骤 | 16/18 | 18/18 |
| 整段严格通过 | 1/3 | 3/3 |
| 完整确认 | 2/2 | 2/2 |
| 长期状态头/工作线索正确 | 各18/18 | 各18/18 |
| 激活正确 | 16/18 | 18/18 |
| 实际新增长期/内容错写/漏长期 | 6/0/0 | 6/0/0 |
| 唯一人物原锚点独立旁表 | 0/1 | 1/1 |
| 首段decode错误/最终处理错误/供应商错 | 0/0/0 | 0/0/0 |
| total tokens | 63,718 | 136,964 |
| 累计HTTP耗时 | 22.594s | 46.628s |

实际金额未提供，不估算成已发生费用。两阶段计算和view不同，且无重复采样/随机对照，不能把分数差归为角色本身的纯因果收益，也不能同历史不同集合宣称提升。

原始calls可核验V5：direct受益人引用`T5:我室友`，analyst draft同样为T5，analyst final改为`T4:我室友`；此例存在真实两阶段纠正，覆盖独立旁表而非只看宽松subject评分。direct U3保留过长topic“小米的手机系统”，当前精确/“手机”后缀对齐漏激活；这是实际失败，不是仅标签差异。analyst U3的draft本已正确，不能一并归功于final。

## 本审计新增验证

所有命令经`rtk proxy`，测试用临时UV缓存；不重复全量测试、不调用API。两次窄范围pytest合计 **32 passed**：

1. management/freshness/feedback/use-policy/analyst-host中，整链删除、旧候选重放、过期不复活、精确反馈、跨请求清理、外部删除、关闭个性化、同实例许可关/关开、范围确认等：**17 passed，128 deselected，0.11s**。
2. invalid-draft/subject-gate/interaction中，坏draft仅final提交、wire/metadata/oversize停止、坏final无重试、已知态度不被未知人物短答改派、助手/重复询问不写长期、实际唯一问题绑定及删除原来源重放：**15 passed，59 deselected，0.19s**。

命令形式：`env UV_CACHE_DIR=/private/tmp/memrec-closure-audit-uv uv run --locked --extra ml --extra api pytest -q <上述文件> -k <对应测试名称子串>`。这32项是本审计的定向安全复核，未增加/替代993全量数字，也不称对全部路径的再次独立审查。工具读取曾误用宿主缺失的`python`，随后改用`uv run ... python`；一次元数据探查未处理direct空analysis-records导致IndexError，改为空表安全读取后完成；两者均未触发模型或改变产品/结果。

## 保留问题与必要收尾

1. **ISS-039/041继续保留**：direct合法错人物原锚点与长topic漏激活。默认模式仍是direct；展示/使用时应明确本轮完整主线通过对应显式analyst，不把默认路径写成已全过。可进入后续针对性改进，不能删除失败或按同集合追分。
2. **ISS-050证据边界**：坏draft转final的程序路径已经交付且严格守卫保留；真实新集未出现坏draft，仍无本轮模型坏draft纠错样本。无需为补漂亮结论再消耗当前冻结集。
3. **泛化与安全边界**：程序能拒绝无效/过期/明确未解候选，不能辨认所有JSON合法但语义错的归因。当前是单进程串行可信Python宿主，不承诺恶意同进程、互联网认证、多进程一致性、已发送网络内容撤回或外部备份擦除。真实自动写入继续关闭。
4. **接入义务**：反馈宿主须登记真实展示并在新推荐请求调用start_request；未来排名/广告如消费这些信号，需显式接入许可与删除规则，不能将结构化blocked信号写成已应用排名。按钮/HTTP/UI与Pi自主循环不属于本次结项缺失。
5. **文档状态同步是结项收尾，不是新增产品任务**：完成清单、README和多份滚动接续仍保留历史“未完成/下一步”，最终负责人应以本报告和REQUEST-INTENT报告补最新范围化结论，保留旧时间戳与失败。本文按授权只写审计，不自行改这些文档或goal；不要求把历史原文抹成通过。
6. **学习与阶段边界**：用户仍无新增独立实现/掌握证据，不安排重复算法填空；后续Phase4排名贡献、Phase5口碑、Phase6工程大矩阵以及Pi计划按各自授权进入。Phase3结项不自动授权它们、发布或推送。

按以上边界收尾，当前没有必须再实现一个接口或再跑一套模型样本才能满足原Phase3范围的证据。后续真实用户验证与推荐收益属于新的验证问题，应独立规划预算和成功口径。
