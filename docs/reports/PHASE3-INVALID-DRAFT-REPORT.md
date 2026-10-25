# Phase3无效草案的独立最终分析报告

> 公开整理版：保留工程方法、实验结果与失败记录；本机路径已泛化，教学和内部协作记录留在本地。原始数据、模型和运行日志不随仓库发布。

2026-10-22。本轮落实ISS-050结构性缺口，不改变旧自然语言分数或最终提交规则。API0、不推送，完整Phase3仍未完成。

## 本次实现

新增inspect_draft与DraftInspection，仍通过原analyze_draft做一次真实调用及元数据校验。只有合法ModelReply/元数据、UTF8不超过65536字节且真实严格decode失败，才把analysis=None、有限raw、固定错误码和首阶段processing_error记录交给最终角色。供应商失败、非法wire/metadata、超大/不可编码内容仍立即停止；direct和单独analyze_draft原拒绝行为保持。

AnalystTask新增兼容默认draft_error_code；builder再次核对raw确实按该码失败，禁止合法候选假标无效、错误码不符或坏草案夹带旧memory entries。无效输入使用draft=null和untrusted_draft数据对象，raw不是system指令、不是证据或可提交候选。无效草案不参与相关记忆挑选，最终角色只得到实际turns/已接受人物来源和空相关视图。

ConversationMemorySession的显式analyst路径始终最多草案＋一次最终分析，不程序修坏输出、不重试原任务、不加第三调用。两次独立任务之后只有最终合法候选可prepare/confirm/commit；坏final不回退，实际问题/确认、删除水位、用途起始revision、snapshot/root保护不变。

原_run_analysis、_metadata、_related、build_memory_view函数逐字一致，最终解码与存储守卫未放宽；仅新增受限转交和独立数据契约，不能称原草案已校验通过。

## 调用与错误记录

首段被拒而最终成功时，整步processing_errors仍按原含义为0，但新增draft_processing_errors保留首阶段失败；analysis-records.jsonl保存实际分析尝试的安全状态和成本元数据。它是角色尝试账本，不代替物理请求分母：发送前冻结拒绝可能有内部尝试记录而无物理发送，物理数仍以calls.jsonl/model_calls为准。旧评分函数与原案例/标签不改。

原两个程序测试按被替代规则更新：坏draft JSON可进第二阶段，固定客户端40而非39请求、最终18/18，但首段失败数1；最终仍坏的路径记录两次拒绝、不发布旧提案。以上均为离线控制，不是DeepSeek新结果。

## 实际验证

新增23项测试：初次缺inspect_draft RED；自查None回调误记尝试与task错误标记repr回显，先2RED再提前client拒绝/隐藏repr；评测漏记首段失败先KeyError RED再补独立计数。旧测试适配初次漏ModelReply导入造成3NameError，补导入后通过，未改产品规则掩盖它。Ruff仅4导入/unused修正。

相关167通过；全量991（含本地练习14）通过，17.73s。Ruff通过、全目录136格式通过，离线build通过，sdist149/wheel47文件成员；私有路径成员0、指定凭证模式候选0。扫描不是所有个人信息的完备保证。

单次独立审查0 Critical/0 Important/0 Minor：167相关＋12新增探针通过。独立覆盖metadata5变体、surrogate/UTF8字节超界/bytes停止；65536字节注入样式raw仅作为user JSON数据；真实pending确认的原证据提交；最终不能引用只在坏raw出现的品牌；坏draft在途root改变第二发送前停止。无API、未改源码，没有实测后复审或模型效果结论。

本增量只改3源码（analyst/interaction/interaction_evaluation）、2旧程序测试并新增1测试文件，114基线其他109保持。旧fixture/decoder/存储/用途许可实现/依赖不改。

## 具体例子与限制

用户本次提三星，但草案只引用不带品牌的“我喜欢”：原decoder仍entity_evidence拒绝；这次宿主可让最终角色重新读完整实际原话，生成有合法品牌证据的新候选。只有它通过校验并经commit后才写盘。若最终还缺证据，整轮不写，不通过裁剪或补品牌来骗过守卫。

这个接口使专职角色有机会纠错，不证明模型一定纠正。没有重跑上一59请求，旧direct15/18、analyst16/18、整段1/3与查看请求错误全部保留。首段曾用过的tokens也不会因为最终成功而消失。

ISS-050的程序入口已补；剩余真实语义缺口仍为“给我看看商品”与品牌观点咨询的区分、保持旧限制不能新提态度、单次人物新Span等。下一项做这组业务契约的边界修正，再另冻完整验收与当前树逐要求结项审计；不反复同集追分，不提前进入Pi/排序或广告。

证据：`.superpowers/sdd/2026-10-22-phase3-invalid-draft/`的baseline/changed-files、原review.md/review_probe.py、author-audit及ledger；docs/ISSUES.md的ISS-050追加。
