# Phase3用途控制接口报告

> 公开整理版：保留工程方法、实验结果与失败记录；本机路径已泛化，教学和内部协作记录留在本地。原始数据、模型和运行日志不随仓库发布。

2026-10-22。ISS-049明确需求落实：用户可控制个性化及未来广告用途。本轮只实现本地设置/宿主门控，API0；不新增广告、前端、HTTP、排名或Agent循环，不提交/推送。

## 实现与语义

新增 `memory_use_policy.py`，`MemoryUsePolicy(personalisation=True, long_term_updates=True, advertising=False)`三项必须为真布尔。`JsonMemoryUseStore`每用户hash路径、严格版1JSON、原子替换；设置保存在记忆root下use-policy目录，不混入偏好事实，不存真实原文或Key。每次set新revision，同值重设也废弃旧票据；当前许可立即生效，不尝试按历史as_of回放过去授权。

ConversationMemorySession.get_use_policy/set_use_policy供可信宿主使用，模型没有改设置工具。set成功后清空原话/人物/弱线索/待确认/提案；其他实例下次访问发现revision变化先清理并stale。首次默认不创建文件，设置重启保持，用户隔离。损坏设置拒绝而不恢复为宽松默认。

personalisation=False：旧本人持久偏好不进入独立角色memory_view、咨询比较背景或推荐策略，不因旧排除触发个性化约束；当前明确请求和他人当前会话仍能处理。用于修订一致性、原证据重放和删除水位的内部存储校验仍可读文件，不将“禁用个性化”虚构为“绝不打开记忆文件”。这也不等于删除历史。

long_term_updates=False：自动提案及提交路径不发布新长期偏好，实际当前表达可作为本次语境；它不改变自然语言候选的原语义，不自动把long_term改current。用户直接查看、纠正、删除属于管理权限，不被自动更新开关封锁。精确目标的用户点击反馈为另一套显式操作，不从这个开关反推反馈许可。

purpose默认recommendation。未来advertising用途须额外advertising=True且personalisation=True，否则在调用模型或提供上下文前拒绝；普通推荐许可不扩大到广告。当前只是记忆用途边界，没有实际广告投放路径。直接resolve/propose/结构化commit及双阶段Hook使用同一设置，不能只隐藏模型字段而仍用旧偏好决策。

## 一个完整例子

原先本人长期排除小米。宿主执行：

```python
from memrecagent.memory_use_policy import MemoryUsePolicy

session.set_use_policy(MemoryUsePolicy(
    personalisation=False,
    long_term_updates=False,
    advertising=False,
))
```

后续“给我推荐手机”不调入旧小米排除；“这次看看小米”仍可作为当前明确要求，磁盘原记录保持。新的“以后偏好三星”候选不会自动写长期。重建同用户session仍读到这组关闭设置；手动删除小米记录仍可用。要重新启用由用户/可信宿主显式设置，不能把用户喜欢某品牌当成广告授权。

## 发现的问题及处理

初次新测试缺模块RED；初版29通过/1失败是测试将共享os.replace故障注入留到后续偏好提交，恢复patch后行为正确，未为此改产品。

作者自查新增2项RED：广告会话撤销自己许可后，重建被禁用宿主使setter报失败；低层propose仍能返回被禁用的长期提案。修复为清理后暂不创建禁用内部宿主，所有使用入口仍先拒绝purpose；propose在入口与结果统一遵守许可。之后37项通过。

单次独立审查0 Critical/1 Important/0 Minor：模型回调在同一实例set_use_policy会刷新实例revision，prepare局部旧turns却仍在；旧分析返回时fresh只比最新实例版本，关闭后甚至关开ABA仍可能返回票据/写入。独立原12探针9通过3RED（含direct/final旧提交和draft额外发送）；结合作者37项＋新增6RED，原审查命令46通过9失败。不是独立PASS。

作者新增direct/draft/final×关闭/关开ABA六项先全RED，再把调用开始的policy_revision绑定到direct返回、draft后、第二发送前、final后及失败finally。修复后43新项通过，相邻87通过；作者重放原12审查探针全部通过。没有第二次独立复审，原报告与负结果保留。

## 最终验证与限制

全量tests＋本地exercises共968通过，17.49s；Ruff通过、全目录135文件格式通过；离线build通过，sdist148/wheel47文件成员，私有路径成员0、指定凭证模式候选0。扫描不是对所有个人信息的完备证明。

本增量新增1源码1测试，只改4既有源码：memory_policy/proposals/commit/interaction；112基线中的其他108保持摘要。decoder/提取提示/分析器/旧数据/评分/依赖不改；无新的DeepSeek调用、没有更新模型15/18或16/18结果。

单进程串行本地接口；不是互联网身份认证/跨进程事务或用户设置UI。已经被允许发送的调用无法在网络上撤回，许可修改后禁止继续使用旧响应或发送下一阶段旧上下文。手动管理和可信宿主仍可直接操作本用户记录；不把Python模块边界称恶意同进程防护。

ISS-049程序接口已补并作者修复独立Important，真实用户交互与未来广告接入仍待相应阶段验证。完整Phase3未完成：ISS-050无效草案阻断最终角色，以及商品查看意图/维持旧限制误提的新实测缺口仍开放。下一增量处理独立重读入口，最终严格提交守卫不放宽，再做当前树结项审计。用户无新增独立学习代码证据。

证据：`.superpowers/sdd/2026-10-22-phase3-use-policy/`中的plan ledger、原review.md/review_probe.py、baseline/changed-files/final-audit。问题过程见docs/ISSUES.md的ISS-049追加。
