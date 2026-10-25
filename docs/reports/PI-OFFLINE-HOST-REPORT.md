# Pi正式离线宿主交付记录

> 公开整理版：保留工程方法、实验结果与失败记录；本机路径已泛化，教学和内部协作记录留在本地。原始数据、模型和运行日志不随仓库发布。

2026-10-23。用户明确要求实际接入与记忆/推荐消融，已沿用native执行既有离线宿主计划。已完成实现与一次独立审查；原0C/1I/0M，作者修复Important并全量回归，不称独立复审通过。

## 已实现

正式agent目录锁定pi-agent-core/pi-ai各1.4.0，TS7.0.2、types/node22.19.19；npm安装88包（包含本机选中的optional），package-lock记录107条非根包及integrity。运行路径不依赖旧spike或Git源码，克隆仅源码核查。

strict-json原文键唯一校验；BridgeClient固定argv/shell:false、最小环境、可信bundle、严格UTF8/字节上限/ID/health；真实Pi Agent串行工具与完整批次守预算；独立截止、取消/迟到防交付、最终状态守卫；offline-model消费真实toolResult；CLI只离线目录状态。测试fixture在临时目录构建固定向量包，不训练。

## 实际验证

- Task1缺模块RED→20Node通过；依赖未就绪的tsc不存在不算RED（ISS-056）。Task2缺catalog-agent编译RED→34通过；CLI缺文件5RED→39通过。补边界和最终守卫后45Node通过/5.076s；审查修复后47Node通过/5.489s。
- TypeScript build通过；最终Python默认1037/19.74s＋本地练习14/1.11s；Ruff通过/141格式；离线sdist/wheel成功。
- 包sdist169/wheel48，指定私有路径0/凭证模式候选0；153既有源/测试/脚本/练习/依赖摘要不变，仅新增测试合成包helper。计划旧150计数未含前一桥接3文件，按实际153核验，不减少保护范围。
- 单次独立审查0C/1I/0M：独立build+45Node、153基线通过；忽略SIGTERM约257ms强杀，CLI取消约265ms固定cancelled/2、子进程消失、stderr空。未独立跑Python全量/供应链审计。原报告见.superpowers/sdd/2026-10-23-pi-offline-host/review.md。
- ISS-057：首成功回执后重复帧/EOF只有客户端终态，宿主第二模型仍等总超时。作者两真实坏管道/Pi测试先RED约1004ms，新增subscribeFailure持续通知、宿主fail保首码，正常close不通知；GREEN约153/248ms，最终套件约229/345ms，第二模型及时abort。全量已回归，无二次复审。
- 审查暂不判断的真实供应商/物理HTTP、记忆排序收益、独立Python全量、包源码一致及漏洞状态继续明确保留；不能借离线绿色宣称上述成立。

## 可见产物与原理

本机完整trace在artifacts/pi_offline_host/delivery-3r3hlzlu/final-trace.json（原trace.json也保留）。事件顺序：model_request→tool_execution_start→bridge_receipt→tool_execution_end→tool_result→model_request→delivered。实际2次离线模型函数、1工具，3商品，bundle_id products-83d76c8ce38cab71410b7c8f5f202fa509bb398c267025242a1d0f7956c969e1，历史时点1656633600000。

这是真实Pi循环与Pythonhealth，模型供应商由确定性streamFn替代，回答是依据回执生成的固定状态说明。没有DeepSeek HTTP/记忆/商品推荐。每请求默认最多2模型函数/2工具、整体30秒/单工具5秒（含启动），关闭宽限250ms；离线函数计数不等于真实HTTP计数，未来供应商需另核重试。固定说明精确比对是本只读目录工具的交付守卫，不能拿它冒充任意推荐问答。

## 任务定位及下一步

[用户目标与四组消融](PI-MEMORY-RECOMMENDATION-TARGET.md)已记录。接下来真实供应商工具流、记忆Hook与实际排名接入；复用已有实现，先一条完整跨会话例子再冻结四组实测。是否更符合偏好由保留的完整回答和评分决定，不预设每次都赢。

API0、不训练/下载数据/写记忆/提交推送。源码/依赖安装不算推荐效果；
