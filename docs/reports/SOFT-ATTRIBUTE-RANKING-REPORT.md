# 软属性实际排序与独立消融

> 公开整理版：保留工程方法、实验结果与失败记录；本机路径已泛化，教学和内部协作记录留在本地。原始数据、模型和运行日志不随仓库发布。

2026-10-24，Task5J。补已保存字段的实际消费，不代表新学习模型或真实推荐收益。

## 评分规则与作用域

同一属性所有谓词合取，已知满足+1、不满足−1、缺失/未来0；按不同属性平均（未知也在分母）。
score=normalised_base+soft_weight×(brand_signal+attribute_signal)，复用原0.75，不新增已调优权重。
原preference_signal仍是品牌信号；新attribute_preference_signal单列，Pi程序显示软属性信号。
谓词/来源全局只列一次，item用索引/状态/事实来源关联，避免复制原文。

硬条件、排除和反馈仍先过滤，soft不滤商品、不取消硬预算。当前同属性硬/软要求覆盖LT soft，
当前soft不能覆盖LT hard；relax仅硬。长期soft须self+memory-on+用途许可；current软四组同享，
本人同轮保存不重复计分，他人只留会话，新会话不继承。use_soft_attributes=false可独立关闭评分。
confidence未校准不乘入，max_memory_age仍可选。未知中性可排在已知不喜欢之前，需评测此取舍。
显式数值/枚举/布尔谓词受支持，不发明“越轻越好”的阈值或连续效用函数。

正式ranker、memory_product_tool、离线replay接受soft_preferences/use_soft_attributes；
soft_preferences复用RequestConstraint完整绑定容器，不是把硬预算改成软要求。
schema4的is_hard=false已在本轮和跨会话消费；新的仅本次软要求语言协议尚未扩展。

## 实际证据

新增23Python：首批13中10实际RED（缺消费/新参数）3旧边界已过；正式子进程新入口、会话本轮source
及工具反馈令牌再有3有效RED；本人旧本次属性硬/软删除重放两项DID NOT RAISE→修复GREEN。
新增2Node实际Pi渲染和越界信号拒绝，分别原输出无软属性说明/Missing expected rejection有效RED→GREEN。
多谓词/属性平均、soft开关类型、他人RAM等其余补充检查原实现已满足，不称所有23逐条先红。

原“不变成硬过滤”旧测试用ABC顺序作断言，新增软排序后失败；改为只核商品集合不变，
新手算用例独立核B/A/C和1.25/.25/0。实验标签/旧报告不改。
ISS083独立工具发布令牌补反馈摘要；ISS084本人旧本次属性按宿主available_at核删除水位。

完整1536Python/50.22s后串行100Node/45.131s、Ruff212/TS/Python编译通过。
早先1534/49.56s为删除重放修复前，原日志保留；最终看python-final-source.log。
离线wheel75/sdist262、指定私有路径/凭证绝对路径候选0、新模块包含、README吻合；
probe前全部源/测试摘要打包再核未变。未独立最终复审。

固定供应商实际schema4宿主探针：长期软价格保存1条→重启→memory-off ABC/on BCA，
独立soft-off恢复ABC；随后当前硬预算2000仅C。四组只读不变磁盘/分析，全部popularity，
总6固定回调、实际HTTP0。另实际Pi消费软属性信号并展示1.000，不能将综合分变化误报为品牌偏好。
见artifacts/soft_attributes/20261024-ranking的probe-result/source-before/测试日志/package-audit。

## 结论与剩余范围

只证明软属性从保存进入排名及消融接线，规则/0.75未证明最优，未知/时间/身份语义不代表全部语言可靠。
原真实模型40例的冻结源码与所有267/30/54/28付费结果保持原样，不因新实现重算覆盖；预留test未评。
本轮无真实Key读取/模型调用/训练下载/依赖变化/推送。

下一先设计可信别名的原文与规范实体分离，不能只替换字符串而破坏证据引用、修订与删除；
仅本次软语言、同品类自然语言真实画像、一次最终test与整体独立终审仍待完成，goal active。
