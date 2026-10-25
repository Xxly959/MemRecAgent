# 乐器同领域接入：可信品类与精确商品实体

> 公开整理版：保留工程方法、实验结果与失败记录；本机路径已泛化，教学和内部协作记录留在本地。原始数据、模型和运行日志不随仓库发布。

2026-10-25。用户已选择“复用现有乐器数据，接入同领域语言记忆（无需新训练下载）”。
这是该路线的首个产品增量；不是完整项目结项，也没有新增真实模型调用。

## 行为与边界

ConversationProductSession / ConversationMemorySession 新增可信宿主 category，默认 phone 保持旧调用兼容。
品类贯穿单次分析、草案、独立最终角色、相关记忆视图、咨询背景、提交前守卫和实际排名。
模型不能通过输出字段改变宿主品类；错品类整批拒绝，不改写证据来迁就目录。
MemoryView 有自己的品类绑定；手机后缀规则仅对 phone 生效，乐器按完整实体或已配置的可信别名匹配。
单次/双阶段/隐藏草案三种模式均覆盖，不能仅改最后一条推荐 query。

Pi ConversationClient、chat-cli 与 Python conversation_bridge 支持 --category musical_instruments，
品类是启动参数，不是交给模型的工具参数。旧 phone 默认消息保持原字节；非手机系统提示明确覆盖手机示例。

新 instrument_catalog / create_instrument_facts 从 load_bundle 已校验的包生成完整事实映射：

- entity=item_id，明确是商品实体，不从标题猜品牌或系列。
- 原根类 Musical Instruments / Instrument Accessories 映射 musical_instruments，共4898件。
- 102件未知或其他根类为 catalog_unknown，保留在完整映射中，但不进入乐器查询。
- 来源绑定 bundle ID / catalog / item ID；available_at 取包创建时刻，不回填为训练截止时已知。
- 没有新增价格、库存、品牌和属性事实；历史目录不能当实时目录。

生成 facts.json 为1,314,692字节，满足实际桥接2MiB上限。
目前非手机属性候选会明确失败，未完成乐器预算等属性协议；不能宣称已支持这些要求。
后续应使未知价格硬预算走明确不可满足/澄清路径，避免将手机属性例子直接迁移。

## 工程验证

新增12项Python行为测试，覆盖跨会话过滤、相关视图、三模式错品类拒绝、他人不继承本人排除、
不套用手机属性协议、缺失目录品类启动前拒绝、映射来源/可用时间/未知类目与坏包拒绝。
第一轮测试收集错误来自测试函数默认参数，已修正；随后7项因缺category接口RED，再实现通过。
映射测试初版遗漏夹具商品C，修正夹具后通过；这些不是产品语义错误。
Node新增实际子进程argv传递与空品类拒绝覆盖；首版worker回执协议错误已修正，旧失败不当成功。

Python全量1618 passed / 52.79s；Ruff通过、219文件格式通过，Python编译通过。
Node首次完整102/103，成功分析四组用例返回error（耗时257.27s）；未记录具体错误码，不能断言原因。
原样独立该文件3/3通过（10.16s），未改源码/阈值/断言；随后原样完整串行103/103通过（47.58s）；首次失败继续保留，不认定已定位根因。
离线sdist/wheel构建成功；最终文档更新后的交付构建另存，保留首包。
无依赖变化、训练、下载、提交、推送；不是再次独立代码复审。

## 同域真实参数的固定响应验收

产物 artifacts/domain_alignment/20261025-instrument-v1：facts.json、check_native.py、native-result.json。
复用原乐器包，不修改模型向量或seen；按原生ID排序选前4个活跃BPR用户，另2个合成冷用户。
每个用户从空库开始：先查询 → 明确长期排除一个原算法首位商品 → 新会话查询 → 同请求四组只读排名。
共36次固定模型回调、24个排名条件，HTTP/API=0；反事实偏好由作者构造，不是真实用户偏好。

- 4个活跃用户的算法组实际使用bpr；2个冷用户算法组使用显式bpr_cold_popularity路由回退热度。
- 重启后长期排除实际生效；记忆开组各过滤一个目标，关闭组不施加该排除。
- 四组共享已接受的请求/人物/数量，比较不改记忆文件。
- 原包文件SHA256前后相同；未读原评论、未来购买目标，也未向第三方发送原生用户ID。

首版探针误认为裸bpr自动冷回退，在冷用户断言失败；保留 check_native-initial.py 与原memory目录。
正式复核使用原有显式 bpr_cold_popularity 策略及新memory-v2目录，不修改生产路由去适配探针。

该验收补上“实际同域记忆路径消费活跃算法”的机制证据。它不证明自然语言自动识别、NDCG提升、
协同增益或真实用户满意度；没有独立效用目标，因此不计算或宣传推荐收益。
原手机开发/test/260调用与所有旧失败保持原口径，不把旧手机故事改称乐器实测。

## 本机复现

先在项目根目录创建新文件，禁止覆盖旧映射：

```bash
rtk proxy env UV_CACHE_DIR=.uv-cache uv run --offline --locked --extra ml --extra api python scripts/create_instrument_facts.py --bundle artifacts/product_bundles/20261020T051100515587Z-e27333ad --output /tmp/instrument-facts-new.json
```

Pi聊天沿用 agent/dist/src/chat-cli.js 的全部既有必填参数，新增 --category musical_instruments，
--facts 指向上面新映射；user-id 必须是真正匹配既有模型的原生ID才有活跃画像，任意新ID只属冷启动。
使用新的独立 memory-root，不把手机命名空间或反事实验收记忆借给真实用户。
live聊天会消费API；本轮没有启动，不复用前次剩余10次额度。

下一项是冻结同域合成语言任务及调用内容/限额，然后在新有限授权后进行真实语言与Pi验收。
真实用户效用、独立相关性标签与完整系统对照仍保留为未完成要求。

最终交付核对：dist-final离线构建成功，wheel78文件/sdist275文件，新instrument_catalog模块与当前源码相同；
五份公开文档与当前字节相同，包内指定私有路径候选0。verification.json记录源测试脚本摘要与完整验证范围。
本轮没有新增独立审查，不以作者复核冒充独立PASS。自然名称尚须显式可信别名，不支持自动模糊识别。
