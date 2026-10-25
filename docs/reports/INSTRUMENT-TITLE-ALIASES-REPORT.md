# 乐器完整标题到原生商品ID的可信映射

> 公开整理版：保留工程方法、实验结果与失败记录；本机路径已泛化，教学和内部协作记录留在本地。原始数据、模型和运行日志不随仓库发布。

2026-10-25。延续用户已选的现有乐器路线；上一轮仅精确ID实体，本轮使完整自然商品标题可被现有宿主消费。
无新API、训练、下载、依赖、提交或推送，不使用原测试剩余10次额度。

## 映射规则

新 build_instrument_title_aliases 先由正式 load_bundle 校验包，再读取既有目录标题；不读取评论或未来购买标签。
标题必须在全部5000商品中按NFKC/casefold/strip后唯一，且属于已确认乐器根类。
标题不能夺取另一个商品的原生ID；重名全部跳过，不按热度或模型分挑一个代表。
原始完整标题作为别名、原生item_id为规范键，来源绑定bundle/catalog/item/title；不猜品牌、系列、缩写或简称。
长于256字符跳过，不截短；未知类别不参与。未登记的标题保持原文、不自动解析，商品仍可按原生ID使用。
本增量没有自动歧义澄清或模糊名称检索；未知原文仍沿用既有候选协议，不能声称任意名称都能匹配目录。

复用已有 EntityAliases 的安装、digest绑定、修订、删除/换表守卫及全部消费者。
有界容量由1000实体/256KiB调整为5000实体/2MiB；别名声明最长256字符以对齐提取器实体上限，规范ID仍最长128。
旧schema1与现有小表的序列化/digest不变；没有旧库迁移或热更新。
报告记录usable_after为包创建时间，仅用于该时间之后的新命名空间；这不是可按as_of查询的历史别名库。

## 实际目录覆盖

| 项目 | 数量 |
| --- | ---: |
| 原商品 | 5000 |
| 登记唯一完整标题 | 4871 |
| 跳过重名商品 | 24 |
| 跳过超过256字符的标题 | 3 |
| 跳过未知/其他根类 | 102 |
| 导出表字节 | 1464961 |

覆盖分母始终是原5000项；过滤的是别名声明，不从原候选目录删除商品来提高指标。
报告与实体表：artifacts/domain_alignment/20261025-title-alias-v1/{report.json,entity-aliases.json}。

## 完整消费链验证

新增5项目录/容量/CLI测试及3模式完整宿主场景：
完整标题长期排除→新会话推荐过滤→标题咨询激活只读视图→本次标题例外不改磁盘→原生ID改口接续同一修订头。
旧/新证据文本精确保持用户原文；没有用规范ID改写标题证据。
新增builder接口缺失收集RED及CLI文件缺失RED后实现；完整链用例验证现有消费者复用，不宣称每个断言都逐项先红。
相关24项通过。全套验证结果完成后记录于下文及verification.json。

另用真实乐器包的4活跃BPR画像+2合成冷用户、完整4871标题表重复同域机制路径：
36次固定模型回调，24个四组排名条件；活跃算法臂实际BPR、冷组显式回退热度，跨会话标题排除生效，比较不改盘。
包SHA前后不变；18个用户轮本机处理耗时中位0.8816秒、范围0.8193–0.9418秒，不含真实模型延迟。
该计时只作运行开销观察，不能与上一轮不同进程/缓存条件作因果性能比较。
商品目标取原算法首位，是反事实机制验收；没有真实用户标签或独立相关性目标，因此不计算NDCG增益。

## 导出与安装

在项目根目录先导出到一个不存在的新目录，默认不写记忆根目录：

```bash
rtk proxy env UV_CACHE_DIR=.uv-cache uv run --offline --locked --extra ml --extra api python scripts/create_instrument_aliases.py --bundle artifacts/product_bundles/20261020T051100515587Z-e27333ad --output-dir /tmp/instrument-title-aliases-new
```

查看report.json与entity-aliases.json后，沿用已有配置CLI预检；安装只允许新空命名空间：

```bash
rtk proxy env UV_CACHE_DIR=.uv-cache uv run --offline --locked --extra ml --extra api python scripts/configure_entity_aliases.py --root /tmp/instrument-memory-new --input /tmp/instrument-title-aliases-new/entity-aliases.json
rtk proxy env UV_CACHE_DIR=.uv-cache uv run --offline --locked --extra ml --extra api python scripts/configure_entity_aliases.py --root /tmp/instrument-memory-new --input /tmp/instrument-title-aliases-new/entity-aliases.json --initialize
```

后续Pi聊天同时使用同一memory-root、原生ID事实映射与 --category musical_instruments。
本轮未启动live聊天，未读取Key或向第三方发送真实目录标题；真实模型验收需另准备有限调用范围。

## 未完成范围

自然语言理解、短名/别名歧义处理、乐器属性/预算协议、独立效用标签及真实用户满意度仍未完成。
旧手机开发/预留测试与全部失败保留原口径；固定响应通过不能替代真实模型消融或优于传统记忆的证据。

工程验证增量：完整1626Python/53.98s通过，Ruff221文件格式/检查与Python编译通过，TS build通过；Node完整串行103/103通过（47.37s）。作者核验不是新增独立代码复审。

最终离线交付包wheel78/sdist277文件，新增源码与导出脚本匹配当前版本，五份公开文档字节一致，指定私有路径候选0；verification.json保留检查范围与源摘要。
