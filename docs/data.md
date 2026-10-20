# 数据来源与契约

## Amazon Reviews 2023

- [官方站点](https://amazon-reviews-2023.github.io/main.html)
- [发布方数据集](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023)
- [发布方使用条件讨论](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/discussions/1)

固定类别 `Musical_Instruments`，revision `2b6d039ed471f2ba5fd2acb718bf33b0a7e5598e`。获取的是该类别两份完整文件，再在本地选择用户子集；不是服务端只下载 1/16。

| 文件 | 字节数 | SHA256 |
| --- | ---: | --- |
| `Musical_Instruments.jsonl` | 1557845107 | `63aee24d5b9eaf0069cf638703e2d62eb80fde5d2c00ca3d29881d645f6214b3` |
| `meta_Musical_Instruments.jsonl` | 631877970 | `96f40b5440918f07e3e08e84b7a6874a06a5e32a9988998ca9520cfd1b51a12c` |

总计 2,189,723,077 字节，脚本默认总预算 2.5GB。下载流式计算摘要，完整匹配后才发布完成文件；已有文件不覆盖。来源 URL、固定版本和校验记录保存在本地产物中。

发布方面向研究，但说明其无权为数据授予许可证；代码仓库的许可不替代数据权利。本项目不再分发原始评价、商品图片、用户标识、处理事件或模型。使用前应自行核对发布方条件，不能据此推断商业许可。

## 事件与时间

必需字段 `user_id`、`parent_asin`、`asin`、`rating`、`timestamp`。前三者为非空字符串，评分为有限 1–5 数值，时间为整数 Unix 毫秒；布尔值与不合理时间范围拒绝。原始 JSONL 的 `details` 是对象，不使用另一转换格式猜测类型。

`item_id = parent_asin`，`variant_id = asin`；来源文件名和从 1 开始的行号保留。评论正文、评论标题、有用票数和购买验证标记不进入当前训练或偏好推断。

历史视图只使用截止前事件，再按用户/父商品选择最新评分。同时间同评分稳定合并；同时间不同评分明确报错。未来更新或冲突不得改写过去。所有历史评分，包括低分，都进入已见过滤。

## 用户与候选

完整 `SHA256(user_id)` 整数取模 16，选择桶 0，不按全时期活跃度筛用户。截止 2023-01-01，结果为 179,630 条事件、106,129 位用户、54,277 个父商品。

候选只按训练期独立评价用户数降序、字符串物品 ID 升序取 5,000；验证与测试使用同一白名单。资源上限为 20 万事件、15 万用户、5,000 候选；超限拒绝，不静默截断。

## 展示快照

仅输出可取得的标题、类别与来源位置。缺字段保持未知，不因元数据缺失删除候选。元数据在历史上的生效时间未知，不能把下载时间当成生效时间。没有实时价格或库存承诺。

## MovieLens

[MovieLens 100K](https://grouplens.org/datasets/movielens/100k/) 使用官方 `u.data` 与 README，时间戳为 Unix UTC 秒，与 Amazon 的毫秒单位分开。下载脚本保存来源和摘要，不读取人口信息。研究、引用和再分发须遵守官方 README。
