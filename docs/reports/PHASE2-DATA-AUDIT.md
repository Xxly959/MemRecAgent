# Phase2数据来源审计

> 公开整理版：保留工程方法、结果和失败；本机路径已泛化，内部记录及原始产物留在本地。

核查日期：2026-10-20。已执行的是官方元信息与有界前缀审计，不是全量数据检查或训练。

## 来源与使用条件

- 官网：https://amazon-reviews-2023.github.io/main.html
- 发布方数据集：https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023
- 发布方置顶许可讨论：https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/discussions/1
- 代码仓库：https://github.com/hyp1231/AmazonReviews2023

发布方2024-03-08说明主要面向研究，但无权为数据授予许可证或规定使用条款；2024-12-08回复学术论文使用没有问题。这不是覆盖评论、图片及商品数据的明确开放许可证。官方API的license字段为空；代码仓库MIT许可不替代数据权利说明。项目仅本地学习研究，不再分发原始数据、评论或图片；未来公开数据演示/商业使用另行核查。

## 类别与资源选择

| 类别 | 用户/物品/评价（官方约数） | raw评价/元数据文件体积（页面约数） | 决定 |
| --- | --- | --- | --- |
| All_Beauty | 63.2万/11.26万/70.15万 | 327MB/213MB | 获取较小，但每用户约1.11条，历史薄 |
| Musical_Instruments | 180万/21.36万/300万 | 1.56GB/632MB | 选定；实物/配件语义适合教学、规模适中 |
| Video_Games | 280万/13.72万/460万 | 2.68GB/437MB | 网络量更大；版本/平台/配件需额外辨析 |

以上统计见官网类别表与官方raw目录清单。不是当前真实切分统计，不代表哪个类别效果好。品牌/属性/目录关联完整率尚未全量验证。

本机M5、16GB内存、工作盘约309GiB可用。按Musical_Instruments约数，评分float32+bool掩码约1790GiB，一张物品float32矩阵约170GiB；不能直接使用全类别稠密适配器。按记录流式处理，限制候选与训练预算。

## 固定版本及实际样例

官方API返回revision=`2b6d039ed471f2ba5fd2acb718bf33b0a7e5598e`；最后修改时间2024-12-08T22:21:49.000Z，gated=false。两文件总计2,189,723,077字节，小于2.5GB预算。

| 官方文件（均在上述revision） | 实际字节 | 官方LFS SHA256 |
| --- | ---: | --- |
| raw/review_categories/Musical_Instruments.jsonl | 1,557,845,107 | 63aee24d5b9eaf0069cf638703e2d62eb80fde5d2c00ca3d29881d645f6214b3 |
| raw/meta_categories/meta_Musical_Instruments.jsonl | 631,877,970 | 96f40b5440918f07e3e08e84b7a6874a06a5e32a9988998ca9520cfd1b51a12c |

各请求Range bytes=0-262143，实际HTTP206与Content-Range核对成功，每文件262144字节。末尾不完整行不解析。评价365条完整记录，元数据79条；仅保存字段/数量/前缀哈希，不输出或保存此次前缀的评论正文。

- 评价实际字段：rating(float)、user_id/asin/parent_asin(str)、timestamp(int)、helpful_vote(int)、verified_purchase(bool)，另有评论title/text/images。本样例没有sort_timestamp。
- timestamp样本范围1278983148000–1678910663667，按毫秒转换为2010-07-13至2023-03-15 UTC；确认为本版本样例的毫秒字段，不以此保证所有行合法。
- 元数据实际details为dict；price为float或null；store为str或null。
- 79条中标题79非空、categories73非空、features65非空、details79非空、price42非空；样本均有parent_asin。
- 前缀样本不是随机样本；未验证跨文件关联覆盖、重复率、全量缺失率或拟议时间切分规模。

实际JSON证据：本地忽略路径data/amazon-2023/source-audit-20261020.json，含完整固定URL、远端大小/SHA256、前缀SHA256、Content-Range和字段统计。尚未完整获取这两份文件，不能把官方SHA256写成已本地复算。

## 获取与实验预算

两文件传输上限2.5GB；获取时逐块计算完整SHA256并匹配官方值，原有文件不覆盖。原始评论不纳入Git或公开包。流式解析节省内存，不等于节省整文件传输。

预定SHA256用户固定桶1/16；处理上限20万交互、15万用户、5千候选，超过停止审计。候选限量只能看训练期；完整新商品正例分母不因候选限量缩小。进程RSS目标4GiB，实际训练耗时未测。具体数据计数、可训练用户与日期可行性在取数审计后冻结，测试指标不得参与选择。
