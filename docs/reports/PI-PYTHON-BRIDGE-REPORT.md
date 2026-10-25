# Python只读工具桥接交付报告

> 公开整理版：保留工程方法、实验结果与失败记录；本机路径已泛化，教学和内部协作记录留在本地。原始数据、模型和运行日志不随仓库发布。

2026-10-22。用户已采用实施计划并授权当前目录逐任务实现；原Phase3结项保持。本报告记录首个正式Python增量，独立单次审查已完成（0C/1I/0M）；作者修复Important并全量回归，无二次复审。

## 本次实现

- 新`src/memrecagent/agent_bridge.py`：CatalogBridge.handle、严格health输出契约、成功ID回执快照、二进制serve_bridge及模块CLI。
- 新两测试文件：35实例测试＋23流/真实子进程测试（初版21，审查后新增2）。复用既有test_product_service.service的三商品实例，真实save_bundle/load_bundle；没有固定成功字典替代服务。
- 公开README/architecture/reproduction新增实际边界和运行说明，本地ISSUES/PROGRESS/ROADMAP/DECISIONS/EXPERIMENTS/LEARNING更新。没有改旧服务/记忆/测试/数据/锁文件。

## 原理和已交付边界

可信宿主CLI绑定目录→load_bundle核身份/哈希→串行JSONL→严格信封→真实health→固定形状回执并立即flush。r1重发语义相同仅重放，异method/params拒绝；256成功ID后新请求拒绝，旧回执仍可取。缓存序列化快照，不被返回值或后端容器后续修改污染；失败不占成功ID，不自动重试。

输入每帧最多65536 UTF8字节（不含LF/CRLF），有界读取；重复键/NaN/Infinity/坏UTF8/解码递归错误脱敏。普通错误继续，超限或无LF截断回执后退出2；空EOF退出0。bundle启动失败stdout空/stderr固定码。模型不能指定目录或用户，也没有记忆写入口。

## 实际验证

统一前缀`rtk proxy env UV_CACHE_DIR=./.uv-cache`，Python命令通过`uv run --locked --extra ml --extra api`。

| 检查 | 实际结果 |
| --- | --- |
| Task1缺模块RED→实现 | 35通过 |
| Task2缺serve_bridge RED→实现 | 新增合计56、两新文件＋服务/包相邻共67通过/1.89s |
| 最终默认pytest -q | 1037通过/20.19s（初版1035/19.41s保留） |
| 最终pytest -q exercises | 14通过/1.23s |
| ruff check . / format --check . | 通过/140文件格式一致 |
| uv build --offline | sdist与wheel成功；曾误加no-build-isolation失败，见ISS-053 |
| 构建包检查 | sdist154/wheel48文件；均含agent_bridge；指定私有路径0、凭证模式候选0 |
| 基线摘要 | 已有src/tests/scripts/exercises及pyproject/uv.lock无变化 |

真实Popen证明stdin仍打开时及时返回；请求路径拒绝、启动后盘上bundle破坏不影响已绑定实例；缺包/坏包仅固定错误。精确65536/65537边界、中文/CRLF、有界读取、逐帧flush、错误后下一请求和尾部不执行均已覆盖。

### 问题与审查

ISS-052：作者误以为2000层数组必触发JSON递归限制；首轮66通过1失败，独立复现本机2000能解析、20000才递归错误。修测试参考后67通过，未改产品去迎合。ISS-053：误禁用构建隔离导致缺hatchling，恢复声明驱动隔离构建成功，未改依赖。原失败保留。

独立审查2026-10-22首次因用量限制未执行；2026-10-23恢复同一reviewer完成，0C/1I/0M。独立相邻67/2.33s通过，150基线一致，另做真实pipe/子进程和异常数值恢复探针；未独立跑作者全量。原审查记录保存在`.superpowers/sdd/2026-10-22-pi-python-bridge/review.md`。

ISS-054 Important：readline(65539)在发送方给65537/65538个非CR/LF字节后保持stdin打开时继续等待，已确定超限却无回执。原BytesIO有EOF/后续帧掩盖了此行为。作者新两真实Popen测试先都失败（21.62s），改为readline(65537)，只有最后CR可能构成合法最大CRLF时补读1字节；相邻69/3.12s通过。当前总新增58，全量1037＋练习14；原0C/1I/0M不改，不称独立复审PASS。

审查暂不判断的Pi预算/取消、未完成合法前缀等待、跨重启/并发/恶意同进程、旧记忆语义与收益均按明确范围保留。超限即时拒绝属于本轮，已处理，不能交给未来客户端超时掩盖。

## 未实现、效果与下一步

API0、无安装新依赖/训练/数据下载/记忆写入/提交推送。仅Python协议和进程，未接正式Pi循环、模型第二轮、客户端5秒/30秒截止、取消或迟到回执拒绝。58项新增工程测试不证明模型识别、真实用户可靠性或记忆推荐收益；原模型及spike结果不改。

下一最小增量是正式Pi客户端离线循环与预算/取消/关联回执，再有限真实工具流。记忆Hook与排名特征后续逐步接，消融需冻结相同模型/候选/预算，当前不执行。
