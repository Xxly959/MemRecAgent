# 复现指南

## 本次明确数量

显式live聊天仍使用原入口。自然请求如“预算三千，推荐两款手机”“改为三款，给我推荐手机”
不新增工具参数：宿主在已接受推荐动作中绑定最新原文，常见明确1至10款，四组复用实际k。
数量不是长期偏好；候选不足会说明，范围/冲突/超限需完整重述，先前未提交的新要求要一并重述。
没有明确数量保留受信fallback k，完整的语言准确率尚未验证。

本机固定工程轨迹在`artifacts/result_count/20261024-request/`，可用
`pytest -q tests/test_result_count.py`检查数量、临时证据和原子不写；Node完整串行检查包含真实Pi与持续Python，
外部模型固定，不能当真实模型成绩。细节与限制见本地RESULT-COUNT-REPORT.md。

## 完整历史与字符TF-IDF的LLM基线

本机准备的15个合成开发任务位于`artifacts/llm_baselines/20261024-dev/suite.json`，
拟发送消息见相邻`message-preview.json`；以下只预检，不读Key、不调用模型或创建输出：

```bash
uv run --offline --locked --extra ml --extra api python scripts/evaluate_llm_baselines.py \
  --suite artifacts/llm_baselines/20261024-dev/suite.json \
  --output-dir artifacts/llm_baselines/new-run
```

真实调用仅在数据与本次有限额度已授权后，加`--live --max-calls N`及环境Key；脚本不读dotenv，
每任务预留完整2次、不重试、拒绝覆盖；默认split=dev，预留test不随开发运行。
本次首测使用相邻`source-before-calls/`快照和冻结输入，未将等待额度期间的隐私修复混入实验。
CLI不是新的记忆写入路径，也不放开生产Pi重排权；其JSON只能含不重复/不越界/最多K个item_ids。
实际30调用已完成且拒绝重启/覆盖；新实验须另有输入版本及有限授权。完整契约、历史来源、限制、成本和负结果见本地LLM-SYSTEM-BASELINES-REPORT.md。

## 完整候选开发集敏感度

输入是已核验的完整候选特征，不是TopK列表。本机首测输入位于
`artifacts/ranking_sensitivity/20261024-dev/features.json`，无外部下载；以下只预检、API0、无输出目录：

```bash
uv run --offline --locked --extra ml --extra api python scripts/evaluate_ranking_sensitivity.py \
  --suite artifacts/ranking_sensitivity/20261024-dev/features.json \
  --output-dir artifacts/ranking_sensitivity/new-run
```

显式加`--run`才扫描，新目录拒绝覆盖，始终不调用模型或读Key。schema1/evidence_kind=saved_response_replay、
split必须dev；cases包含case_id/cluster_id/cohort/reference_status/reference/frames。
frames为minmax/rank各自完整四组，含实际status/used_strategy/algorithm_effective、n_candidates/n_filtered及
全部过滤后items（item_id/base_score/normalised_base_score/preference_signal）。ready组必须满足
商品数=n_candidates−n_filtered，两种归一化的候选/基础分/信号相同，拒绝截断帧。
参考仅评分，不改变排序；错误留分母。当前固定网格与默认值不做选择，见本地RANKING-SENSITIVITY-REPORT.md。

## 同一实际请求的Pi只读分支

正式 `ConversationClient` 正常 step 成功推荐后，用 `comparisonBinding(signal)` 获取宿主绑定，
再以 `comparisonConnection(owner,binding,{useLongTermMemory,strategy})` 接入
`runConversationTurn`。prompt 必须等于绑定的实际原文，不能用 gold 重建 query。
分支不重复分析/提交，不能 observe 反事实；结束后调用方统一关闭 owner。
新用户输入、删除/用途或相关状态变化会使旧绑定拒绝，缓存命中也检查。

本次两故事输入和一次性脚本在本机 `artifacts/pi_comparison/20261024-live/`，
源码/编译JS/核心SDK/依赖锁均有首调前快照，详见PI-FOUR-GROUP-COMPARISON-REPORT.md。
launch拒绝覆盖或重启，不重复旧数据追分。该入口是受控实验脚本，不是新的通用付费CLI；
新实验须有独立输入/完整预算/冻结，预留测试未运行。不能以排名off冒充整个Agent无记忆。

## 共享一次草案的实际配对控制

从源码目录运行；本机18例作者模板语料位于 `data/evaluation/shared_draft_v1_final.json`，
不随安装包分发。下面只预检并报告54次上限，零模型调用，不建输出目录：

```bash
uv run --offline --locked --extra ml --extra api python scripts/evaluate_shared_analysis.py \
  --suite data/evaluation/shared_draft_v1_final.json \
  --output-dir artifacts/shared_analysis/new-run
```

语料为schema_version=1、synthetic=true、label_source及cases；每例含case_id、cluster_id、split、
runtime和references。runtime包含带双时间的实际turns、as_of、active_memories及pending_question_turn_id；
记忆使用完整PreferenceMemory.to_dict，归属固定synthetic-shared。references是严格协议候选的可接受列表，
必须能引用实际对话并通过正式解码，不进入模型；这里只保证结构一致，不证明人工真值。

live需已授权的合成数据及预算、环境Key，同时加 `--live --max-calls N`，拒绝覆盖既有目录。
整例预留三次尝试，剩不足三次不开始；无自动重试。源码、输入和实际请求/响应留档，
原文只存在显式合成实验产物，普通终端仅输出摘要。退出0表示配对完成，不表示全部候选匹配。
总物理成本只累计一次共享草案，不可将三臂逻辑等价成本相加。

本轮原始结果和限制见本地 `docs/reports/SHARED-DRAFT-CONTROL-REPORT.md`。
该入口不提交、不生成Pi回答，不能替代完整Agent或真实推荐效用评测；旧结果不覆盖、不重跑追分。

## 外部合成数据派生与实际重放

当前quality和external重放入口可选 `--analysis-mode analyst_blind`，两次调用上限与analyst相同，
只隐藏最终角色收到的draft及坏草案正文/错误码，相关记忆view仍由草案选择。
默认仍为计划零调用；此选项不是Pi聊天CLI开关，也没有新增模型成绩。
严格归因还须共享同一次真实草案及view，详见本地ANALYSIS-BLIND-CONTROL-REPORT.md。

原始文件留在 `data/evaluation/incoming/`，修订政策位于本机
`data/evaluation/policies/external-v1.json`。两者不随安装包分发。
以下默认只检查并预览，不调用模型，也不写输出：

```bash
uv run --offline --locked --extra ml --extra api python scripts/derive_external_evaluation.py \
  --stories data/evaluation/incoming/stories.jsonl \
  --tasks data/evaluation/incoming/recommendation_tasks.jsonl \
  --policy data/evaluation/policies/external-v1.json \
  --output-dir artifacts/external_derived/new-version
```

显式加 `--derive` 才保存新版本；既有目录拒绝覆盖。输出包含raw副本、源码、政策及
逐字段前后值、独立摘要和派生套件。政策只允许参考字段修订且必须匹配原值摘要。
新版本需在模型调用前确定，不能根据本次模型成绩调整标签。

重放默认同样零调用：

```bash
uv run --offline --locked --extra ml --extra api python scripts/evaluate_external_replay.py \
  --suite artifacts/external_derived/20261024-v1/derived-suite.json \
  --output-dir artifacts/external_replay/new-run --split dev --analysis-mode direct
```

真实模式必须在已授权的数据/预算范围内，环境提供 `DEEPSEEK_API_KEY`，并同时加
`--live --max-calls N`；不要将Key写进命令或输入文件。脚本不读取dotenv，拒绝覆盖输出，
没有自动重试。analyst每轮最多两次，direct一次，预留test应独立授权并只评一次。
报告 `successful_steps` 只表示处理成功，不能代替状态吻合、约束或效用检查。
四组共用实际已接纳请求，本数据无匹配算法画像；详见[评测限制](evaluation.md#外部合成集验收)。

## 结构化长期属性接口

可信宿主可构造`memrecagent.attribute_memory.AttributeMemory`，通过`JsonlMemoryStore.add_many`与品牌记录一次原子保存。长期记录只接受本人；数值/单位与`RequestConstraint`使用同一规则。用`list_attribute_memories`查看历史、`active_attribute_memories`读取指定时刻的当前头、`correct_attribute`显式修订，`delete_attribute`删除某类别属性的全部方向及硬软历史。旧品牌管理接口只处理品牌，不能代替属性删除。

排名默认加载获准的本人长期hard属性，关闭长期记忆或给他人推荐时不套用。当前`constraints`覆盖同属性同方向的旧值；可信本次解除使用`RequestRelaxation`，与用户/会话/受益人/request_id及证据时间完整绑定。两种正式JSON入口均接受`relaxations`列表，元素字段为上述四项身份、attribute、evidence_ref、available_at；available_at须早于查询时刻，不得与同属性当前新条件混用。聊天schema4另有attribute_changes列表，已接实际语言提取及受控长期提交/撤销；原文主体或范围不清时须完整重述。他人属性只留会话。软属性可保存但尚未计分。

## 显式live本地聊天

已有完整环境后，在项目根目录构建合成目录；示例目录须为空，数据明确为合成。聊天记忆根使用独立新路径，避免把预置夹具当自然语言学习结果。

```bash
uv run --locked --extra ml --extra api python scripts/create_memory_recommendation_fixture.py /tmp/memrec-chat-catalog
npm --prefix agent run build
node agent/dist/src/chat-cli.js --live --python "$PWD/.venv/bin/python" --bundle /tmp/memrec-chat-catalog/bundle --memory-root /tmp/memrec-chat-preferences --facts /tmp/memrec-chat-catalog/facts.json --user-id local-demo --session-id S1
```

Key仅从`DEEPSEEK_API_KEY`环境读取，不自动读取文件、不在命令行放Key。逐行输入，例如“我长期排除品牌A。我一直更喜欢品牌C。”；退出后改session-id重新启动，再输入“给我自己推荐手机”。同一用户的本人长期记录保留，亲属喜好不随进程继承。输出为JSONL，`result.finalText`为实际回复。

这是显式允许模型分析并受控保存本人明确偏好的入口，原只读提取CLI保持原行为。每轮最多4次供应商调用，最多64用户轮，无重试；原文/相关历史发往用户配置的DeepSeek。普通回答可自然生成，推荐顺序/数值由程序控制；未知商品事实应说明未知。固定k=10，尚不识别自然语言数量要求。

无回执中断时`analysisStatus`/`memoryCommitStatus`可能为unknown；空memoryIds仅表示已知ID为空，不能当0写证明。故障会终止会话，不自动重放。源码已有本地查看/纠正/删除管理工具；真实用户泛化、四组收益、完整独立审查仍未证明。

## 运行Pi记忆融合推荐

在项目根目录运行以下命令，目标示例目录须为空。只创建合成手机/明确本人偏好与固定测试向量，无训练下载；需要完整源码检出（示例复用tests）。

```bash
uv run --locked --extra ml --extra api python scripts/create_memory_recommendation_fixture.py /tmp/memrec-ranking-demo
npm --prefix agent run build
node agent/dist/src/recommendation-cli.js --python "$PWD/.venv/bin/python" --bundle /tmp/memrec-ranking-demo/bundle --memory-root /tmp/memrec-ranking-demo/memory --facts /tmp/memrec-ranking-demo/facts.json --request /tmp/memrec-ranking-demo/request.json
```

默认固定假模型、HTTP0；Pi和Python均真实执行，预期C/B，综合分.75/.5。请求文件绑定用户/会话/受益人、phone类别、查询时刻、策略/k；事实文件将商品与实体/来源/可用时间显式对应。不能把此例当自然语言提取或真实手机目录。

同一Node命令在文件名后加`--live`才接DeepSeek，Key仅从`DEEPSEEK_API_KEY`环境读取，不自动读取本机文件；最多2HTTP、无重试。将发送固定任务及推荐工具摘要（含品牌、分和证据引用，不含原始证据正文），默认不保存原始模型内容。当前用途开关控制持久偏好读取，不会删除已拟合CF画像；非本人路径则不使用owner画像。损坏manifest根类型的直接Python CLI traceback为已知限制，Node不透传。

工程回归可使用`node --test --test-concurrency=1 agent/dist/test/*.test.js`，避免与重型Python全量同时争抢启动时限；一次并行失败保留，不通过延长旧测试阈值掩盖。

## 显式DeepSeek目录工具流

已有Node依赖/构建及Python推理包时，可信宿主可从本机安全配置注入`DEEPSEEK_API_KEY`环境变量，再显式运行（不要把Key写进命令或聊天）：

```bash
node agent/dist/src/live-catalog.js --live --python "$PWD/.venv/bin/python" --bundle "$BUNDLE_DIR" --case catalog-1
```

入口不自动读凭证文件；缺live/Key/可信参数时0HTTP退出2。只允许catalog-1/2/3固定合成目录任务，不接受真实对话文件。每次最多2实际HTTP、无自动重试；保留固定目录元信息与合成模型回答，错误不输出任意供应商原文。默认为非思考模式，输出仅3字段JSON，经真实health校验后才展示。失败退出2、成功0；正常离线CLI不变。

已执行的原三例首测因首例400停止，不要覆盖其冻结产物或将新运行算作原分数。修复后已用不同输入做单例协议探针；新评测应另冻协议和目录，不能把一次工具流成功当偏好收益。所有普通Node测试使用假fetch，无需Key，也不联网推理。

## 运行正式Pi离线工具循环

需要Node>=22.19.0及已有Python ml/api环境。正式依赖独立锁定，无需克隆Pi或从spike加载包；安装依赖不调用模型，安装脚本关闭：

```bash
npm --prefix agent ci --ignore-scripts --no-audit --no-fund
npm --prefix agent run build
node --test --test-concurrency=1 agent/dist/test/*.test.js
node agent/dist/src/cli.js --python "$PWD/.venv/bin/python" --bundle "$BUNDLE_DIR"
```

`BUNDLE_DIR`指向你已导出的可信推理包（导出流程见下文），不可省略；测试会自行创建临时合成bundle，不训练/下载。成功输出单个JSON：status=ok、modelCalls=2、toolCalls=1及包含实际bundle_id/商品数/历史时点的状态说明；退出0。输入/工具/预算/取消失败仅固定code，退出2。SIGINT/SIGTERM触发取消。

这条CLI是真实Pi Agent→真实Python工具→Pi第二轮，供应商由确定性离线模型代替，无Key/HTTP调用。暂不接任意用户提示、DeepSeek、长期记忆或推荐；两次modelCalls不是两次付费请求。后续真实模型接入需另统计实际HTTP尝试，不借此声称推荐效果。

## 启动Python只读JSONL桥接

使用自己已导出的可信目录（目录导出流程见下文），不需要模型Key。此命令不下载数据或训练：

```bash
uv run --locked --extra ml --extra api python -m memrecagent.agent_bridge --bundle artifacts/product_bundles/<run-id>
```

向stdin发送一行，末尾必须有换行：

```json
{"version":1,"id":"r1","method":"catalog_status","params":{}}
```

stdout立即给出带`r1`的`ok:true/result`，结果含`status/bundle_id/as_of_ms/n_catalog_items/strategies`。相同请求ID与参数只重放首次成功快照；同ID异参返回`id_conflict`，不会调用服务。只读状态不包含商品排名、价格或实时库存。

普通协议错误的形状为`{"version":1,"id":null,"ok":false,"error":{"code":"invalid_json"}}`；无法信任ID时为null，合法ID的业务错误则回带ID。固定码还包括`invalid_request/unknown_method/id_conflict/session_full/service_error/frame_too_large/frame_truncated`。失败不回显输入或后端异常。帧上限64KiB按UTF8字节计（不含LF/CRLF）；超限或无末尾换行退出2，正常EOF退出0。启动目录不存在/损坏时stdout空、stderr只有`bundle_load_failed`、退出2。CLI用法错误由argparse退出2。

父进程须用固定解释器与argv、不要shell拼接；请求后flush，使用有超时的读取，不必关闭stdin等待回执。后续Pi客户端应实施5秒工具超时/总30秒预算、取消与迟到回执拒绝；本轮未交付Pi客户端。成功ID最多256个，满额新ID返回`session_full`，旧回执仍可读；重启清空缓存。失败回执不占成功容量，不自动重试。

离线验证使用测试生成的小型真实格式合成bundle，无需准备数据：

```bash
uv run --locked --extra ml --extra api pytest -q tests/test_agent_bridge.py tests/test_agent_bridge_cli.py
```

需要 Python 3.12、uv 和 Git；先在仓库根目录执行：

```bash
uv sync --locked --extra ml --extra api
```

本指南不会自动执行。以下真实实验会获取数据并使用本地磁盘/CPU；单元与合成 CLI 测试直接运行 `uv run --locked --extra ml --extra api pytest`，无需下载真实数据。

## 1. 核对来源并获取

先阅读[数据使用条件](data.md)。默认只打印固定文件、大小与摘要：

```bash
uv run --locked --extra ml --extra api python scripts/acquire_amazon.py
```

确认本地资源和使用条件后显式获取：

```bash
uv run --locked --extra ml --extra api python scripts/acquire_amazon.py --fetch
```

固定类别两文件约 2.19GB，默认预算 2.5GB。脚本验证完整字节数与 SHA256，保留已有数据，不使用未完成 `.part` 文件作为输入。

## 2. 生成受控子集

指定一个尚不存在的输出目录；不要覆盖旧实验：

```bash
uv run --locked --extra ml --extra api python scripts/prepare_amazon.py --output data/processed/amazon-2023/products-v1
```

输出 `events.jsonl`、`products.json`、`candidates.json`、`report.json` 和 `execution.json`。输入/输出摘要、来源版本与处理资源记录一并保存。

后续命令统一引用这个目录：

```bash
DATA_DIR=data/processed/amazon-2023/products-v1
```

## 3. 运行冻结协议

```bash
uv run --locked --extra ml --extra api python scripts/evaluate_products.py --data "$DATA_DIR"
uv run --locked --extra ml --extra api python scripts/train_products.py --data "$DATA_DIR"
```

热门输出到 `artifacts/products_popularity/<run-id>/`；模型输出到 `artifacts/products_models/<run-id>/`。每次使用新目录。训练使用协议固定默认参数；两种模型验证选择先保存，再从头拟合测试历史。不要根据测试成绩追加搜索并继续当作原实验。

保存配置、数据摘要、源码快照、环境、选择、权重、指标和预测。源码快照不包含数据集或本地环境。当前机器可选使用 RTK；没有 RTK 时使用普通 Git，不影响复现。

## 4. 导出与服务

将 `MODEL_RUN` 替换为上一步终端输出的真实目录：

```bash
MODEL_RUN=artifacts/products_models/<run-id>
uv run --locked --extra ml --extra api python scripts/export_products_bundle.py --data "$DATA_DIR" --models "$MODEL_RUN"
```

导出会核对处理数据、时间协议、源权重摘要与热门统计一致。将 `BUNDLE_DIR` 替换为导出输出：

```bash
BUNDLE_DIR=artifacts/product_bundles/<run-id>
uv run --locked --extra ml --extra api python scripts/check_product_api.py --bundle "$BUNDLE_DIR"
uv run --locked --extra ml --extra api python scripts/serve_products.py --bundle "$BUNDLE_DIR"
```

烟测需允许绑定本地端口。服务使用固定包，无自动选择“最新”版本的隐式行为；请求方式见[API](api.md)。

## MovieLens

```bash
uv run --locked --extra ml --extra api python scripts/download_movielens.py
uv run --locked --extra ml --extra api python scripts/evaluate_movielens.py
uv run --locked --extra ml --extra api python scripts/evaluate_movielens_bpr.py
```

五种子与统计分析入口分别为 `scripts/evaluate_movielens_multiseed.py` 和 `scripts/bootstrap_movielens.py`。后者需要显式提供对应四模型运行目录 `--run`。数据、参数和候选不可混用不同运行版本。

## 校验与构建

```bash
uv run --locked --extra ml --extra api pytest
uv run --locked --extra ml --extra api ruff check .
uv run --locked --extra ml --extra api ruff format --check .
uv build
```

Git 只保存源码、测试、锁文件和项目技术文档；原始/处理数据、用户级日志、模型、环境及本地工作记录均不上传。结果表是历史实验汇总，不代表在其他平台重新验证。

## 自然语言提取预览

不需要商品数据或额外SDK。默认命令只校验仓库内24条合成案例并展示计划：

```bash
uv run --locked --extra ml --extra api python scripts/evaluate_memory_extraction.py
```

在自己的运行进程配置`DEEPSEEK_API_KEY`后，显式加`--live`才调用官方DeepSeek；CLI不自动扫描本机配置或其他凭证文件。不要将Key填入源码、命令参数或Git。真实调用最多24次、无自动重试；401/403/429等供应商错误中止余下，未执行仍保留分母。

```bash
uv run --locked --extra ml --extra api python scripts/evaluate_memory_extraction.py --live
```

每次生成唯一的本地忽略目录，含manifest、输入、逐案例合成响应和report；已有目录不会覆盖。退出0表示严格24例均通过，语义未全通过退出1，缺Key退出2，不能将非零退出隐藏成验收通过。提取器和提案器不写真实用户记忆；调用方须另行审查提案，当前自动应用未启用。

## 本地记忆管理

无需模型API或额外依赖。以下命令只操作自己指定的本机私有目录；不要将原始记忆、输入文件或查看输出加入Git。`list`显式显示证据原文，默认仅当前版本，`--history`显示仍保存的历史。`--as-of`必须带时区，记录的发生/可用时间均严格早于它才可见。

```bash
uv run --locked --extra ml --extra api python scripts/manage_memories.py \
  --root data/private_memory --user-id demo-user \
  list --category phone --entity 小米 --as-of 2026-10-21T00:00:00+08:00
```

纠正现有偏好时，先查询其`memory_id`，把它填入私有JSON文件的`expected_head`；用明确的新证据修订，不让模型猜测用户确认。输入字段如下（合成示例，`expected_head`需替换为实际查到的ID）：

```json
{
  "category": "phone",
  "entity": "小米",
  "expected_head": "待替换为查询结果中的memory_id",
  "relation": "neutral",
  "evidence_ref": "demo-session:T5",
  "evidence_text": "以后不用排除小米了（合成示例）",
  "occurred_at": "2026-10-21T09:00:00+08:00",
  "available_at": "2026-10-21T09:00:01+08:00"
}
```

```bash
uv run --locked --extra ml --extra api python scripts/manage_memories.py \
  --root data/private_memory --user-id demo-user \
  correct --input data/private_inputs/correction.json

uv run --locked --extra ml --extra api python scripts/manage_memories.py \
  --root data/private_memory --user-id demo-user \
  delete --category phone --entity 小米 --at 2026-10-21T10:00:00+08:00
```

纠正时间需晚于当前证据，链头过期则拒绝并要求重新查看。删除时间不能早于已保存证据；删除移除该对象全部历史，即使历史时点查询也不再返回。保留的摘要/时间水位阻止旧证据重放，新表达在删除后仍可重新建立偏好。管理程序不自动清理用户自己提供的输入文件或导出副本，这些文件需由持有人另行处理；也不提供公网鉴权、并发写入或备份擦除。成功退出0，输入/存储错误退出2并输出不含原话的错误码。

## 受控提交接口验证

不需要API Key或真实对话。下面测试在临时目录使用合成候选，覆盖票据、确认、原子提交和删除后重放：

```bash
uv run --locked --extra ml --extra api pytest -q \
  tests/test_memory_commit.py tests/test_memory_commit_store.py
```

接入顺序是`MemoryCommitSession.prepare(candidate, turns=..., as_of=...)`→展示预览；`action=clarify`时先取得实际新用户回复和结构化主体/范围选择，再调用`confirm`生成新票据；只有可应用的新票据才由宿主显式`commit`。`current/noop`不写长期，`cancel(ticket_id)`丢弃当前票据。API位于`src/memrecagent/memory_commit.py`；完整输入示例与实际文件断言在上述测试。

`confirm`不是任意中文识别器；调用方必须明确传入self/other和current/long_term，other还需回复内的人物证据区间。品牌/态度未解、多人物或多目标应重新表达，不能用一声“确认”覆盖全部。准备之后用户数据有变或已删除时必须重新提案，不能重用票据。当前没有开启真实对话的自动写入，不要把模型返回的结构化字段当作可靠的用户确认。

## 合成多轮与时效验证

默认命令只显示3段故事、18基础步和最多20次请求的计划，不读取Key、不创建实验目录：

```bash
uv run --locked --extra ml --extra api python scripts/evaluate_memory_dialogues.py
```

默认`--suite development`使用已看过的旧故事。新合成套件用固定选项选择，仍默认不调用模型：

```bash
uv run --locked --extra ml --extra api python scripts/evaluate_memory_dialogues.py --suite acceptance-v2
```

在本机进程配置`DEEPSEEK_API_KEY`后，显式加`--live`才调用。仅发送冻结合成文本，不接受真实对话文件或真实记忆根；自动写入位于新产物根下的合成stores。

```bash
uv run --locked --extra ml --extra api python scripts/evaluate_memory_dialogues.py --live
```

新套件真实运行则加`--suite acceptance-v2 --live`。每次运行使用新目录，单套最多20次；命令不会自动连跑两套。新集额外比较初始请求和态度语义，`semantic_scored_steps`为18；旧集为0，此时`semantic_correct=0`表示不适用，不能解释为准确率0。

澄清触发对照套件为`--suite acceptance-v3`，同样默认只显示计划，显式加`--live`才调用；包括无操作背景、真实主体/范围歧义、缺品牌及混合未完成表达。最新v2开发/v3首测各一次均18/18，不表示任意真实对话可靠，详见[评测](evaluation.md#2026-10-21澄清触发校准)。

不重试，不覆盖旧目录。18次基础提取加最多2次实际被问后才出现的确认，最多20次。任何供应商错误停止后续，失败和未执行保留总分母。严格18步全通过退出0，否则退出1；缺Key退出2。退出1可以是如实记录的模型协议/语义失败，不能改成通过。

`manifest.json`/`inputs.json`/`source/`冻结数据、提示和代码；`calls.jsonl`保存请求与合成响应、usage/延迟，`steps.jsonl`保存实际分支/写入/当前和活跃状态，`report.json`汇总。参考为助手按规则编写，不称独立人工标注。默认长期偏好无年龄上限；第三段显式采用一天的实验配置，过期保留历史，不能把该阈值当作生产默认或用户偏好变化。

当前真实首轮严格14/18、确认契约1/2，尚不启用真实对话自动写入。旧24例单窗口结果保留，两套分母不混合。假模型测试入口：

```bash
uv run --locked --extra ml --extra api pytest -q \
  tests/test_memory_freshness.py tests/test_memory_confirmation.py tests/test_memory_dialogue.py
```

契约修复后实际各运行一次开发/新集，各17/18、确认2/2、整段2/3，仍有多余澄清。当前全量571项程序测试通过；新增语义评分与实际CLI选择回归在`tests/test_memory_contract_stability.py`。模型效果及评分差异见[评测说明](evaluation.md#2026-10-21契约修复与新合成首测)，不能将开发集回归称新盲测。

最新澄清触发校准的全量程序回归为575项，保护测试在`tests/test_memory_clarification_gate.py`。它们验证无完整断言但存在未解实体/态度时仍不能提交；仅修提取提示，宿主不会静默丢弃模型报告的不确定项。


### 验证会话偏好接续

```bash
uv run --locked --extra ml --extra api pytest -q tests/test_memory_session_context.py
```

24项测试使用临时存储和本地合成客户端，不读Key、不调用模型、不下载数据。包括三组固定18步运行器变体，各20次假响应，验证接续/切会话/重启；原冻结JSON不改。这些检查只验证程序接续与隔离，不代表自然语言模型效果。

编程调用时复用绑定用户/会话的`MemoryCommitSession`，先prepare核验，再显式commit接受。只prepare的预览不会跨轮生效；`not_applied`意为无长期写入，可能接受他人会话偏好。查看实际激活内容用`preview.proposal.decision.context_preferences`，`proposal.session_preferences`仅本轮新提取。新建宿主即清空他人会话记忆。


### 原24例当前版本回归的口径

`evaluate_memory_extraction.py`使用六组各四条固定合成案例，默认仍只展示计划；显式`--live`才发送当前提示与合成表达，不发送期望答案。2026-10-21当前版本单次实测18/24，因此该次CLI退出1正确；不要修改标签或隐藏退出码将其当通过。

同响应宿主回放是单独的离线诊断，不再请求模型，也不把其结果反写原strict成绩。错误提案、真正提交的临时合成记录、人物目标和澄清分别核对；当前7条新增/0错误长期内容不能解释为24条请求都正确。六个失败及预算边界见[评测](evaluation.md#原24例当前版本开发回归)。


### 检查版本2交互候选协议

```bash
uv run --locked --extra ml --extra api pytest -q tests/test_memory_interaction_extraction.py
```

相关合成结构化测试无需Key或网络，验证意图/话题/弱线索与长期断言分离、持久证据的原文唯一定位、角色/最新表达/确认引用和畸形输出拒绝。非持久topic只要求用户原文存在、允许重复去重；其他quote仍转为唯一Span后交原严格解码。新增恢复回归可运行`uv run --offline --locked --extra ml --extra api pytest -q tests/test_interaction_recovery.py`，不能把旧响应离线解码当新模型实验。

当前仅协议解析及消息构造，普通双向对话宿主和新模型验收尚未接入；已有`evaluate_memory_extraction.py`仍用原版本契约。工程检查通过不代表DeepSeek已正确生成新版本输出，也不代表渐进学习或商品排名完成。

## 普通双方对话的记忆Hook

`ConversationMemorySession`接收实际发生的用户和助手轮次。它返回回答动作和可核验记忆背景，不生成手机事实、商品清单或排名。询问品牌不等于喜欢；弱线索保留在工作区，明确本人长期偏好仍须宿主显式提交。他人稳定偏好只在当前实例中复用。

下面展示应用接入顺序；`complete`为分析模型，`answer`为应用实际回答函数，须返回带真实时间的assistant `Turn`。两者职责分开，记忆分析响应本身不能冒充正常助手回答。

```python
from collections.abc import Callable
from datetime import datetime

from memrecagent.deepseek_client import ModelReply
from memrecagent.memory_extraction import Turn
from memrecagent.memory_interaction import ConversationMemorySession, InteractionPreview


def handle_user(
    session: ConversationMemorySession,
    user_turn: Turn,
    *,
    as_of: datetime,
    complete: Callable[[list[dict[str, str]]], ModelReply],
    answer: Callable[[InteractionPreview], Turn],
) -> Turn:
    preview = session.prepare_user(user_turn, complete=complete, as_of=as_of)
    if preview.response_action != "clarify":
        # 明确接受整轮后才发布；只prepare不会保存偏好或工作线索。
        session.commit(preview.ticket_id, as_of=as_of)

    assistant_turn = answer(preview)
    if preview.response_action == "clarify":
        # 短答必须绑定宿主实际提出的这个问题，不能改成多个问题。
        question = preview.memory_preview.proposal.decision.clarification
        if assistant_turn.content != question:
            raise ValueError("需要输出待澄清问题原文")
    session.observe_assistant(assistant_turn)
    return assistant_turn
```

应用可使用`preview.context_memories`作为相关本人背景，推荐时还可读取`preview.memory_preview.proposal.decision`中实际激活的会话偏好和当前约束。明确给亲属推荐时不能套用本人背景。这里尚未把记忆特征接入商品排序，也不应根据示例编造手机价格、库存或性能。

模型输入由`build_interaction_messages`生成`output_contract`，不是让应用或模型填写用户喜好标签：最新轮次和允许分支来自实际上下文，确认空candidate只是协议模板。确认回复不应重复原断言；新完整表达或换题也不应强行确认旧待办。无论提示如何引导，混合输出、旧证据冒充新表达仍会被严格解码拒绝，不会自动删字段重试。

外部纠正或删除会使现有工作区过期；调用方收到`stale`后使用新实际前缀重建，不自动重发旧原话。`deleted_evidence`拒绝删除前来源重提，`binding`表示存储归属变化，不能当普通模型错误自动重试。

新渐进交互验收默认仅校验合成输入并显示计划，无Key、无输出目录、0模型调用：

```bash
rtk proxy .venv/bin/python scripts/evaluate_memory_interactions.py
```

显式`--live`才读取本机环境变量`DEEPSEEK_API_KEY`。冻结3段×6个用户基础轮次，加最多2条实际澄清后的条件回复；每次请求无重试，总上限20次。普通助手轮次标记为`scripted_assistant_context`，不能声称由模型自主生成完整对话；运行只写独立合成存储，不触及真实用户记忆。历史模型套件和新套件必须分别报告。


## 只读 MemoryAnalyst 离线验证

```bash
uv run --locked --extra ml --extra api pytest tests/test_memory_analyst.py -q
```

测试注入固定ModelReply，验证相关记忆筛选、用户/会话/时间边界、实际证据与问题绑定、单次失败不回退及调用账本。无需Key、网络或数据下载；不会把固定响应分数当作DeepSeek准确率。会话宿主暂未使用此模块，不提供新的live或自动保存开关。


双阶段宿主离线验证：`uv run --locked --extra ml --extra api pytest tests/test_memory_analyst_host.py -q`。运行接口显式选择`analysis_mode="analyst"`，由相同complete回调完成两阶段；默认direct保持一次。此模式不是自动保存开关，仍须prepare/confirm后可信调用方显式commit。


### 查看新记忆角色验收计划

```bash
uv run --locked --extra ml --extra api python scripts/evaluate_memory_interactions.py --suite analyst-v1 --analysis-mode analyst
```

默认只验证合成输入、显示40次上限，API0且不读Key、不建结果目录；direct模式上限20。显式live才发送，实际请求按direct/draft/analyst逐次记录，确认物理请求与用户确认轮分开。旧默认套件与20上限保留。新一轮迭代应首调用前冻结新输入/参考/版本与预算；保留历史失败，不能用同集重复追分替代首测。

## 最新记忆生命周期的离线参考检查

新合成输入在tests/fixtures/memory_request_lifecycle_v1.json。运行 `uv run --locked --extra ml --extra api pytest -q tests/test_memory_request_lifecycle.py` 可核查direct/analyst固定客户端控制、真实状态与参考隔离；这不会调用模型，也不能重现模型成绩。公开CLI的--suite仍仅支持progressive-v1和analyst-v1，没有冒称新套件可由该选项直接执行。

60次新模型验收采用本地冻结双模式入口，逐发送验证源码/输入摘要及最多20/40调用，原日志保留不重复执行追分。程序API load_interactions(path) 与 evaluate_interactions(cases, complete=..., output_dir=..., analysis_mode=...) 支持显式新数据；新的付费实验须另冻输入与预算，并单独保存结果，不能覆盖历史产物。详见[新结果与局限](evaluation.md#商品选项与维持旧偏好-request-lifecycle-v1)。
# 离线四组排名重放

入口默认仅显示计划，显式`--run-offline`执行，无模型API。需要同一冻结bundle、完整可信实体映射、隔离合成记忆和版本1套件。套件每例将request与relevant/excluded标签分开；标签不传给排名器。输出目录必须不存在。

```bash
uv run --locked --extra ml --extra api python scripts/evaluate_ranking_ablation.py \
  --bundle PATH_TO_BUNDLE --facts PATH_TO_FACTS_JSON \
  --memory-root PATH_TO_SYNTHETIC_MEMORY --suite PATH_TO_SUITE_JSON \
  --output-dir NEW_OUTPUT_DIRECTORY --run-offline
```

`--normalisation rank`显式比较同分rank百分位；默认仍minmax。保存manifest.json、源码、suite/facts、cases.jsonl和report.json。运行失败仍计案例；退出0表示执行无错误且输入未漂移，不表示有效果收益。当前只支持structured_oracle机制重放，自动语言记忆实验尚未由此入口完成。

套件schema包含`schema_version:1`、`evidence_kind:"structured_oracle"`、`label_source`、`cases`。每例包含`case_id`、`cluster_id`、`stratum`、`request`、`relevant`、`excluded`；request要求user_id/session_id/beneficiary_id/category/as_of/strategy/k，所有案例k相同，strategy必须mf/bpr/bpr_cold_popularity。实体facts沿用正式记忆排名工具的bundle_id+facts结构。不要把真实用户标识或证据复制为公开样例。

# 统一写入质量评测

192步作者合成语料，开发/预留测试按故事和模板族划分；非独立人工盲测。默认显示计划、不读Key、不写输出。

```bash
uv run --locked --extra ml --extra api python scripts/evaluate_memory_quality.py \
  --split dev --analysis-mode direct --output-dir artifacts/quality/NEW_RUN
```

真实调用额外需要显式`--live --max-calls 96`和当前进程环境中的`DEEPSEEK_API_KEY`。不得把密钥写进命令、语料或聊天。analyst为草案+最终角色，上述dev全量上限192；Key或显式预算缺失时零调用。provider失败停止整次运行，处理失败留账后继续，未执行轮仍计分母。源码每次调用前后冻结检查，原输出目录不可覆盖。

存储在每个故事内从空开始，跨session实际重建；不会按参考答案补回失败写入。calls/steps/report含原始合成对话与模型输出，仅存本地忽略目录。精确率、召回率、错误归属、修订、澄清、状态/激活、完整故事、用量缺失和失败分别报告，不能把固定客户端测试当模型准确率。

## 外部评测数据只读验收

将故事和任务JSONL保留在本机`data/evaluation/incoming/`，使用新产物目录执行：

```bash
rtk proxy env UV_CACHE_DIR=.uv-cache uv run --offline --locked --extra ml --extra api python scripts/audit_external_evaluation.py --stories data/evaluation/incoming/stories.jsonl --tasks data/evaluation/incoming/recommendation_tasks.jsonl --output-dir artifacts/external_evaluation_audit/new-run
```

不需要Key、不调用模型、不执行生成源码；原始字节与摘要封存到指定目录。退出码0表示无已发现问题，1表示存在问题或待复核，2表示输入/目录错误。检查通过不证明语义标注正确；当前收到的60故事尚未完成语义复核与正式划分，不能直接当作独立测试效果。


### 可信负反馈排名检查

```bash
rtk proxy env UV_CACHE_DIR=.uv-cache uv run --offline --locked --extra ml --extra api pytest -q tests/test_feedback_ranking.py
```

该检查使用固定商品/模型夹具与真实JSON反馈存储，无供应商HTTP。
持久反馈位于相同记忆根目录`feedback/`，写入仍须可信HostFeedbackSession的实际展示目标。
当前聊天CLI已有下面的显式反馈命令。测试中的固定供应商回调仍不是模型识别质量或真实用户曝光。


### 聊天中的显式屏蔽与撤销

使用上文显式live聊天启动方法。看到普通推荐结果后，可在同一进程输入：

```text
/block 2 product request
/block 1 brand persistent
/undo 由上一条反馈返回的32位回执ID
```

命令须完整给出位置、商品/品牌粒度、长期/本次范围；每次按最新实际展示的列表编号。
`persistent`只允许本人且用途/更新许可开启。给他人选购仅使用`request`。
本次临时屏蔽在下一条普通用户输入时结束；持久屏蔽跨重启读取。
反馈会立即用相同预算和算法重排并显示，不新增模型调用；收到新列表后方能再次按位置屏蔽。
撤销目前限当前会话已知回执，临时范围结束后不可撤销已过期临时操作。

输出为JSONL，普通回答type=reply，反馈type=feedback；反馈status与feedbackCommitStatus须分开看。
未知写入或显示登记失败会停止，不自动重放。同请求相同RPC ID去重只限当前Python进程。

新检查：

```bash
rtk proxy env UV_CACHE_DIR=.uv-cache uv run --offline --locked --extra ml --extra api pytest -q tests/test_conversation_feedback.py
```

本地封存实际CLI离线探针在`artifacts/conversation_feedback/20261024-host/cli-probe`，
真实Pi/SDK与Python桥接，替换供应商transport和分析响应；ABC→BC→ABC，实际HTTP0。
封存目录拒绝覆盖，不要直接重复运行原run.mjs。


### 统一查看与精确删除（可信本机管理）

下面root/user替换为自己的本地值，时间使用带时区ISO格式。inspect会显式展示原文，只在私有终端执行。

```bash
rtk proxy env UV_CACHE_DIR=.uv-cache uv run --offline --locked --extra ml --extra api python scripts/manage_memories.py --root /absolute/private-memory --user-id demo inspect --as-of 2026-10-24T12:00:00+08:00
rtk proxy env UV_CACHE_DIR=.uv-cache uv run --offline --locked --extra ml --extra api python scripts/manage_memories.py --root /absolute/private-memory --user-id demo forget --kind brand --category phone --target 品牌A --at 2026-10-24T12:01:00+08:00
rtk proxy env UV_CACHE_DIR=.uv-cache uv run --offline --locked --extra ml --extra api python scripts/manage_memories.py --root /absolute/private-memory --user-id demo forget-feedback --kind brand --category phone --target 品牌B --at 2026-10-24T12:02:00+08:00
rtk proxy env UV_CACHE_DIR=.uv-cache uv run --offline --locked --extra ml --extra api python scripts/manage_memories.py --root /absolute/private-memory --user-id demo recover
```

forget brand删除同品牌偏好与品牌反馈；forget product只删精确商品反馈；forget attribute删除如price_cny的全部属性历史。
forget-feedback仅删反馈，适合重启后移除屏蔽且保留偏好；它是删除反馈历史，不是恢复已删除内容。
旧delete命令仍只删除品牌偏好，未自动扩大其范围。
删除中断的management_pending会阻止该用户记忆/反馈读取与推荐，执行recover完成已验证意图。
损坏或非预期文件漂移不会自动修复；保留目录并检查错误，不手工移除标记来绕过保护。
本机单进程串行，不要同时启动两个写入者。管理操作不调用模型，当前聊天旧回执会在下一fresh失效。


### 已有真实包的活跃画像机制审计

```bash
rtk proxy env UV_CACHE_DIR=.uv-cache OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run --offline --locked --extra ml --extra api python scripts/audit_warm_profiles.py --bundle artifacts/product_bundles/20261020T051100515587Z-e27333ad --output-dir artifacts/warm_profile_audit/new-run --as-of 2026-10-24T12:00:00+08:00 --category musical_instruments --active-count 16 --cold-count 4
```

默认只展示计划，不读包/Key或创建产物；加`--run-offline`才读取既有包，输出必须新目录。
类别由可信调用方根据已核来源指定；本例是乐器，不能换标手机。
输出private-plan含本机原生owner ID，仅留私有忽略目录，不发送给供应商；case trace与统计用摘要标识。
目标按哈希从未见候选选，固定prefer/exclude是假想干预，不是用户话语或真实购买标签。
无模型调用、无训练、无下载；新run是机制重放，不能用来重复旧模型集追分或作为新的盲测。


### 软属性排序与独立关闭

可信ranker/recommend请求或memory_product_tool请求JSON可加`use_soft_attributes:false`，
用于保持品牌/硬约束/反馈不变而只关闭软属性评分。省略时默认开启。
`soft_preferences`与`constraints`使用同样完整的绑定谓词JSON字段，分别走评分、过滤；禁止以软项替代硬预算。
离线ranking_ablation请求也接两字段，四组共用current软项，memory-off仅关长期属性。

```bash
rtk proxy env UV_CACHE_DIR=.uv-cache uv run --offline --locked --extra ml --extra api pytest -q tests/test_soft_attribute_ranking.py
```

实际语言schema4的is_hard=false支持scope=long_term或request。仅本次要求使用request，不要伪装为长期。
request active=false/value=null仅暂时忽略该属性软偏好；硬预算不变。后续推荐沿用任务，明确start/end切换。
手工工具/离线请求可传soft_relaxations（与relaxations相同绑定字段，但只影响软属性），不能与同属性soft_preferences混用。
新离线语言宿主闭环见`artifacts/request_soft_attributes/20261025-integrated`，其中固定回调不是模型质量证据。
冻结机制探针见`artifacts/soft_attributes/20261024-ranking`，不含真实供应商HTTP，不覆盖已封存实验。


### 配置可信实体别名

操作者先准备JSON，来源与别名须依据实际目录人工核实；以下只是合成示例，不是产品内置品牌知识：

```json
{"schema_version":1,"entries":[{"category":"phone","canonical":"品牌A","aliases":["BrandA","A牌"],"source":"synthetic:catalog:v1"}]}
```

```bash
rtk proxy env UV_CACHE_DIR=.uv-cache uv run --offline --locked --extra ml --extra api python scripts/configure_entity_aliases.py --root /absolute/new-private-memory --input /absolute/alias-input.json
rtk proxy env UV_CACHE_DIR=.uv-cache uv run --offline --locked --extra ml --extra api python scripts/configure_entity_aliases.py --root /absolute/new-private-memory --input /absolute/alias-input.json --initialize
```

默认仅验证表并展示摘要，不创建root；--initialize才检查并原子写入新的空namespace，文件0600。
必须在启动聊天/存储实例之前配置，后续chat-cli/memory_product_tool的memory-root使用同一路径。
已有JSONL、反馈或管理意图的root不允许直接安装；不要手工放表或覆盖表来绕过。已有数据继续使用原namespace；
旧库迁移与表扩展需另行设计，不自动丢弃记录/删除水位或迁移用途设置。
只有操作者离线命令可配置，模型没有此工具；source仅是来源声明，不是自动事实认证。

启用后原文entity仍按原写法提供、引用范围不变，由宿主生成规范键；不同别名共享修订、反馈与删除。
表变化会使实例/旧文件拒绝读取，恢复管理事务也要求同一份表；不会静默热更新。
旧版程序不支持启用后的JSONL元数据头/反馈schema2，无表旧格式保持兼容。


仅本次软偏好检查（无Key、无HTTP）：

```bash
rtk proxy env UV_CACHE_DIR=.uv-cache uv run --offline --locked --extra ml --extra api pytest -q tests/test_request_soft_attributes.py
```


## 按请求数量选择评分K（下一版实验）

`evaluate_external_replay.py --scoring-mode request`只从原请求明确数量决定各任务K；无数量用`--k`fallback，
歧义数量在模型调用前拒绝整个运行。manifest/cases保存K、来源和原文位置，四组共享，错误空列表不缩分母。
默认`fixed`保持历史固定K，按用户数量评估须显式选`request`；不可把新旧协议混报提升或重跑旧test改分。

```bash
rtk proxy env UV_CACHE_DIR=.uv-cache uv run --offline --locked --extra ml --extra api python scripts/evaluate_external_replay.py --suite artifacts/evaluation_cutoff/20261025-request-v1/new-synthetic-suite.json --output-dir artifacts/evaluation_cutoff/new-run --scoring-mode request
```

以上是新合成fixture的plan，无Key/零调用；新live仍须另有输入/有限额度授权。详见本机REQUEST-SCORING-CUTOFF-REPORT.md。

## 现有乐器目录与可信品类

在项目根目录执行 `uv run --offline --locked --extra ml --extra api python scripts/create_instrument_facts.py --bundle <现有包路径> --output <新映射文件>`，输出禁止覆盖。
Pi聊天/ConversationClient/持续Python桥接显式传 `--category musical_instruments`，其余必填参数及live/Key要求不变；映射完整包含5000原ID，乐器4898件，未知根类102件不参加乐器查询。
新命名空间起步；entity为精确商品ID，不猜品牌，没有自动标题模糊匹配。当前非手机属性协议明确拒绝，不能据此承诺支持预算。
本轮固定响应6画像24条件验证的是实际BPR与跨会话记忆接线；自然语言准确率和独立效用待验证，详见INSTRUMENT-DOMAIN-INTEGRATION-REPORT.md。

乐器完整标题：`python scripts/create_instrument_aliases.py --bundle <既有包> --output-dir <不存在的新目录>` 导出entity-aliases.json与覆盖报告；使用既有configure_entity_aliases.py先plan、后--initialize安装新空namespace。精确NFKC/casefold/strip整标题，不截短或猜重名，真实目录4871登记/24重名/3过长/102未知。详见INSTRUMENT-TITLE-ALIASES-REPORT.md。

## 原生乐器环境的语言回放（待新有限授权）

复用evaluate_external_replay.py，新增可选 `--native-environment <environment.json>`。默认仍只plan，配置含本机bundle/facts/aliases/user_bindings/base_at，不能把真实ID放进模型消息。用 `--scoring-mode request` 事前按明确三款定K；没有独立效用参考时指标为N/A，不编造相关性。
本轮6段42步同域包已获新授权并使用direct42+analyst84共126次；原剩余10次未复用。可信串行启动器默认plan，协议见INSTRUMENT-LANGUAGE-EVALUATION-PROTOCOL.md。

同域真实调用已结束，原live输出目录及controller-status存在；启动器会拒绝重复执行。结果见INSTRUMENT-LANGUAGE-EVALUATION-REPORT.md，复核使用冻结summarize.py读取原calls/steps/cases，不重新发送故事。
