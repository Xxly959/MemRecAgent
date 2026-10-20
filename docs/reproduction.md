# 复现指南

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
