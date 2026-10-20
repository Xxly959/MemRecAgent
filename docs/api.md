# 本地推荐 API

## 启动

先按[复现指南](reproduction.md)导出冻结推理包，再执行：

```bash
uv run --locked --extra ml --extra api python scripts/serve_products.py --bundle artifacts/product_bundles/<run-id>
```

默认 `127.0.0.1:8000`，`--port` 可修改端口。进程启动先验证并加载包，失败不会提供就绪服务。接口文档位于 `/docs`。

## GET /health

返回 `status`、`bundle_id`、`as_of_ms`、`n_catalog_items` 和可用 `strategies`，不返回原始用户列表。`as_of_ms` 是历史拟合截止，不是元数据生效时间。

## POST /recommendations

| 字段 | 必需 | 约束 |
| --- | --- | --- |
| `user_id` | 是 | 去空白后 1–128 字符 |
| `bundle_id` | 是 | 从当前 `/health` 获取 |
| `strategy` | 是 | `popularity`、`mf`、`bpr`、`bpr_cold_popularity` |
| `k` | 否 | 默认 10，严格整数 1–50 |

拒绝额外字段、未来标签和任意 `as_of_ms`。版本不符返回 409，参数不符返回 422，不静默切换版本或策略。

```json
{
  "user_id": "demo-user",
  "bundle_id": "替换为 /health 返回的 bundle_id",
  "strategy": "bpr_cold_popularity",
  "k": 10
}
```

上述为请求格式示意，不是实际运行结果。

## 响应语义

- `requested_strategy` / `used_strategy`：请求与实际使用策略。
- `items`：父商品 `item_id`、标题、类别、真实原始 `score`、`score_kind` 和元数据来源（revision、URL、文件与行号）。
- `metadata_snapshot`：快照版本、未知的 `available_at` 和 `display_only` 用途。
- `trace`：是否无历史、模型是否认识用户、是否有效 BPR 用户、候选量、已见过滤量、请求/实际数量及兜底原因。
- `empty_reason`：没有未见候选时为 `no_unseen_candidates`；候选不足返回短列表。

热门分数是正向独立评价人数，模型分数是向量点积，均不是购买概率。缺元数据保留未知，不剔除候选或补造事实。没有实时价格、库存或 LLM 生成解释。

`bpr_cold_popularity` 仅在用户完全没有原始历史时采用热门；候选内无向量、无正向、负分或空列表不触发。纯 MF/BPR 的未知用户统一零分并按 ID 同分排序。

## 冻结包与边界

推理包包含 `bundle.json`、`service.json`、`mf.pt`、`bpr.pt`。保存候选、历史已见、热门计数、最小展示快照与模型；不包含未来答案或评论正文。每文件 SHA256 和清单身份校验后才加载，模型使用张量与基本类型恢复。

请求路径不访问原始数据、不调用网络、不训练或更新参数。包用于本地可信环境的可复现推理；不提供公网认证、多租户隔离或生产高可用配置。不要从不可信来源加载模型包。

真实 TCP 烟测：

```bash
uv run --locked --extra ml --extra api python scripts/check_product_api.py --bundle artifacts/product_bundles/<run-id>
```

脚本临时启动自己的本地服务，比较 HTTP 与 Python 响应、409/422，结束后关闭子进程；它不是延迟或吞吐基准。
