"""流式筛选用户，生成受控商品数据；候选选择仅使用训练历史。"""

import json
from collections import Counter
from collections.abc import Callable, Iterator, Mapping
from dataclasses import asdict
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path

from memrecagent.amazon import history_before, parse_product, parse_review

TRAIN_END_MS = 1640995200000
VALIDATION_END_MS = 1656633600000
TEST_END_MS = 1672531200000


def selected_user(user_id: str, *, modulus: int = 16, bucket: int = 0) -> bool:
    """按完整SHA256整数取模，与活跃度、评分和Python进程随机哈希无关。"""
    if (
        type(modulus) is not int
        or modulus <= 0
        or type(bucket) is not int
        or not 0 <= bucket < modulus
    ):
        raise ValueError("用户桶需要正模数与合法桶编号")
    return int.from_bytes(sha256(user_id.encode("utf-8")).digest(), "big") % modulus == bucket


def _objects(path: Path, digest_update: Callable[[bytes], object]) -> Iterator[tuple[int, dict]]:
    with path.open("rb") as stream:
        for line_number, line in enumerate(stream, start=1):
            digest_update(line)
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError
            except (ValueError, UnicodeError):
                raise ValueError(f"{path.name}:{line_number}: 非法JSON对象") from None
            yield line_number, row


def prepare_amazon(
    review_path: Path,
    meta_path: Path,
    output: Path,
    *,
    modulus: int = 16,
    bucket: int = 0,
    max_events: int = 200000,
    max_users: int = 150000,
    catalog_limit: int = 5000,
    expected_hashes: Mapping[str, str] | None = None,
) -> dict:
    """严格扫描原始评价；未来数据不选择候选，预算不足则停止而不截断。"""
    if output.exists():
        raise FileExistsError("处理目录已存在，保留已有结果")
    selected_user("配置校验", modulus=modulus, bucket=bucket)
    if min(max_events, max_users, catalog_limit) <= 0:
        raise ValueError("处理预算必须为正")
    events, users = [], set()
    review_digest, meta_digest = sha256(), sha256()
    n_source_reviews = 0
    time_min, time_max = None, None
    for line_number, row in _objects(review_path, review_digest.update):
        event = parse_review(row, source=review_path.name, line_number=line_number)
        n_source_reviews += 1
        time_min = event.timestamp_ms if time_min is None else min(time_min, event.timestamp_ms)
        time_max = event.timestamp_ms if time_max is None else max(time_max, event.timestamp_ms)
        if event.timestamp_ms >= TEST_END_MS or not selected_user(
            event.user_id, modulus=modulus, bucket=bucket
        ):
            continue
        events.append(event)
        users.add(event.user_id)
        if len(events) > max_events or len(users) > max_users:
            raise ValueError("用户或交互超过预定预算，未截断或生成处理结果")
    if not events:
        raise ValueError("受控子集没有事件，未生成处理结果")
    events.sort(key=lambda r: (r.timestamp_ms, r.user_id, r.item_id, r.variant_id))
    history = history_before(events, TRAIN_END_MS)
    if not history:
        raise ValueError("训练时间窗口为空，不能选择候选")
    # 历史已经每个用户/父商品唯一，故计数即独立评价用户数，不看未来热度。
    counts = Counter(r.item_id for r in history)
    candidates = sorted(counts, key=lambda item: (-counts[item], item))[:catalog_limit]
    candidates = sorted(candidates)
    wanted = {r.item_id for r in events}
    products = {}
    n_source_products = 0
    for line_number, row in _objects(meta_path, meta_digest.update):
        n_source_products += 1
        if row.get("parent_asin") not in wanted:
            continue
        product = parse_product(row, source=meta_path.name, line_number=line_number)
        if product.item_id in products:
            raise ValueError(f"{meta_path.name}:{line_number}: 展示目录父商品重复")
        products[product.item_id] = product
    actual_hashes = {
        review_path.name: review_digest.hexdigest(),
        meta_path.name: meta_digest.hexdigest(),
    }
    if expected_hashes is not None and dict(expected_hashes) != actual_hashes:
        raise ValueError("原始文件SHA256与固定版本不符，未发布处理结果")
    window_counts = {"train": 0, "validation": 0, "test": 0}
    for event in events:
        window = (
            "train"
            if event.timestamp_ms < TRAIN_END_MS
            else "validation"
            if event.timestamp_ms < VALIDATION_END_MS
            else "test"
        )
        window_counts[window] += 1
    missing = wanted - products.keys()
    report = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "user_selection": {"hash": "SHA256 full integer", "modulus": modulus, "bucket": bucket},
        "cutoffs_ms": {
            "train_end": TRAIN_END_MS,
            "validation_end": VALIDATION_END_MS,
            "test_end": TEST_END_MS,
        },
        "input_sha256": actual_hashes,
        "n_source_reviews": n_source_reviews,
        "source_time_min_ms": time_min,
        "source_time_max_ms": time_max,
        "n_selected_events": len(events),
        "n_selected_users": len(users),
        "n_selected_items": len(wanted),
        "selected_window_counts": window_counts,
        "n_train_effective_pairs": len(history),
        "n_train_repeated_events": window_counts["train"] - len(history),
        "n_source_products": n_source_products,
        "n_selected_products_with_metadata": len(products),
        "n_selected_items_without_metadata": len(missing),
        "n_candidate_items": len(candidates),
        "n_candidate_items_without_metadata": len(set(candidates) - products.keys()),
        "candidate_rule": "training distinct reviewers descending, item_id ascending; fixed cap",
        "candidate_limit": catalog_limit,
        "limits": {"max_events": max_events, "max_users": max_users},
        "scope": "评价字段严格扫描；元数据仅解析引用商品；快照字段不用于历史候选/评分",
    }
    # 预算、解析和目录检查全部完成后才建立输出目录；不覆盖任何原结果。
    output.mkdir(parents=True, exist_ok=False)
    with (output / "events.jsonl").open("x", encoding="utf-8") as stream:
        for event in events:
            stream.write(json.dumps(asdict(event), ensure_ascii=False) + "\n")
    for filename, value in [
        ("candidates.json", candidates),
        ("products.json", {item: asdict(products[item]) for item in sorted(products)}),
        ("report.json", report),
    ]:
        with (output / filename).open("x", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)
    return report
