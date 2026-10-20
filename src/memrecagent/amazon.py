"""解析原始商品数据，并构建不读取未来事件的父商品历史视图。"""

from collections.abc import Iterable, Mapping
from copy import deepcopy
from dataclasses import dataclass
from math import isfinite

MIN_TIMESTAMP_MS = 631152000000  # 1990-01-01 UTC，用于拒绝秒/纳秒误传。
MAX_TIMESTAMP_MS = 4102444800000  # 2100-01-01 UTC，不表示实际数据覆盖范围。


@dataclass(frozen=True)
class AmazonReview:
    """一次评价事件；时间暂作可用时间代理，不等同于购买时间。"""

    user_id: str
    item_id: str
    variant_id: str
    rating: float
    timestamp_ms: int
    source: str
    line_number: int


@dataclass(frozen=True)
class ProductSnapshot:
    """展示快照；其属性不能自动当成历史推荐时刻已知的特征。"""

    item_id: str
    title: str | None
    main_category: str | None
    store: str | None
    categories: tuple[str, ...]
    details: dict[str, object]
    source: str
    line_number: int


def _location(source: str, line_number: int) -> str:
    if not isinstance(source, str) or not source.strip():
        raise ValueError("来源名称必须是非空字符串")
    if type(line_number) is not int or line_number <= 0:
        raise ValueError("来源行号必须是正整数")
    return f"{source}:{line_number}"


def _string(row: Mapping, field: str, location: str, *, required: bool = False) -> str | None:
    value = row.get(field)
    if value is None or isinstance(value, str) and not value.strip():
        if required:
            raise ValueError(f"{location}: 缺失必需字符串字段 {field}")
        return None
    if not isinstance(value, str):
        raise ValueError(f"{location}: 字符串字段类型错误 {field}")
    return value


def _timestamp(value: object, location: str) -> int:
    if type(value) is not int or not MIN_TIMESTAMP_MS <= value < MAX_TIMESTAMP_MS:
        raise ValueError(f"{location}: 时间必须是合理范围内的整数毫秒，禁止猜测单位")
    return value


def parse_review(row: Mapping, *, source: str, line_number: int) -> AmazonReview:
    """从固定原始JSONL模式读取必需字段；错误不包含用户或评论原文。"""
    location = _location(source, line_number)
    if not isinstance(row, Mapping):
        raise ValueError(f"{location}: 评价必须是JSON对象")
    user = _string(row, "user_id", location, required=True)
    parent = _string(row, "parent_asin", location, required=True)
    variant = _string(row, "asin", location, required=True)
    rating = row.get("rating")
    if type(rating) not in (int, float) or not isfinite(rating) or not 1 <= rating <= 5:
        raise ValueError(f"{location}: rating必须是有限的1至5数值")
    timestamp = _timestamp(row.get("timestamp"), location)
    return AmazonReview(user, parent, variant, float(rating), timestamp, source, line_number)


def parse_product(row: Mapping, *, source: str, line_number: int) -> ProductSnapshot:
    """只提取展示需要的原始字段，不推断品牌或补齐缺失商品事实。"""
    location = _location(source, line_number)
    if not isinstance(row, Mapping):
        raise ValueError(f"{location}: 商品必须是JSON对象")
    item = _string(row, "parent_asin", location, required=True)
    categories = row.get("categories")
    if categories is None:
        categories = []
    if not isinstance(categories, list) or any(not isinstance(c, str) for c in categories):
        raise ValueError(f"{location}: categories必须是字符串列表")
    details = row.get("details")
    if details is None:
        details = {}
    if not isinstance(details, dict):
        raise ValueError(f"{location}: 原始details必须是字典")
    return ProductSnapshot(
        item,
        _string(row, "title", location),
        _string(row, "main_category", location),
        _string(row, "store", location),
        tuple(categories),
        deepcopy(details),
        source,
        line_number,
    )


def history_before(events: Iterable[AmazonReview], cutoff_ms: int) -> tuple[AmazonReview, ...]:
    """先取时间前缀，再对父商品选择最新评分；未来冲突不影响过去。"""
    _timestamp(cutoff_ms, "cutoff_ms")
    latest: dict[tuple[str, str], AmazonReview] = {}
    values_at_time: dict[tuple[str, str, int], float] = {}
    for event in events:
        if event.timestamp_ms >= cutoff_ms:
            continue
        timed_pair = (event.user_id, event.item_id, event.timestamp_ms)
        previous_rating = values_at_time.get(timed_pair)
        if previous_rating is not None and previous_rating != event.rating:
            raise ValueError(f"{event.source}:{event.line_number}: 同时刻父商品评分冲突")
        values_at_time[timed_pair] = event.rating
        pair = (event.user_id, event.item_id)
        previous = latest.get(pair)
        if previous is None or event.timestamp_ms > previous.timestamp_ms:
            latest[pair] = event
        elif event.timestamp_ms == previous.timestamp_ms:
            # 相同评分只保留一个稳定来源，不能让输入顺序改变结果。
            latest[pair] = min(
                (previous, event), key=lambda r: (r.variant_id, r.source, r.line_number)
            )
    return tuple(latest[pair] for pair in sorted(latest))
