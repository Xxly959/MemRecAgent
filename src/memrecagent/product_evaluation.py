"""商品发现协议：历史、无标签预测与未来答案分别处理。"""

from collections.abc import Iterable, Mapping, Sequence, Set
from dataclasses import asdict, dataclass

from memrecagent.amazon import AmazonReview, history_before
from memrecagent.metrics import evaluate_users
from memrecagent.popularity import count_positive_ratings, recommend_popular


@dataclass(frozen=True)
class ProductWindow:
    """离线答案对象；线上预测函数不接收该对象或其中的未来标签。"""

    history: tuple[AmazonReview, ...]
    users: tuple[str, ...]
    seen: dict[str, set[str]]
    relevant: dict[str, set[str]]
    stats: dict[str, int]
    targets: tuple[AmazonReview, ...]


def build_window(
    events: Sequence[AmazonReview], start_ms: int, end_ms: int, threshold: float = 4.0
) -> ProductWindow:
    """窗口内首次新父商品评价定义目标；历史已见重复不当作新商品。"""
    if type(end_ms) is not int or end_ms <= start_ms or not 1 <= threshold <= 5:
        raise ValueError("需要有序毫秒窗口和1至5阈值")
    history = history_before(events, start_ms)
    seen: dict[str, set[str]] = {}
    for event in history:
        seen.setdefault(event.user_id, set()).add(event.item_id)
    future = sorted(
        (r for r in events if start_ms <= r.timestamp_ms < end_ms),
        key=lambda r: (r.timestamp_ms, r.user_id, r.item_id, r.variant_id, r.source, r.line_number),
    )
    users = tuple(sorted({r.user_id for r in future}))
    first_new: dict[tuple[str, str], AmazonReview] = {}
    at_time: dict[tuple[str, str, int], float] = {}
    repeated = 0
    for event in future:
        timed_pair = (event.user_id, event.item_id, event.timestamp_ms)
        previous = at_time.get(timed_pair)
        if previous is not None and previous != event.rating:
            raise ValueError(f"{event.source}:{event.line_number}: 窗口同时刻评分冲突")
        at_time[timed_pair] = event.rating
        if event.item_id in seen.get(event.user_id, set()):
            repeated += 1
        else:
            first_new.setdefault((event.user_id, event.item_id), event)
    relevant: dict[str, set[str]] = {user: set() for user in users}
    for event in first_new.values():
        if event.rating >= threshold:
            relevant[event.user_id].add(event.item_id)
    return ProductWindow(
        history,
        users,
        seen,
        relevant,
        {
            "n_window_events": len(future),
            "n_repeat_events": repeated,
            "n_new_pairs": len(first_new),
            "n_duplicate_new_events": len(future) - repeated - len(first_new),
            "n_window_users": len(users),
            "n_cold_users": sum(user not in seen for user in users),
            "n_positive_targets": sum(map(len, relevant.values())),
        },
        tuple(first_new[pair] for pair in sorted(first_new)),
    )


def fit_popularity(
    history: Sequence[AmazonReview], candidate_ids: Sequence[str], threshold: float = 4.0
) -> dict[str, int]:
    """复用已有计数；候选只能取自历史，不从展示元数据补入未来商品。"""
    if len(candidate_ids) != len(set(candidate_ids)) or not 1 <= threshold <= 5:
        raise ValueError("候选必须唯一，阈值在1至5")
    counts = count_positive_ratings([(r.user_id, r.item_id, r.rating) for r in history], threshold)
    if not set(candidate_ids) <= counts.keys():
        raise ValueError("候选包含历史目录外商品")
    return {item: counts[item] for item in sorted(candidate_ids)}


def predict_popularity(
    counts: dict[str, int],
    seen: Mapping[str, Set[str]],
    user_ids: Iterable[str],
    k: int = 10,
) -> dict[str, list[str]]:
    """无标签预测；没有候选或候选不足时保留空/短列表。"""
    if type(k) is not int or k <= 0:
        raise ValueError("k必须为正整数")
    return {
        user: recommend_popular(counts, set(seen.get(user, set())), k)
        for user in sorted(set(user_ids))
    }


def evaluate_window(
    predictions: Mapping[str, Sequence[str]],
    window: ProductWindow,
    candidate_ids: Iterable[str],
    k: int = 10,
) -> dict:
    """预测生成后独立计分；目录外正例仍在分母，所有排除量有计数。"""
    catalog = set(candidate_ids)
    for user, ranked in predictions.items():
        if (
            len(ranked) > k
            or len(ranked) != len(set(ranked))
            or not set(ranked) <= catalog
            or set(ranked) & window.seen.get(user, set())
        ):
            raise ValueError("推荐列表违反长度、唯一性、目录或已见过滤规则")
    metrics = asdict(evaluate_users(predictions, window.relevant, k))
    return {
        "metrics": metrics,
        "stats": {
            **window.stats,
            "n_candidate_items": len(catalog),
            "n_positive_targets_outside_catalog": sum(
                len(items - catalog) for items in window.relevant.values()
            ),
            "n_users_without_candidates": sum(
                not (catalog - window.seen.get(user, set())) for user in window.users
            ),
        },
    }
