"""Binary relevance ranking metrics shared by recommendation baselines."""

from collections.abc import Mapping, Sequence, Set
from dataclasses import dataclass
from math import log2
from statistics import fmean


@dataclass(frozen=True)
class RankingMetrics:
    """Scores at the K supplied to the evaluator; relevance is binary."""

    precision: float
    recall: float
    hit_rate: float
    ndcg: float


@dataclass(frozen=True)
class EvaluationReport:
    """Macro means and explicit counts; None means no eligible users."""

    k: int
    n_users: int
    n_evaluated: int
    n_without_relevant: int
    mean: RankingMetrics | None


def evaluate_ranking(recommendations: Sequence[str], relevant: Set[str], k: int) -> RankingMetrics:
    """Evaluate one user's ordered, unique recommendations.

    Precision divides by requested K even when the list is shorter. Recall
    divides by all relevant items, without intersecting the recommendation
    list. The caller defines eligible candidates and relevance consistently.
    Empty relevance is excluded explicitly by evaluate_users instead.
    """
    if k <= 0:
        raise ValueError("k must be positive")
    if len(set(recommendations)) != len(recommendations):
        raise ValueError("recommendations must not contain duplicate items")
    if not relevant:
        raise ValueError("At least one relevant item is required")

    # 推荐顺序决定折扣；不能把推荐列表转换成集合后计算 NDCG。
    hits = [item in relevant for item in recommendations[:k]]
    n_hits = sum(hits)
    dcg = sum(hit / log2(rank + 1) for rank, hit in enumerate(hits, start=1))
    ideal_dcg = sum(1 / log2(rank + 1) for rank in range(1, min(k, len(relevant)) + 1))
    return RankingMetrics(
        precision=n_hits / k,
        recall=n_hits / len(relevant),
        hit_rate=float(n_hits > 0),
        ndcg=dcg / ideal_dcg,
    )


def evaluate_users(
    recommendations: Mapping[str, Sequence[str]],
    relevant: Mapping[str, Set[str]],
    k: int,
) -> EvaluationReport:
    """Average each eligible user's metrics with equal user weight.

    Both mappings must contain the same evaluation cohort. Use an explicit
    empty list for a user with no recommendations; that user still counts.
    Users without test positives are counted separately and excluded from
    all four macro means. If all are excluded, mean is None, never a fake 0.
    """
    if k <= 0:
        raise ValueError("k must be positive")
    if recommendations.keys() != relevant.keys():
        raise ValueError("recommendations and relevant must have the same users")

    per_user = [
        evaluate_ranking(recommendations[user], answers, k)
        for user, answers in relevant.items()
        if answers
    ]
    mean = None
    if per_user:
        mean = RankingMetrics(
            precision=fmean(row.precision for row in per_user),
            recall=fmean(row.recall for row in per_user),
            hit_rate=fmean(row.hit_rate for row in per_user),
            ndcg=fmean(row.ndcg for row in per_user),
        )
    return EvaluationReport(
        k=k,
        n_users=len(relevant),
        n_evaluated=len(per_user),
        n_without_relevant=len(relevant) - len(per_user),
        mean=mean,
    )
