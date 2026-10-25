"""四组配对统计：用户/故事等权，失败留分母，标签只在评分端使用。"""

from collections import defaultdict
from dataclasses import asdict
from statistics import fmean

import numpy as np

from memrecagent.metrics import evaluate_ranking

GROUPS = ("G00", "G10", "G01", "G11")
METRICS = ("precision", "recall", "hit_rate", "ndcg")


def _identifiers(value: object) -> bool:
    return (
        isinstance(value, list)
        and all(isinstance(v, str) and v.strip() for v in value)
        and len(value) == len(set(value))
    )


def _validate(rows: list[dict]) -> None:
    if not rows or len(rows) > 100_000:
        raise ValueError("需要1至100000个完整四组案例")
    seen = set()
    for row in rows:
        if not isinstance(row, dict) or row.keys() != {
            "case_id",
            "cluster_id",
            "stratum",
            "relevant",
            "excluded",
            "groups",
        }:
            raise ValueError("案例字段无效")
        if any(
            not isinstance(row[k], str) or not row[k].strip()
            for k in ("case_id", "cluster_id", "stratum")
        ):
            raise ValueError("案例、聚类与分层标识不能为空")
        if row["case_id"] in seen:
            raise ValueError("重复案例不能充当新增样本")
        seen.add(row["case_id"])
        if not _identifiers(row["relevant"]) or not _identifiers(row["excluded"]):
            raise ValueError("答案和约束必须为去重ID列表")
        groups = row["groups"]
        if not isinstance(groups, dict) or set(groups) != set(GROUPS):
            raise ValueError("必须保留全部四组及失败")
        for group in GROUPS:
            arm = groups[group]
            if not isinstance(arm, dict) or arm.keys() != {
                "item_ids",
                "status",
                "algorithm_effective",
            }:
                raise ValueError("组结果字段无效")
            if (
                not _identifiers(arm["item_ids"])
                or arm["status"] not in ("ready", "clarify", "error")
                or type(arm["algorithm_effective"]) is not bool
                or (arm["status"] != "ready" and arm["item_ids"])
                or (group in ("G00", "G10") and arm["algorithm_effective"])
            ):
                raise ValueError("组结果包含非法状态、重复商品或错误算法标记")


def _summary(rows: list[dict], *, k: int, n_resamples: int, seed: int) -> dict:
    clusters = sorted({r["cluster_id"] for r in rows})
    eligible = [r for r in rows if r["relevant"]]
    eligible_clusters = sorted({r["cluster_id"] for r in eligible})
    groups = {}
    matrix = np.empty((len(eligible_clusters), 4), dtype=np.float64)
    for column, group in enumerate(GROUPS):
        by_cluster = defaultdict(list)
        for row in eligible:
            metrics = evaluate_ranking(row["groups"][group]["item_ids"], set(row["relevant"]), k)
            by_cluster[row["cluster_id"]].append(asdict(metrics))
        cluster_scores = {
            cluster: {key: fmean(r[key] for r in values) for key in METRICS}
            for cluster, values in by_cluster.items()
        }
        for index, cluster in enumerate(eligible_clusters):
            matrix[index, column] = cluster_scores[cluster]["ndcg"]
        arms = [r["groups"][group] for r in rows]
        violations = [
            len(set(r["groups"][group]["item_ids"][:k]) & set(r["excluded"])) for r in rows
        ]
        groups[group] = {
            **{
                key: fmean(v[key] for v in cluster_scores.values()) if cluster_scores else None
                for key in METRICS
            },
            "case_mean_ndcg": fmean(v["ndcg"] for vs in by_cluster.values() for v in vs)
            if eligible
            else None,
            "n_errors": sum(a["status"] == "error" for a in arms),
            "n_clarify": sum(a["status"] == "clarify" for a in arms),
            "n_empty": sum(not a["item_ids"] for a in arms),
            "empty_rate": sum(not a["item_ids"] for a in arms) / len(rows),
            "n_constraint_violations": sum(v > 0 for v in violations),
            "n_violating_items": sum(violations),
            "constraint_violation_rate": sum(v > 0 for v in violations) / len(rows),
            "n_algorithm_effective": sum(a["algorithm_effective"] for a in arms),
        }
    # 三个对比共享聚类抽样；每个cluster内部的全部轮保持绑定。
    coefficients = np.array([[0, 0, -1, 1], [0, -1, 0, 1], [1, -1, -1, 1]], dtype=float)
    names = ("memory_given_algorithm", "algorithm_given_memory", "interaction")
    contrasts = {}
    if eligible_clusters:
        differences = matrix @ coefficients.T
        distribution = np.empty((n_resamples, 3), dtype=float)
        rng = np.random.default_rng(seed)
        batch = max(1, min(256, 1_000_000 // len(eligible_clusters)))
        for start in range(0, n_resamples, batch):
            count = min(batch, n_resamples - start)
            indices = rng.integers(0, len(eligible_clusters), size=(count, len(eligible_clusters)))
            distribution[start : start + count] = differences[indices].mean(axis=1)
        intervals = np.quantile(distribution, [0.025, 0.975], axis=0, method="linear")
        for index, name in enumerate(names):
            contrasts[name] = {
                "delta": float(differences[:, index].mean()),
                "interval": intervals[:, index].tolist(),
            }
    else:
        contrasts = {name: {"delta": None, "interval": None} for name in names}
    return {
        "k": k,
        "n_cases": len(rows),
        "n_clusters": len(clusters),
        "n_eligible_clusters": len(eligible_clusters),
        "n_without_relevant": len(rows) - len(eligible),
        "n_effective_algorithm_cases": sum(
            r["groups"]["G01"]["algorithm_effective"] and r["groups"]["G11"]["algorithm_effective"]
            for r in rows
        ),
        "groups": groups,
        "contrasts": contrasts,
        "confidence": 0.95,
        "n_resamples": n_resamples,
        "seed": seed,
        "method": "equal cluster mean; paired cluster percentile bootstrap; NDCG contrasts",
        "rate_denominator": "all cases, including errors, clarification and empty relevance",
        "inference_limit": "descriptive for the supplied cohort; not real-user generalisation",
    }


def summarize_ablation(
    rows: list[dict], *, k: int, n_resamples: int = 10_000, seed: int = 42
) -> dict:
    """主指标按cluster等权；分层单列，不删除候选外答案和过程失败。"""
    if type(k) is not int or not 1 <= k <= 50:
        raise ValueError("k必须是1至50整数")
    if type(n_resamples) is not int or not 1 <= n_resamples <= 100_000:
        raise ValueError("bootstrap次数必须是1至100000整数")
    if type(seed) is not int or not 0 <= seed < 2**64:
        raise ValueError("seed必须是非负64位整数")
    _validate(rows)
    result = _summary(rows, k=k, n_resamples=n_resamples, seed=seed)
    result["strata"] = {
        stratum: _summary(
            [r for r in rows if r["stratum"] == stratum], k=k, n_resamples=n_resamples, seed=seed
        )
        for stratum in sorted({r["stratum"] for r in rows})
    }
    return result
