"""完整冻结候选的开发集敏感度；不调用模型、不择优改默认、不读取测试集。"""

import json
from collections import defaultdict
from hashlib import sha256
from math import isfinite
from pathlib import Path
from statistics import fmean

import numpy as np

from memrecagent.external_replay import graded_ranking_metrics
from memrecagent.memory_product_tool import _read
from memrecagent.memory_quality_evaluation import _source_paths

GROUPS = ("G00", "G10", "G01", "G11")
NORMALISATIONS = ("minmax", "rank")
WEIGHTS = (0.0, 0.25, 0.5, 0.75, 1.0)


def _validate(suite: dict) -> None:
    if (
        not isinstance(suite, dict)
        or suite.keys() != {"schema_version", "evidence_kind", "split", "cases"}
        or type(suite["schema_version"]) is not int
        or suite["schema_version"] != 1
        or suite["evidence_kind"] != "saved_response_replay"
        or suite["split"] != "dev"
        or not isinstance(suite["cases"], list)
        or not 1 <= len(suite["cases"]) <= 1000
    ):
        raise ValueError("invalid development feature suite")
    seen = set()
    for c in suite["cases"]:
        if not isinstance(c, dict) or c.keys() != {
            "case_id",
            "cluster_id",
            "cohort",
            "reference_status",
            "reference",
            "frames",
        }:
            raise ValueError("invalid case")
        if (
            any(
                not isinstance(c[k], str) or not c[k].strip()
                for k in ("case_id", "cluster_id", "cohort")
            )
            or c["case_id"] in seen
            or c["cohort"] not in ("direct", "analyst", "analyst_blind")
            or c["reference_status"] not in ("provisional", "review_required")
            or not isinstance(c["reference"], list)
            or not 1 <= len(c["reference"]) <= 5000
            or not isinstance(c["frames"], dict)
            or set(c["frames"]) != set(NORMALISATIONS)
        ):
            raise ValueError("invalid case identity or reference")
        seen.add(c["case_id"])
        ids = set()
        for r in c["reference"]:
            if (
                not isinstance(r, dict)
                or r.keys() != {"item_id", "eligible", "relevance"}
                or not isinstance(r["item_id"], str)
                or not r["item_id"].strip()
                or r["item_id"] in ids
                or type(r["eligible"]) is not bool
                or type(r["relevance"]) is not int
                or not 0 <= r["relevance"] <= 3
            ):
                raise ValueError("invalid graded reference")
            ids.add(r["item_id"])
        basis = {}
        for norm in NORMALISATIONS:
            if not isinstance(c["frames"][norm], dict) or set(c["frames"][norm]) != set(GROUPS):
                raise ValueError("missing group")
            for group, arm in c["frames"][norm].items():
                if not isinstance(arm, dict) or arm.keys() != {
                    "status",
                    "used_strategy",
                    "algorithm_effective",
                    "n_candidates",
                    "n_filtered",
                    "items",
                }:
                    raise ValueError("invalid frame")
                if (
                    arm["status"] not in ("ready", "clarify", "error")
                    or arm["used_strategy"] not in (None, "popularity", "mf", "bpr")
                    or type(arm["algorithm_effective"]) is not bool
                    or type(arm["n_candidates"]) is not int
                    or type(arm["n_filtered"]) is not int
                    or not 0 <= arm["n_filtered"] <= arm["n_candidates"] <= 5000
                    or not isinstance(arm["items"], list)
                    or len(arm["items"]) > 50
                    or (
                        arm["status"] == "ready"
                        and len(arm["items"]) != arm["n_candidates"] - arm["n_filtered"]
                    )
                    or (arm["status"] != "ready" and arm["items"])
                    or (
                        arm["algorithm_effective"]
                        and (group in ("G00", "G10") or arm["used_strategy"] not in ("mf", "bpr"))
                    )
                ):
                    raise ValueError("truncated or invalid candidate frame")
                items = set()
                values = []
                for item in arm["items"]:
                    if (
                        not isinstance(item, dict)
                        or item.keys()
                        != {"item_id", "base_score", "normalised_base_score", "preference_signal"}
                        or item["item_id"] not in ids
                        or item["item_id"] in items
                        or any(
                            type(item[k]) not in (int, float) or not isfinite(item[k])
                            for k in ("base_score", "normalised_base_score")
                        )
                        or not 0 <= item["normalised_base_score"] <= 1
                        or type(item["preference_signal"]) is not int
                        or abs(item["preference_signal"]) > 100
                    ):
                        raise ValueError("invalid feature")
                    items.add(item["item_id"])
                    values.append((item["item_id"], item["base_score"], item["preference_signal"]))
                identity = (
                    arm["status"],
                    arm["used_strategy"],
                    arm["algorithm_effective"],
                    arm["n_candidates"],
                    arm["n_filtered"],
                    sorted(values),
                )
                if norm == "minmax":
                    basis[group] = identity
                elif basis[group] != identity:
                    raise ValueError("normalisation changed candidate basis")


def sensitivity_plan(suite: dict) -> dict:
    """只验证开发候选契约，不创建目录，不执行排名或选择配置。"""
    _validate(suite)
    return dict(
        api_calls=0,
        n_cases=len(suite["cases"]),
        normalisations=list(NORMALISATIONS),
        weights=list(WEIGHTS),
        n_configurations=10,
        default=dict(normalisation="minmax", weight=0.75),
    )


def _mean_by_cluster(rows: list[dict], group: str) -> dict[str, float]:
    buckets = defaultdict(list)
    for r in rows:
        score = r["groups"][group]["metrics"]["ndcg"]
        if score is not None:
            buckets[r["cluster_id"]].append(score)
    return {key: fmean(values) for key, values in buckets.items()}


def _paired(current: dict, baseline: dict, *, n_resamples: int, seed: int) -> dict:
    if set(current) != set(baseline):
        raise ValueError("paired clusters differ")
    if not current:
        return dict(delta=None, interval=None, n_clusters=0)
    differences = np.array([current[c] - baseline[c] for c in sorted(current)])
    rng = np.random.default_rng(seed)
    # 每轮全族重采样，包含各族全部案例；不把多变体当新增独立样本。
    indices = rng.integers(0, len(differences), size=(n_resamples, len(differences)))
    distribution = differences[indices].mean(axis=1)
    return dict(
        delta=float(differences.mean()),
        interval=np.quantile(distribution, [0.025, 0.975]).tolist(),
        n_clusters=len(differences),
    )


def evaluate_sensitivity(
    suite: dict, *, k: int = 3, n_resamples: int = 1000, seed: int = 20261024
) -> dict:
    """保持全部错误和待复核案例，参考只进入评分函数，不改变候选和信号。"""
    plan = sensitivity_plan(suite)
    if (
        type(k) is not int
        or not 1 <= k <= 50
        or type(n_resamples) is not int
        or not 1 <= n_resamples <= 10000
        or type(seed) is not int
        or not 0 <= seed < 2**63
    ):
        raise ValueError("invalid statistics configuration")
    rows = []
    for case in suite["cases"]:
        for norm in NORMALISATIONS:
            for weight in WEIGHTS:
                groups = {}
                for group, arm in case["frames"][norm].items():
                    # 数据入口已保证完整候选；重算整个列表后才截取K。
                    ordered = sorted(
                        arm["items"],
                        key=lambda i: (
                            -(i["normalised_base_score"] + weight * i["preference_signal"]),
                            -i["base_score"],
                            i["item_id"],
                        ),
                    )
                    ids = [i["item_id"] for i in ordered[:k]] if arm["status"] == "ready" else []
                    groups[group] = dict(
                        status=arm["status"],
                        item_ids=ids,
                        used_strategy=arm["used_strategy"],
                        algorithm_effective=arm["algorithm_effective"],
                        metrics=graded_ranking_metrics(ids, case["reference"], k=k),
                    )
                rows.append(
                    {
                        key: case[key]
                        for key in ("case_id", "cluster_id", "cohort", "reference_status")
                    }
                    | dict(normalisation=norm, weight=weight, groups=groups)
                )
    summaries = []
    for cohort in sorted({c["cohort"] for c in suite["cases"]}):
        for scope in ("provisional", "all_including_review"):
            selected = [
                r
                for r in rows
                if r["cohort"] == cohort
                and (scope != "provisional" or r["reference_status"] == "provisional")
            ]
            if not selected:
                continue
            baseline = [
                r for r in selected if r["normalisation"] == "minmax" and r["weight"] == 0.75
            ]
            for norm in NORMALISATIONS:
                for weight in WEIGHTS:
                    current = [
                        r for r in selected if r["normalisation"] == norm and r["weight"] == weight
                    ]
                    groups, contrasts = {}, {}
                    for group in GROUPS:
                        clusters = _mean_by_cluster(current, group)
                        arms = [r["groups"][group] for r in current]
                        groups[group] = dict(
                            ndcg=fmean(clusters.values()) if clusters else None,
                            n_cases=len(current),
                            n_errors=sum(a["status"] == "error" for a in arms),
                            n_clarify=sum(a["status"] == "clarify" for a in arms),
                            n_without_gain=sum(a["metrics"]["ndcg"] is None for a in arms),
                            constraint_violations=sum(
                                a["metrics"]["constraint_violations"] for a in arms
                            ),
                            n_empty=sum(not a["item_ids"] for a in arms),
                            n_algorithm_effective=sum(a["algorithm_effective"] for a in arms),
                        )
                        contrasts[group] = _paired(
                            clusters,
                            _mean_by_cluster(baseline, group),
                            n_resamples=n_resamples,
                            seed=seed,
                        )
                    summaries.append(
                        dict(
                            cohort=cohort,
                            reference_scope=scope,
                            normalisation=norm,
                            weight=weight,
                            n_clusters=len({r["cluster_id"] for r in current}),
                            groups=groups,
                            paired_vs_default=contrasts,
                        )
                    )
    return plan | dict(
        k=k,
        n_resamples=n_resamples,
        seed=seed,
        selected_configuration=None,
        algorithm_effect=None,
        rows=rows,
        summaries=summaries,
        inference=(
            "exploratory fixed-memory development sensitivity; equal family mean and paired "
            "percentile bootstrap; no multiplicity correction or test selection"
        ),
    )


def replay_sensitivity(*, suite_path: Path, output_dir: Path, run: bool = False) -> dict:
    """离线CLI预检；显式run才建新目录并保存源码/输入/结果。"""
    raw, suite = _read(suite_path, 16 * 1024 * 1024)
    plan = sensitivity_plan(suite)
    source = Path(__file__).resolve().parent
    target = output_dir.resolve()
    if (
        target == source
        or source in target.parents
        or target in source.parents
        or target == suite_path.resolve()
        or target in suite_path.resolve().parents
    ):
        raise ValueError("unsafe output path")
    if not run:
        return plan | dict(executed=False)
    paths = _source_paths()
    script = Path(__file__).resolve().parents[2] / "scripts/evaluate_ranking_sensitivity.py"
    if script.is_file():
        paths[script.name] = script
    hashes = {name: sha256(path.read_bytes()).hexdigest() for name, path in paths.items()}
    output_dir.mkdir(parents=True, exist_ok=False)
    for name, path in paths.items():
        dest = output_dir / "source" / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(path.read_bytes())
    (output_dir / "inputs.json").write_bytes(raw)
    (output_dir / "manifest.json").write_text(
        json.dumps(
            dict(source_sha256=hashes, inputs_sha256=sha256(raw).hexdigest(), **plan),
            sort_keys=True,
        )
    )
    report = evaluate_sensitivity(suite)
    if (
        any(sha256(path.read_bytes()).hexdigest() != hashes[name] for name, path in paths.items())
        or suite_path.read_bytes() != raw
    ):
        raise ValueError("source or input changed")
    report["source_inputs_unchanged"] = True
    (output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, sort_keys=True, allow_nan=False)
    )
    return {key: value for key, value in report.items() if key not in ("rows", "summaries")} | dict(
        executed=True
    )
