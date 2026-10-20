"""迁移现有MF/BPR，固定候选与预算，验证选择先于任何模型测试。"""

import argparse
import json
import platform
import resource
import sys
import time
from dataclasses import asdict
from datetime import UTC, datetime
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path
from uuid import uuid4

import torch
from evaluate_movielens import capture_source, save_json
from evaluate_products import DEFAULT_DATA, load_prepared

from memrecagent.amazon_preparation import TEST_END_MS, TRAIN_END_MS, VALIDATION_END_MS
from memrecagent.bpr_evaluation import BPRConfig, FittedBPR, fit_bpr_fixed, select_bpr
from memrecagent.mf_evaluation import FittedMF, MFConfig, fit_fixed, select_mf
from memrecagent.movielens import Rating
from memrecagent.product_evaluation import (
    build_window,
    evaluate_window,
    fit_popularity,
    predict_popularity,
)
from memrecagent.product_models import predict_model

ROOT = Path(__file__).resolve().parents[1]


def save_checkpoint(path: Path, fitted: FittedMF, config: MFConfig) -> None:
    """只保存张量与基本类型，不序列化可执行模型对象。"""
    torch.save(
        {
            "state_dict": fitted.model.state_dict(),
            "users": fitted.users,
            "items": fitted.items,
            "seen": {user: sorted(items) for user, items in fitted.seen.items()},
            "active_users": sorted(fitted.active_users) if isinstance(fitted, FittedBPR) else None,
            "config": asdict(config),
            "model_kind": "bpr" if isinstance(fitted, FittedBPR) else "mf",
        },
        path,
    )


def main() -> None:
    """保留实际目标评分/毫秒时间，仅向已有训练器传递合法历史。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-root", type=Path, default=ROOT / "artifacts/products_models")
    parser.add_argument("--max-epochs", type=int, default=20)
    parser.add_argument("--checkpoint-every", type=int, default=5)
    parser.add_argument("--factors", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()
    if not (
        1 <= args.max_epochs <= 20
        and 1 <= args.checkpoint_every <= args.max_epochs
        and 1 <= args.factors <= 16
        and 1 <= args.batch_size <= 256
        and 1 <= args.k <= 50
    ):
        parser.error("参数超出已批准的有限预算")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    events, candidates, data = load_prepared(args.data)
    if len(events) > 200000 or len(candidates) > 5000:
        raise ValueError("数据超过阶段预算，不能静默截断")
    catalog = set(candidates)
    validation = build_window(events, TRAIN_END_MS, VALIDATION_END_MS)
    history = [
        Rating(r.user_id, r.item_id, r.rating, r.timestamp_ms)
        for r in validation.history
        if r.item_id in catalog
    ]
    targets = [Rating(r.user_id, r.item_id, r.rating, r.timestamp_ms) for r in validation.targets]
    mf_config = MFConfig(
        factors=args.factors,
        max_epochs=args.max_epochs,
        checkpoint_every=args.checkpoint_every,
        batch_size=args.batch_size,
        lr=0.5,
        l2=0.001,
        seed=42,
    )
    bpr_config = BPRConfig(**asdict(mf_config), threshold=4.0, materialize_pools=False)
    configs = {"mf": mf_config, "bpr": bpr_config}
    created = datetime.now(UTC)
    run = args.output_root / f"{created.strftime('%Y%m%dT%H%M%S%fZ')}-{uuid4().hex[:8]}"
    run.mkdir(parents=True, exist_ok=False)
    config = {
        "protocol": "p2-products-v1",
        "k": args.k,
        "threshold": 4.0,
        "timestamp_unit": "milliseconds",
        "cutoffs_ms": [TRAIN_END_MS, VALIDATION_END_MS, TEST_END_MS],
        "mf": asdict(mf_config),
        "bpr": asdict(bpr_config),
        "candidate_ids_sha256": data["sha256"]["candidates.json"],
    }
    save_json(run / "config.json", config)
    code = capture_source(run)
    started = time.perf_counter()
    selections, timings = {}, {}
    for name, select in [("mf", select_mf), ("bpr", select_bpr)]:
        before = time.perf_counter()
        print(f"开始验证选择 {name}", flush=True)
        selections[name] = select(history, targets, configs[name], k=args.k)
        timings[f"{name}_selection"] = time.perf_counter() - before
        save_checkpoint(run / f"{name}_validation.pt", selections[name].fitted, configs[name])
    # 两种模型的全部验证选择先落盘；本行之后才构造/评测测试答案。
    target_users = {row.user_id for row in validation.targets}
    repeat_only_users = len(validation.users) - len(target_users)
    # 旧选择器只接收新商品目标；补列只有已见重复评价、没有正例的窗口用户。
    # 四项均值与选轮保持原值，仅让审计人数与完整商品评测使用同一窗口口径。
    selection_rows = {
        name: {
            "best_epoch": result.best_epoch,
            "cohort": {
                "n_window_users": len(validation.users),
                "n_target_users": len(target_users),
                "n_repeat_only_users": repeat_only_users,
            },
            "checkpoints": [
                {
                    **row,
                    "validation_metrics": {
                        **row["validation_metrics"],
                        "n_users": len(validation.users),
                        "n_without_relevant": row["validation_metrics"]["n_without_relevant"]
                        + repeat_only_users,
                    },
                }
                for row in result.checkpoints
            ],
        }
        for name, result in selections.items()
    }
    save_json(run / "selection.json", selection_rows)
    test = build_window(events, VALIDATION_END_MS, TEST_END_MS)
    final_history = [
        Rating(r.user_id, r.item_id, r.rating, r.timestamp_ms)
        for r in test.history
        if r.item_id in catalog
    ]
    fitted_models = {}
    for name, fit in [("mf", fit_fixed), ("bpr", fit_bpr_fixed)]:
        before = time.perf_counter()
        print(f"开始从头重训 {name}: {selections[name].best_epoch}轮", flush=True)
        fitted_models[name] = fit(final_history, configs[name], selections[name].best_epoch)
        timings[f"{name}_refit"] = time.perf_counter() - before
        save_checkpoint(run / f"{name}_test.pt", fitted_models[name], configs[name])
    stages, predictions = {}, {}
    for stage, window in [("validation", validation), ("test", test)]:
        before = time.perf_counter()
        counts = fit_popularity(window.history, candidates)
        ranked = {"popularity": predict_popularity(counts, window.seen, window.users, args.k)}
        save_json(run / f"popularity_{stage}.json", counts)
        for name in ["mf", "bpr"]:
            fitted = selections[name].fitted if stage == "validation" else fitted_models[name]
            ranked[name] = predict_model(fitted, window.users, window.seen, args.k)
        evaluated = {
            name: evaluate_window(items, window, candidates, args.k)
            for name, items in ranked.items()
        }
        stages[stage] = {
            "stats": evaluated["popularity"]["stats"],
            "metrics": {name: value["metrics"] for name, value in evaluated.items()},
        }
        predictions[stage] = {
            "predictions": ranked,
            "relevant": {user: sorted(items) for user, items in window.relevant.items()},
        }
        timings[f"{stage}_ranking"] = time.perf_counter() - before
    save_json(run / "predictions.json", predictions)
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    report = {
        "created_at_utc": created.isoformat(),
        "config": config,
        "code": code,
        "data": data,
        "selection": selection_rows,
        "stages": stages,
        "timings_seconds": timings,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_rss_bytes": rss if sys.platform == "darwin" else rss * 1024,
        "checkpoint_sha256": {
            path.name: sha256(path.read_bytes()).hexdigest() for path in sorted(run.glob("*.pt"))
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "backend": "CPU float32 single thread deterministic",
            "packages": {name: version(name) for name in ["torch", "numpy"]},
        },
        "limits": [
            "single seed fixed settings",
            "test viewed after popularity; no retuning",
            "review discovery not purchase or memory effect",
        ],
    }
    save_json(run / "report.json", report)
    print(
        json.dumps(
            {
                "run": str(run.resolve()),
                "selection": {name: row["best_epoch"] for name, row in selection_rows.items()},
                "test": stages["test"],
                "elapsed_seconds": report["elapsed_seconds"],
                "peak_rss_bytes": report["peak_rss_bytes"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
