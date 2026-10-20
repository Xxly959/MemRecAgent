"""Compare rating MF with existing MovieLens baselines using validation epoch selection."""

import argparse
import platform
import sys
import time
from dataclasses import asdict
from datetime import UTC, datetime
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path
from uuid import uuid4

import torch
from evaluate_movielens import capture_source, save_json, utc_boundary

from memrecagent.evaluation import evaluate_holdout
from memrecagent.metrics import evaluate_users
from memrecagent.mf_evaluation import MFConfig, fit_fixed, rank_mf, select_mf
from memrecagent.movielens import load_ratings, temporal_split

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data/raw/ml-100k/u.data")
    parser.add_argument("--output-root", type=Path, default=ROOT / "artifacts/movielens_mf")
    parser.add_argument("--train-end", default="1998-03-01")
    parser.add_argument("--validation-end", default="1998-04-01")
    parser.add_argument("--threshold", type=float, default=4.0)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--factors", type=int, default=16)
    parser.add_argument("--max-epochs", type=int, default=80)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.5)
    parser.add_argument("--l2", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    try:
        config = MFConfig(
            args.factors,
            args.max_epochs,
            args.checkpoint_every,
            args.batch_size,
            args.lr,
            args.l2,
            args.seed,
        )
        if args.k <= 0 or not 1 <= args.threshold <= 5:
            raise ValueError("Positive K and threshold in [1,5] required")
        records = load_ratings(args.data)
        train_end, validation_end = utc_boundary(args.train_end), utc_boundary(args.validation_end)
        train, validation, test = temporal_split(records, train_end, validation_end)
        if not train or not validation or not test:
            raise ValueError("Each time window must contain records")
        if not any(r.rating >= args.threshold for r in validation):
            raise ValueError("Validation must have positive targets")
    except (OSError, ValueError) as exc:
        parser.error(str(exc))

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    created = datetime.now(UTC)
    run = args.output_root / (created.strftime("%Y%m%dT%H%M%S%fZ-") + uuid4().hex[:8])
    run.mkdir(parents=True, exist_ok=False)
    protocol = {
        "mf": asdict(config),
        "train_end_utc": args.train_end,
        "validation_end_utc": args.validation_end,
        "threshold": args.threshold,
        "k": args.k,
        "selection": "max validation macro Recall@K; ties earliest checkpoint; epoch0 excluded",
        "refit": "fresh model on train+validation for chosen epoch count; no test selection",
        "catalog": "all items observed in fitting history; all scores, including negative/zero",
        "filter": "all observed user-item pairs, including low ratings",
        "ties": "string item ID ascending",
        "relevance": "all holdout positives, including outside catalog, stay in denominator",
        "cold_user": "uniform zero scores for MF/ItemCF; no popularity fallback",
        "cohort": "all users with holdout ratings; count and exclude no-positive users",
        "loss": "observed-rating MSE + L2 mean per-interaction vector squared norms",
        "timestamp": "rating time used as availability proxy; no within-window updates",
        "test_reused": True,
        "scope": "single-seed exploratory comparison, not a new blind test",
    }
    save_json(run / "config.json", protocol)
    code = capture_source(run)
    start = time.perf_counter()
    print(f"Run directory: {run.resolve()}", flush=True)
    print("Selecting MF epoch on validation only...", flush=True)
    selected = select_mf(train, validation, config, args.k, args.threshold)
    selection = {
        "best_epoch": selected.best_epoch,
        "criterion": "validation Recall@K",
        "checkpoints": selected.checkpoints,
    }
    save_json(run / "selection.json", selection)  # 在计算测试指标前冻结。
    for row in selected.checkpoints:
        print(
            f"epoch {row['epoch']:3}: train MSE={row['train_mse']:.6f}, "
            f"validation Recall@{args.k}={row['validation_metrics']['mean']['recall']:.6f}",
            flush=True,
        )
    print(f"Selected epoch: {selected.best_epoch}; refitting train+validation...", flush=True)
    refit = fit_fixed(train + validation, config, selected.best_epoch)
    mf_test = rank_mf(refit, test, args.k, args.threshold)

    stages, predictions, subgroups = {}, {}, {}
    for name, history, holdout, mf_result, fitted in [
        ("validation", train, validation, selected.validation, selected.fitted),
        ("test", train + validation, test, mf_test, refit),
    ]:
        baseline = evaluate_holdout(history, holdout, args.threshold, args.k)
        if mf_result["relevant"] != baseline["relevant"]:
            raise RuntimeError("MF and baseline relevance protocols disagree")
        baseline["metrics"]["mf"] = mf_result["metrics"]
        baseline["predictions"]["mf"] = mf_result["predictions"]
        stages[name] = {key: baseline[key] for key in ["stats", "metrics"]}
        predictions[name] = {key: baseline[key] for key in ["predictions", "relevant", "users"]}
        subgroups[name] = {}
        for group, cold in [("warm", False), ("cold", True)]:
            answers = {
                u: set(items)
                for u, items in baseline["relevant"].items()
                if baseline["users"][u]["cold_user"] == cold
            }
            subgroups[name][group] = {
                model: asdict(evaluate_users({u: ranks[u] for u in answers}, answers, args.k))
                for model, ranks in baseline["predictions"].items()
            }
        with (run / f"mf_{name}.pt").open("xb") as stream:
            torch.save(
                {
                    "state_dict": fitted.model.state_dict(),
                    "users": fitted.users,
                    "items": fitted.items,
                    "config": asdict(config),
                    "epochs": selected.best_epoch,
                },
                stream,
            )
        print(f"\n{name}:", flush=True)
        for model, row in baseline["metrics"].items():
            print(model, row, flush=True)

    save_json(run / "predictions.json", predictions)
    report = {
        "created_at_utc": created.isoformat(),
        "config": protocol,
        "code": code,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "packages": {name: version(name) for name in ["torch", "numpy"]},
            "torch_threads": torch.get_num_threads(),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "backend": "CPU float32",
        },
        "data": {
            "path": str(args.data.resolve()),
            "sha256": sha256(args.data.read_bytes()).hexdigest(),
            "n_ratings": len(records),
        },
        "split_sizes": {"train": len(train), "validation": len(validation), "test": len(test)},
        "selection": selection,
        "stages": stages,
        "subgroups": subgroups,
        "refit_trace": refit.trace,
        "elapsed_seconds": time.perf_counter() - start,
    }
    save_json(run / "report.json", report)
    print(f"Report: {run / 'report.json'}", flush=True)


if __name__ == "__main__":
    main()
