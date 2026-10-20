"""Compare BPR, rating MF, and frozen baselines under one MovieLens protocol."""

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

from memrecagent.bpr_evaluation import BPRConfig, fit_bpr_fixed, rank_bpr, select_bpr
from memrecagent.evaluation import evaluate_holdout
from memrecagent.metrics import evaluate_users
from memrecagent.mf_evaluation import MFConfig, fit_fixed, rank_mf, select_mf
from memrecagent.movielens import load_ratings, temporal_split

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data/raw/ml-100k/u.data")
    parser.add_argument("--output-root", type=Path, default=ROOT / "artifacts/movielens_bpr")
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
    # MF retains its prior configuration; these limits allow small CLI fixtures.
    parser.add_argument("--mf-max-epochs", type=int, default=80)
    parser.add_argument("--mf-checkpoint-every", type=int, default=10)
    args = parser.parse_args()
    try:
        config = BPRConfig(
            factors=args.factors,
            max_epochs=args.max_epochs,
            checkpoint_every=args.checkpoint_every,
            batch_size=args.batch_size,
            lr=args.lr,
            l2=args.l2,
            seed=args.seed,
            threshold=args.threshold,
        )
        mf_config = MFConfig(
            max_epochs=args.mf_max_epochs, checkpoint_every=args.mf_checkpoint_every
        )
        if args.k <= 0:
            raise ValueError("Positive K required")
        records = load_ratings(args.data)
        train, validation, test = temporal_split(
            records, utc_boundary(args.train_end), utc_boundary(args.validation_end)
        )
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
        "bpr": asdict(config),
        "mf": asdict(mf_config),
        "train_end_utc": args.train_end,
        "validation_end_utc": args.validation_end,
        "threshold": args.threshold,
        "k": args.k,
        "selection": "each model max validation macro Recall@K; ties earliest; epoch0 excluded",
        "refit": "fresh train+validation models for their selected epoch counts; no test selection",
        "catalog": "all items observed in fitting history; all scores including negative/zero",
        "filter": "all observed user-item pairs including low ratings",
        "ties": "string item ID ascending",
        "relevance": "all holdout positives including outside catalog remain in denominator",
        "cold_user": "uniform zero scores for MF/ItemCF/BPR; no popularity fallback",
        "bpr_inactive_user": "history user without legal positive-negative triple also gets zeros",
        "sampling": "one uniform unobserved history-catalog j per positive pair each epoch",
        "rng": "local continuing Python sampling and independent Torch shuffle streams",
        "loss": "BPR mean softplus(sj-si) + L2 mean per-triple p/qi/qj squared norms",
        "training_log": (
            "record-weighted online losses; varying samples/weights; not fixed check loss"
        ),
        "sampling_audit": "counts and SHA256(json.dumps(ordered triples)); replay from source/seed",
        "cohort": "all holdout users; count and exclude users without positive targets",
        "timestamp": "rating time is availability proxy; no within-window updates",
        "test_reused": True,
        "scope": (
            "single-seed exploratory comparison; not new blind test or statistical significance"
        ),
    }
    save_json(run / "config.json", protocol)
    code = capture_source(run)
    start = time.perf_counter()
    print(f"Run directory: {run.resolve()}", flush=True)
    print("Selecting BPR on validation only...", flush=True)
    selected_bpr = select_bpr(train, validation, config, args.k)
    for row in selected_bpr.checkpoints:
        loss = row["mean_training_loss"]
        display = f"{loss:.6f}" if loss is not None else "N/A"
        print(
            f"BPR epoch {row['epoch']:3}: online loss={display}, "
            f"validation Recall@{args.k}={row['validation_metrics']['mean']['recall']:.6f}",
            flush=True,
        )
    print("Reproducing rating MF validation selection...", flush=True)
    selected_mf = select_mf(train, validation, mf_config, args.k, args.threshold)
    selection = {
        name: {
            "best_epoch": selected.best_epoch,
            "criterion": "validation Recall@K",
            "checkpoints": selected.checkpoints,
        }
        for name, selected in [("bpr", selected_bpr), ("mf", selected_mf)]
    }
    save_json(run / "selection.json", selection)  # Freeze before any test metric is computed.
    print(
        f"Selected epochs: BPR={selected_bpr.best_epoch}, "
        f"MF={selected_mf.best_epoch}; refitting...",
        flush=True,
    )
    refit_bpr = fit_bpr_fixed(train + validation, config, selected_bpr.best_epoch)
    refit_mf = fit_fixed(train + validation, mf_config, selected_mf.best_epoch)
    bpr_test = rank_bpr(refit_bpr, test, args.k, args.threshold)
    mf_test = rank_mf(refit_mf, test, args.k, args.threshold)
    stages, predictions, subgroups = {}, {}, {}
    for stage, history, holdout, results, fitted_models in [
        (
            "validation",
            train,
            validation,
            {"bpr": selected_bpr.validation, "mf": selected_mf.validation},
            {"bpr": selected_bpr.fitted, "mf": selected_mf.fitted},
        ),
        (
            "test",
            train + validation,
            test,
            {"bpr": bpr_test, "mf": mf_test},
            {"bpr": refit_bpr, "mf": refit_mf},
        ),
    ]:
        baseline = evaluate_holdout(history, holdout, args.threshold, args.k)
        for name, result in results.items():
            if result["relevant"] != baseline["relevant"]:
                raise RuntimeError("Model relevance protocols disagree")
            baseline["metrics"][name] = result["metrics"]
            baseline["predictions"][name] = result["predictions"]
        stages[stage] = {key: baseline[key] for key in ["stats", "metrics"]}
        stages[stage]["stats"]["bpr_n_inactive_history_users"] = len(
            fitted_models["bpr"].users
        ) - len(fitted_models["bpr"].active_users)
        predictions[stage] = {key: baseline[key] for key in ["predictions", "relevant", "users"]}
        subgroups[stage] = {}
        for group, cold in [("warm", False), ("cold", True)]:
            answers = {
                u: set(items)
                for u, items in baseline["relevant"].items()
                if baseline["users"][u]["cold_user"] == cold
            }
            subgroups[stage][group] = {
                model: asdict(evaluate_users({u: ranks[u] for u in answers}, answers, args.k))
                for model, ranks in baseline["predictions"].items()
            }
        for model, fitted in fitted_models.items():
            checkpoint = {
                "state_dict": fitted.model.state_dict(),
                "users": fitted.users,
                "items": fitted.items,
                "config": asdict(config if model == "bpr" else mf_config),
                "epochs": selection[model]["best_epoch"],
            }
            if model == "bpr":
                checkpoint["active_users"] = sorted(fitted.active_users)
            with (run / f"{model}_{stage}.pt").open("xb") as stream:
                torch.save(checkpoint, stream)
        print(f"\n{stage}:", flush=True)
        for model, metrics in baseline["metrics"].items():
            print(model, metrics, flush=True)
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
        "bpr_validation_trace": selected_bpr.fitted.trace,
        "refit_trace": {"bpr": refit_bpr.trace, "mf": refit_mf.trace},
        "elapsed_seconds": time.perf_counter() - start,
    }
    save_json(run / "report.json", report)
    print(f"Report: {run / 'report.json'}", flush=True)


if __name__ == "__main__":
    main()
