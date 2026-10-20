"""Repeat frozen MF/BPR settings across prespecified seeds; never select on test."""

import argparse
import platform
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path
from uuid import uuid4

import torch
from evaluate_movielens import capture_source, save_json, utc_boundary

from memrecagent.bpr_evaluation import BPRConfig, fit_bpr_fixed, rank_bpr, select_bpr
from memrecagent.metrics import evaluate_users
from memrecagent.mf_evaluation import MFConfig, fit_fixed, rank_mf, select_mf
from memrecagent.movielens import load_ratings, temporal_split
from memrecagent.multiseed import summarize_runs

ROOT = Path(__file__).resolve().parents[1]


def checkpoint(path, fitted, config, epochs):
    torch.save(
        {
            "state_dict": fitted.model.state_dict(),
            "users": fitted.users,
            "items": fitted.items,
            "config": asdict(config),
            "epochs": epochs,
            "active_users": sorted(getattr(fitted, "active_users", [])),
        },
        path,
    )


def validate_rankings(result, history, k):
    """Guard each model's candidate pool independently of its scorer."""
    catalog = {row.item_id for row in history}
    seen = {}
    for row in history:
        seen.setdefault(row.user_id, set()).add(row.item_id)
    for user, items in result["predictions"].items():
        if len(items) > k or len(items) != len(set(items)):
            raise ValueError("Invalid Top-K ranking")
        if not set(items) <= catalog or set(items) & seen.get(user, set()):
            raise ValueError("Candidate pool or seen-item filtering mismatch")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data/raw/ml-100k/u.data")
    parser.add_argument("--output-root", type=Path, default=ROOT / "artifacts/movielens_multiseed")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    parser.add_argument("--train-end", default="1998-03-01")
    parser.add_argument("--validation-end", default="1998-04-01")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--threshold", type=float, default=4)
    parser.add_argument("--factors", type=int, default=16)
    parser.add_argument("--max-epochs", type=int, default=80)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.5)
    parser.add_argument("--l2", type=float, default=0.001)
    args = parser.parse_args()
    try:
        if len(args.seeds) < 2 or len(set(args.seeds)) != len(args.seeds) or min(args.seeds) < 0:
            raise ValueError("At least two distinct nonnegative seeds required")
        if args.k <= 0:
            raise ValueError("Positive K required")
        configs = {}
        for seed in args.seeds:
            common = dict(
                factors=args.factors,
                max_epochs=args.max_epochs,
                checkpoint_every=args.checkpoint_every,
                batch_size=args.batch_size,
                lr=args.lr,
                l2=args.l2,
                seed=seed,
            )
            configs[seed] = {
                "mf": MFConfig(**common),
                "bpr": BPRConfig(**common, threshold=args.threshold),
            }
        records = load_ratings(args.data)
        train, validation, test = temporal_split(
            records, utc_boundary(args.train_end), utc_boundary(args.validation_end)
        )
        if not train or not validation or not test:
            raise ValueError("Nonempty train, validation and test required")
        if not any(row.rating >= args.threshold for row in validation):
            raise ValueError("Validation must contain positive targets")
    except (OSError, ValueError) as exc:
        parser.error(str(exc))

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    now = datetime.now(UTC)
    run = args.output_root / (now.strftime("%Y%m%dT%H%M%S%fZ-") + uuid4().hex[:8])
    run.mkdir(parents=True, exist_ok=False)
    protocol = {
        "seeds": args.seeds,
        "models": {
            str(seed): {m: asdict(c) for m, c in values.items()} for seed, values in configs.items()
        },
        "train_end": args.train_end,
        "validation_end": args.validation_end,
        "k": args.k,
        "threshold": args.threshold,
        "selection": "per model per seed: highest validation macro Recall, ties earlier epoch",
        "refit": "fresh train+validation fit for selected epochs; all selections saved before test",
        "catalog": "historical items only; all scores; filter all seen; string-ID ties",
        "cold": "uniform zero; BPR inactive history users also zero; no popularity fallback",
        "cohort": (
            "all holdout users; exclude/count no-positive users; out-of-catalog positives retained"
        ),
        "aggregation": "all prespecified seeds; mean and sample std, no best-seed selection",
        "std_ddof": 1,
        "test_reused": True,
        "scope": ("fixed-config seed robustness; no hyperparameter expansion or new blind test"),
        "data_sha256": sha256(args.data.read_bytes()).hexdigest(),
    }
    save_json(run / "protocol.json", protocol)
    code = capture_source(run)
    print(f"Run directory: {run.resolve()}", flush=True)
    selected, selections = {}, {}
    # Complete every validation choice before any test evaluation is performed.
    for seed, cfg in configs.items():
        print(f"Selecting seed {seed} on validation only...", flush=True)
        selected[seed] = {
            "mf": select_mf(train, validation, cfg["mf"], args.k, args.threshold),
            "bpr": select_bpr(train, validation, cfg["bpr"], args.k),
        }
        selections[str(seed)] = {
            model: {"best_epoch": value.best_epoch, "checkpoints": value.checkpoints}
            for model, value in selected[seed].items()
        }
    save_json(run / "selection.json", selections)

    rows = []
    history = train + validation
    for seed, cfg in configs.items():
        folder = run / f"seed-{seed}"
        folder.mkdir()
        row = {"seed": seed, "validation": {}, "test": {}}
        predictions = {stage: {"predictions": {}} for stage in ["validation", "test"]}
        print(f"Refitting fixed epochs for seed {seed}...", flush=True)
        for model, refit_fn, rank_fn in [
            ("mf", fit_fixed, rank_mf),
            ("bpr", fit_bpr_fixed, rank_bpr),
        ]:
            choice = selected[seed][model]
            final = refit_fn(history, cfg[model], choice.best_epoch)
            checkpoint(
                folder / f"{model}_validation.pt", choice.fitted, cfg[model], choice.best_epoch
            )
            checkpoint(folder / f"{model}_test.pt", final, cfg[model], choice.best_epoch)
            for stage, result, past in [
                ("validation", choice.validation, train),
                ("test", rank_fn(final, test, args.k, args.threshold), history),
            ]:
                validate_rankings(result, past, args.k)
                if (
                    "relevant" in predictions[stage]
                    and predictions[stage]["relevant"] != result["relevant"]
                ):
                    raise ValueError("Models must use identical relevant sets")
                predictions[stage]["relevant"] = result["relevant"]
                predictions[stage]["predictions"][model] = result["predictions"]
                row[stage][model] = result["metrics"]
                # Same cold/warm cohorts for both models and every seed.
                users = {r.user_id for r in past}
                groups = {}
                for group, cold in [("cold", True), ("warm", False)]:
                    answers = {
                        u: set(v) for u, v in result["relevant"].items() if (u not in users) == cold
                    }
                    groups[group] = asdict(
                        evaluate_users(
                            {u: result["predictions"][u] for u in answers}, answers, args.k
                        )
                    )
                row.setdefault("subgroups", {}).setdefault(stage, {})[model] = groups
            save_json(folder / f"{model}_refit_trace.json", {"trace": final.trace})
        save_json(folder / "predictions.json", predictions)
        save_json(folder / "metrics.json", row)
        rows.append(row)

    summary = summarize_runs(rows)
    report = {
        "created_at_utc": now.isoformat(),
        "config": protocol,
        "code": code,
        "split_sizes": {"train": len(train), "validation": len(validation), "test": len(test)},
        "runs": rows,
        "summary": summary,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "packages": {name: version(name) for name in ["torch", "numpy", "pytest", "ruff"]},
            "backend": "CPU float32, single thread, deterministic algorithms",
        },
    }
    save_json(run / "report.json", report)
    for model in ["mf", "bpr", "mf_minus_bpr"]:
        print(model, summary["test"][model], flush=True)
    print(f"Report: {run / 'report.json'}", flush=True)


if __name__ == "__main__":
    main()
