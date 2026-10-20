"""Paired Recall bootstrap on frozen MF/BPR test predictions; no training."""

import argparse
import json
import math
import platform
import sys
from datetime import UTC, datetime
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path
from uuid import uuid4

from evaluate_movielens import capture_source, save_json

from memrecagent.uncertainty import paired_recall_bootstrap

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="Frozen four-model run directory")
    parser.add_argument("--output-root", type=Path, default=ROOT / "artifacts/movielens_bootstrap")
    parser.add_argument("--n-resamples", type=int, default=10_000)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    try:
        source_bytes = {
            name: (args.run / name).read_bytes() for name in ["report.json", "predictions.json"]
        }
        source_report = json.loads(source_bytes["report.json"])
        stage = json.loads(source_bytes["predictions.json"])["test"]
        k = source_report["config"]["k"]
        relevant = {user: set(items) for user, items in stage["relevant"].items()}
        result = paired_recall_bootstrap(
            stage["predictions"]["mf"],
            stage["predictions"]["bpr"],
            relevant,
            k,
            n_resamples=args.n_resamples,
            confidence=args.confidence,
            seed=args.seed,
        )
        # When an original aggregate is present, guard against mixing artifacts.
        if "stages" in source_report:
            for model, key in [("mf", "mean_a"), ("bpr", "mean_b")]:
                original = source_report["stages"]["test"]["metrics"][model]
                if original["n_evaluated"] != result["n_evaluated"] or not math.isclose(
                    original["mean"]["recall"], result[key], rel_tol=0, abs_tol=1e-12
                ):
                    raise ValueError("Frozen report and predictions disagree")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.error(str(exc))

    created = datetime.now(UTC)
    run = args.output_root / (created.strftime("%Y%m%dT%H%M%S%fZ-") + uuid4().hex[:8])
    run.mkdir(parents=True, exist_ok=False)
    code = capture_source(run)
    draws = result.pop("bootstrap_deltas")
    save_json(run / "bootstrap_deltas.json", {"delta_a_minus_b": draws})
    report = {
        "created_at_utc": created.isoformat(),
        "comparison": {"a": "mf", "b": "bpr", "direction": "a minus b"},
        "source": {
            "run": str(args.run.resolve()),
            "sha256": {name: sha256(value).hexdigest() for name, value in source_bytes.items()},
        },
        "result": result,
        "protocol": {
            "stage": "test",
            "selection": "none; frozen models and primary Recall comparison",
            "unit": (
                "eligible user; iid empirical resampling with replacement, same indices per model"
            ),
            "interval": "two-sided percentile, numpy linear quantiles; not BCa",
            "model_retraining": False,
            "test_reused": source_report["config"].get("test_reused", True),
            "limits": (
                "conditional on fixed fits and this cohort; not training-seed or domain uncertainty"
            ),
            "inference": (
                "approximate user-level analysis, not a p-value or probability one model is true"
            ),
            "dependence": (
                "users treated as exchangeable; shared-item/population dependence not modeled"
            ),
        },
        "references": [
            "https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html",
            "https://numpy.org/doc/stable/reference/generated/numpy.quantile.html",
        ],
        "code": code,
        "environment": {
            "python": sys.version,
            "numpy": version("numpy"),
            "platform": platform.platform(),
        },
    }
    save_json(run / "report.json", report)
    low, high = result["interval"]
    print(f"Run directory: {run.resolve()}")
    print(
        f"Eligible users: {result['n_evaluated']}; "
        f"excluded without positives: {result['n_without_relevant']}"
    )
    print(f"MF Recall@{k}: {result['mean_a']:.6f}; BPR: {result['mean_b']:.6f}")
    print(f"MF minus BPR: {result['delta_a_minus_b']:.6f}")
    print(f"{args.confidence:.0%} paired-user percentile interval: [{low:.6f}, {high:.6f}]")
    print("Contains zero:", low <= 0 <= high)
    print("Frozen models; no retraining or tuning. User sampling uncertainty only.")
    print(f"Report: {run / 'report.json'}")


if __name__ == "__main__":
    main()
