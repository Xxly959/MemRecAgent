"""Run the frozen MovieLens temporal baseline experiment; no network access."""

import argparse
import json
import platform
import shutil
import subprocess
import sys
import time
import zipfile
from datetime import UTC, datetime
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path
from uuid import uuid4

from memrecagent.evaluation import evaluate_holdout
from memrecagent.movielens import load_ratings, temporal_split

ROOT = Path(__file__).resolve().parents[1]


def utc_boundary(value: str) -> int:
    """Parse a calendar date as midnight UTC, independent of machine timezone."""
    return int(datetime.strptime(value, "%Y-%m-%d").replace(tzinfo=UTC).timestamp())


def save_json(path: Path, value: dict) -> None:
    """Write each artifact once; never overwrite another run's evidence."""
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def capture_source(run: Path) -> dict:
    """Snapshot project code/config only, excluding data, environments and secrets."""
    paths = sorted(
        [
            p
            for folder in ["src", "scripts", "tests", "examples"]
            for p in (ROOT / folder).rglob("*.py")
        ]
        + [ROOT / name for name in ["pyproject.toml", "uv.lock", ".python-version"]]
    )
    files = {}
    fingerprint = sha256()
    with zipfile.ZipFile(run / "source_snapshot.zip", "x", zipfile.ZIP_DEFLATED) as archive:
        for path in paths:
            relative = path.relative_to(ROOT).as_posix()
            content = path.read_bytes()
            files[relative] = sha256(content).hexdigest()
            fingerprint.update(relative.encode() + b"\0" + content + b"\0")
            archive.writestr(relative, content)
    # RTK只属于本地开发约定；普通Git环境也能完整保存可复现信息。
    git = ["rtk", "proxy", "git"] if shutil.which("rtk") else ["git"]
    revision = subprocess.run(git + ["rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True)
    status = subprocess.run(
        git + ["status", "--porcelain"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return {
        "sha256": fingerprint.hexdigest(),
        "files": files,
        "git_revision": revision.stdout.strip() if revision.returncode == 0 else None,
        "git_status": status.stdout.splitlines(),
        "dirty": bool(status.stdout.strip()),
    }


def main() -> None:
    """Evaluate validation first, refit fixed config, then evaluate final test."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data/raw/ml-100k/u.data")
    parser.add_argument("--output-root", type=Path, default=ROOT / "artifacts/movielens")
    parser.add_argument("--train-end", default="1998-03-01")
    parser.add_argument("--validation-end", default="1998-04-01")
    parser.add_argument("--threshold", type=float, default=4.0)
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()
    if not 1 <= args.threshold <= 5 or args.k <= 0:
        parser.error("threshold must be in [1,5] and k must be positive")
    try:
        train_end, validation_end = utc_boundary(args.train_end), utc_boundary(args.validation_end)
        records = load_ratings(args.data)
        train, validation, test = temporal_split(records, train_end, validation_end)
        if not train or not validation or not test:
            raise ValueError("Each of train, validation, and test must contain ratings")
    except (ValueError, OSError) as exc:
        parser.error(str(exc))

    created = datetime.now(UTC)
    run = args.output_root / (created.strftime("%Y%m%dT%H%M%S%fZ-") + uuid4().hex[:8])
    run.mkdir(parents=True, exist_ok=False)
    config = {
        "train_end_utc": args.train_end,
        "validation_end_utc": args.validation_end,
        "train_end_unix": train_end,
        "validation_end_unix": validation_end,
        "boundaries": "history < boundary; holdout >= boundary",
        "k": args.k,
        "threshold": args.threshold,
        "catalog": "items observed in fitting history only",
        "filter": "all user-item ratings in fitting history, including low ratings",
        "ties": "string item ID ascending",
        "zero_scores": "rank_in_both_models",
        "relevance": "all holdout ratings >= threshold, including outside catalog",
        "cohort": "all users with holdout ratings; skip no-positive users with counts",
        "aggregation": "equal user weight; precision divides by requested K",
        "test_fit": "train+validation; config fixed before test; no within-window updates",
        "cold_user_itemcf": "zero vector, string-ID tie; no fallback",
        "neighbor_limit": None,
        "negative_sampling": None,
        "random_seed": None,
        "hyperparameter_search": "none; single prespecified config",
        "timestamp_assumption": "rating timestamp is availability proxy; no ingestion times",
    }
    save_json(run / "config.json", config)
    code = capture_source(run)
    environment = {
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": {name: version(name) for name in ["numpy", "torch", "pytest", "ruff"]},
        "backend": "NumPy CPU float32; no MPS or gradient training",
    }
    stages, predictions, timings = {}, {}, {}
    for name, history, holdout in [
        ("validation", train, validation),
        ("test", train + validation, test),
    ]:
        start = time.perf_counter()
        result = evaluate_holdout(history, holdout, threshold=args.threshold, k=args.k)
        timings[name] = time.perf_counter() - start
        stages[name] = {key: result[key] for key in ["stats", "metrics"]}
        predictions[name] = {key: result[key] for key in ["predictions", "relevant", "users"]}
    save_json(run / "predictions.json", predictions)
    report = {
        "created_at_utc": created.isoformat(),
        "config": config,
        "code": code,
        "environment": environment,
        "data": {
            "path": str(args.data.resolve()),
            "sha256": sha256(args.data.read_bytes()).hexdigest(),
            "n_ratings": len(records),
            "n_users": len({r.user_id for r in records}),
            "n_items": len({r.item_id for r in records}),
            "min_timestamp": min(r.timestamp for r in records),
            "max_timestamp": max(r.timestamp for r in records),
        },
        "split_sizes": {"train": len(train), "validation": len(validation), "test": len(test)},
        "stages": stages,
        "elapsed_seconds": timings,
    }
    save_json(run / "report.json", report)
    print(f"Run directory: {run.resolve()}")
    print("Split sizes:", report["split_sizes"])
    for stage, result in stages.items():
        print(f"\n{stage}:")
        for model, scores in result["metrics"].items():
            print(model, scores)


if __name__ == "__main__":
    main()
