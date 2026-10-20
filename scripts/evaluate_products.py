"""执行冻结商品热门基线；预测与答案分离，保存可复算来源。"""

import argparse
import json
import platform
import sys
import time
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from uuid import uuid4

from evaluate_movielens import capture_source, save_json

from memrecagent.amazon import AmazonReview
from memrecagent.amazon_preparation import TEST_END_MS, TRAIN_END_MS, VALIDATION_END_MS
from memrecagent.product_evaluation import (
    build_window,
    evaluate_window,
    fit_popularity,
    predict_popularity,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "data/processed/amazon-2023/20261020T044103694215Z-1e20301b"


def load_prepared(path: Path) -> tuple[tuple[AmazonReview, ...], list[str], dict]:
    """先核对处理产物哈希，再读取事件；损坏数据不能产生成功报告。"""
    execution = json.loads((path / "execution.json").read_text())
    hashes = {}
    for name in ["events.jsonl", "candidates.json", "products.json", "report.json"]:
        hashes[name] = sha256((path / name).read_bytes()).hexdigest()
        if hashes[name] != execution["output_sha256"].get(name):
            raise ValueError(f"{name}: SHA256与处理记录不符")
    with (path / "events.jsonl").open() as stream:
        events = tuple(AmazonReview(**json.loads(line)) for line in stream)
    candidates = json.loads((path / "candidates.json").read_text())
    return events, candidates, {"path": str(path.resolve()), "sha256": hashes}


def main() -> None:
    """固定候选与两个窗口，分别拟合热门计数后预测，再读取答案计分。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-root", type=Path, default=ROOT / "artifacts/products_popularity")
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()
    if args.k <= 0:
        parser.error("k必须为正")
    try:
        events, candidates, data = load_prepared(args.data)
    except (ValueError, OSError, KeyError, TypeError) as exc:
        parser.error(str(exc))
    created = datetime.now(UTC)
    run = args.output_root / f"{created.strftime('%Y%m%dT%H%M%S%fZ')}-{uuid4().hex[:8]}"
    run.mkdir(parents=True, exist_ok=False)
    config = {
        "protocol": "p2-products-v1",
        "k": args.k,
        "threshold": 4.0,
        "cutoffs_ms": [TRAIN_END_MS, VALIDATION_END_MS, TEST_END_MS],
        "candidate_ids_sha256": data["sha256"]["candidates.json"],
        "target": "first rating of new parent item in frozen window; repeat seen items separate",
        "future_outside_catalog": "retain in denominator",
        "metadata_features": False,
        "negative_sampling": None,
        "tuning": "none",
    }
    save_json(run / "config.json", config)
    code = capture_source(run)
    stages, predictions, elapsed = {}, {}, {}
    for name, start, end in [
        ("validation", TRAIN_END_MS, VALIDATION_END_MS),
        ("test", VALIDATION_END_MS, TEST_END_MS),
    ]:
        started = time.perf_counter()
        window = build_window(events, start, end)
        counts = fit_popularity(window.history, candidates)
        # 预测只接收历史计数、已见与用户ID，不接收未来标签。
        ranked = predict_popularity(counts, window.seen, window.users, args.k)
        result = evaluate_window(ranked, window, candidates, args.k)
        elapsed[name] = time.perf_counter() - started
        stages[name] = {"stats": result["stats"], "metrics": {"popularity": result["metrics"]}}
        predictions[name] = {
            "predictions": {"popularity": ranked},
            "relevant": {user: sorted(items) for user, items in window.relevant.items()},
        }
        save_json(run / f"popularity_{name}.json", counts)
    save_json(run / "predictions.json", predictions)
    report = {
        "created_at_utc": created.isoformat(),
        "config": config,
        "code": code,
        "data": data,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "backend": "CPU counters",
        },
        "stages": stages,
        "elapsed_seconds": elapsed,
        "limits": [
            "historical review discovery, not purchase prediction",
            "fixed 5000-item resource scope",
            "no long-term memory",
        ],
    }
    save_json(run / "report.json", report)
    print(json.dumps({"run": str(run.resolve()), "stages": stages}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
