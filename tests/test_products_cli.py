"""完整商品CLI应保存可复算证据，未来标签变化不能改变推荐。"""

import hashlib
import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

from memrecagent.amazon import AmazonReview

ROOT = Path(__file__).resolve().parents[1]
T, V = 1640995200000, 1656633600000


def prepared_fixture(path, test_rating=5):
    path.mkdir(exist_ok=True)
    events = [
        AmazonReview("u", "A", "Av", 5, T - 2, "synthetic", 1),
        AmazonReview("v", "B", "Bv", 5, T - 1, "synthetic", 2),
        AmazonReview("v", "C", "Cv", 1, T - 1, "synthetic", 3),
        AmazonReview("u", "B", "Bv", 5, T + 1, "synthetic", 4),
        AmazonReview("cold", "C", "Cv", 5, T + 2, "synthetic", 5),
        AmazonReview("u", "C", "Cv", test_rating, V, "synthetic", 6),
        AmazonReview("cold", "A", "Av", test_rating, V + 1, "synthetic", 7),
        AmazonReview("new", "OUTSIDE", "Ov", test_rating, V + 2, "synthetic", 8),
    ]
    (path / "events.jsonl").write_text("".join(json.dumps(asdict(e)) + "\n" for e in events))
    for name, data in [
        ("candidates.json", ["A", "B", "C"]),
        ("products.json", {}),
        ("report.json", {"scope": "synthetic"}),
    ]:
        (path / name).write_text(json.dumps(data))
    (path / "execution.json").write_text(
        json.dumps(
            {
                "output_sha256": {
                    name: hashlib.sha256((path / name).read_bytes()).hexdigest()
                    for name in ["events.jsonl", "candidates.json", "products.json", "report.json"]
                }
            }
        )
    )


def test_real_product_cli_keeps_unreachable_targets_and_replays_predictions(tmp_path):
    data, output = tmp_path / "data", tmp_path / "runs"
    prepared_fixture(data)
    cmd = [
        sys.executable,
        str(ROOT / "scripts/evaluate_products.py"),
        "--data",
        str(data),
        "--output-root",
        str(output),
        "--k",
        "2",
    ]
    first = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    assert first.returncode == 0, first.stderr
    first_run = next(output.iterdir())
    report = json.loads((first_run / "report.json").read_text())
    assert report["stages"]["test"]["metrics"]["popularity"]["mean"]["recall"] == 2 / 3
    assert report["stages"]["test"]["stats"]["n_positive_targets_outside_catalog"] == 1
    prepared_fixture(data, test_rating=1)
    second = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    assert second.returncode == 0, second.stderr
    second_run = next(p for p in output.iterdir() if p != first_run)
    a = json.loads((first_run / "predictions.json").read_text())
    b = json.loads((second_run / "predictions.json").read_text())
    assert a["test"]["predictions"] == b["test"]["predictions"]
    assert b["test"]["relevant"] == {"cold": [], "new": [], "u": []}
    assert (first_run / "source_snapshot.zip").exists()


def test_corrupt_processed_input_is_rejected_before_success_report(tmp_path):
    data, output = tmp_path / "data", tmp_path / "runs"
    prepared_fixture(data)
    with (data / "events.jsonl").open("a") as stream:
        stream.write("corruption\n")
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/evaluate_products.py"),
            "--data",
            str(data),
            "--output-root",
            str(output),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "SHA256" in result.stderr
    assert not list(output.glob("*/report.json"))
