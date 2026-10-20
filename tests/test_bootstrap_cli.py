"""Exercise frozen-result comparison without training or changing the source run."""

import json
import subprocess
import sys


def test_cli_saves_new_report_and_preserves_frozen_input(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    report = {"config": {"k": 1, "threshold": 4, "test_reused": True}}
    predictions = {
        "test": {
            "predictions": {"mf": {"u": ["A"], "v": []}, "bpr": {"u": [], "v": []}},
            "relevant": {"u": ["A"], "v": ["B"]},
        }
    }
    (source / "report.json").write_text(json.dumps(report))
    (source / "predictions.json").write_text(json.dumps(predictions))
    before = {p.name: p.read_bytes() for p in source.iterdir()}
    command = [
        "rtk",
        "proxy",
        sys.executable,
        "scripts/bootstrap_movielens.py",
        "--run",
        str(source),
        "--output-root",
        str(tmp_path / "output"),
        "--n-resamples",
        "100",
    ]
    completed = subprocess.run(command, text=True, capture_output=True)
    assert completed.returncode == 0, completed.stderr
    output = next((tmp_path / "output").iterdir())
    result = json.loads((output / "report.json").read_text())
    assert result["comparison"] == {"a": "mf", "b": "bpr", "direction": "a minus b"}
    assert result["result"]["delta_a_minus_b"] == 0.5
    assert result["result"]["n_evaluated"] == 2
    draws = json.loads((output / "bootstrap_deltas.json").read_text())
    assert len(draws["delta_a_minus_b"]) == 100
    assert (output / "source_snapshot.zip").is_file()
    assert {p.name: p.read_bytes() for p in source.iterdir()} == before
