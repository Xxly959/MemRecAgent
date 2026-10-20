"""Frozen comparison routes from checkpoint vocabularies, not future labels."""

import json
import subprocess
import sys

import torch


def test_cli_uses_stage_vocabularies_and_preserves_source_and_routing(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    report = {"config": {"k": 1, "test_reused": True}, "selection": {"bpr": {"best_epoch": 70}}}
    stages = {}
    for stage in ["validation", "test"]:
        stages[stage] = {
            "predictions": {
                "bpr": {"warm": ["A"], "newval": ["A"], "empty": ["A"]},
                "popularity": {"warm": ["B"], "newval": ["B"], "empty": ["B"]},
            },
            "relevant": {"warm": ["A"], "newval": ["B"], "empty": []},
        }
        users = {"warm": 0} if stage == "validation" else {"warm": 0, "newval": 1}
        torch.save({"users": users}, source / f"bpr_{stage}.pt")
    (source / "report.json").write_text(json.dumps(report))
    (source / "predictions.json").write_text(json.dumps(stages))
    before = {p.name: p.read_bytes() for p in source.iterdir()}
    command = [
        "rtk",
        "proxy",
        sys.executable,
        "scripts/evaluate_cold_fallback.py",
        "--run",
        str(source),
        "--output-root",
        str(tmp_path / "output"),
    ]
    completed = subprocess.run(command, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    output = next((tmp_path / "output").iterdir())
    r = json.loads((output / "report.json").read_text())
    p = json.loads((output / "predictions.json").read_text())
    assert p["validation"]["predictions"]["bpr_cold_popularity"] == {
        "warm": ["A"],
        "newval": ["B"],
        "empty": ["B"],
    }
    assert p["test"]["predictions"]["bpr_cold_popularity"] == {
        "warm": ["A"],
        "newval": ["A"],
        "empty": ["B"],
    }
    assert r["stages"]["validation"]["metrics"]["bpr_cold_popularity"]["mean"]["recall"] == 1
    assert r["stages"]["test"]["metrics"]["bpr_cold_popularity"]["mean"]["recall"] == 0.5
    assert p["test"]["routes"]["empty"] == "popularity"
    assert r["stages"]["test"]["metrics"]["bpr_cold_popularity"]["n_without_relevant"] == 1
    assert (output / "source_snapshot.zip").is_file()
    assert {p.name: p.read_bytes() for p in source.iterdir()} == before

    # Flip future relevance; strategy selection and all ranked outputs must stay fixed.
    for stage in stages:
        stages[stage]["relevant"] = {"warm": [], "newval": [], "empty": ["A"]}
    (source / "predictions.json").write_text(json.dumps(stages))
    completed = subprocess.run(command, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    other = next(path for path in (tmp_path / "output").iterdir() if path != output)
    q = json.loads((other / "predictions.json").read_text())
    for stage in stages:
        assert q[stage]["routes"] == p[stage]["routes"]
        assert q[stage]["predictions"] == p[stage]["predictions"]
