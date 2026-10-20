"""Run the real CLI on a small timestamped fixture and reload its checkpoints."""

import json
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]


def test_cli_preserves_protocol_and_reproducible_artifacts(tmp_path):
    data = tmp_path / "u.data"
    data.write_text(
        "1\t1\t5\t1\n1\t2\t1\t2\n2\t2\t5\t3\n2\t3\t4\t4\n"
        "1\t3\t5\t86401\n3\t4\t5\t86402\n"
        "1\t4\t5\t172801\n3\t1\t5\t172802\n"
    )
    command = [
        "rtk",
        "proxy",
        sys.executable,
        str(ROOT / "scripts/evaluate_movielens_bpr.py"),
        "--data",
        str(data),
        "--output-root",
        str(tmp_path / "runs"),
        "--train-end",
        "1970-01-02",
        "--validation-end",
        "1970-01-03",
        "--max-epochs",
        "2",
        "--checkpoint-every",
        "1",
        "--factors",
        "2",
        "--batch-size",
        "2",
        "--mf-max-epochs",
        "2",
        "--mf-checkpoint-every",
        "1",
        "--k",
        "2",
    ]
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    run = next((tmp_path / "runs").iterdir())
    report = json.loads((run / "report.json").read_text())
    assert report["split_sizes"] == {"train": 4, "validation": 2, "test": 2}
    assert set(report["stages"]["test"]["metrics"]) == {"popularity", "itemcf", "mf", "bpr"}
    assert report["config"]["test_reused"] is True
    assert report["selection"]["bpr"]["best_epoch"] in [1, 2]
    assert (run / "source_snapshot.zip").is_file()
    val = torch.load(run / "bpr_validation.pt", weights_only=True)
    test = torch.load(run / "bpr_test.pt", weights_only=True)
    assert "3" not in val["users"] and "4" not in val["items"]
    assert "3" in test["users"] and "4" in test["items"]
    assert "active_users" in val and "active_users" in test
    assert test["state_dict"]["item_vectors.weight"].shape == (4, 2)
    assert (run / "selection.json").is_file()
