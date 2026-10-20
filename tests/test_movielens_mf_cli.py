"""Exercise MF selection/refit and baseline comparison through the real CLI."""

import json
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]


def test_cli_selects_without_test_labels_and_saves_both_checkpoint_vocabularies(tmp_path):
    data = tmp_path / "u.data"
    original = (
        "1\t101\t5\t883612800\n1\t103\t5\t883699200\n"
        "2\t101\t5\t883785600\n2\t102\t5\t883872000\n"
        "3\t101\t5\t883958400\n3\t103\t5\t889488000\n"
        "3\t102\t5\t892166400\n"
    )
    data.write_text(original)
    output = tmp_path / "runs"
    cmd = [
        sys.executable,
        str(ROOT / "scripts/evaluate_movielens_mf.py"),
        "--data",
        str(data),
        "--output-root",
        str(output),
        "--k",
        "2",
        "--max-epochs",
        "2",
        "--checkpoint-every",
        "1",
        "--factors",
        "2",
    ]
    reports = []
    for change_test in [False, False, True]:
        if change_test:
            data.write_text(original.replace("3\t102\t5", "3\t102\t1"))
        result = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
        run = sorted(output.iterdir())[-1]
        reports.append(json.loads((run / "report.json").read_text()))
        assert (run / "selection.json").is_file()
        for name in ["validation", "test"]:
            saved = torch.load(run / f"mf_{name}.pt", weights_only=True)
            assert "user_vectors.weight" in saved["state_dict"]
            assert saved["users"] and saved["items"]
            assert saved["epochs"] == reports[-1]["selection"]["best_epoch"]
    assert len(list(output.iterdir())) == 3
    assert reports[0]["stages"] == reports[1]["stages"]
    assert reports[0]["selection"] == reports[2]["selection"]
    test = reports[0]["stages"]["test"]
    assert set(test["metrics"]) == {"popularity", "itemcf", "mf"}
    assert test["metrics"]["mf"]["mean"]["recall"] == 1.0
    assert test["metrics"]["popularity"]["mean"]["recall"] == 1.0
    assert reports[2]["stages"]["test"]["metrics"]["mf"]["mean"] is None
