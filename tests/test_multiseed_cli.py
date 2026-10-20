"""Run multi-seed selection/refit on real tiny data, including future-label isolation."""

import json
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]


def run_cli(tmp_path, data):
    result = subprocess.run(
        [
            "rtk",
            "proxy",
            sys.executable,
            str(ROOT / "scripts/evaluate_movielens_multiseed.py"),
            "--data",
            str(data),
            "--output-root",
            str(tmp_path / "runs"),
            "--train-end",
            "1970-01-02",
            "--validation-end",
            "1970-01-03",
            "--seeds",
            "7",
            "8",
            "--max-epochs",
            "2",
            "--checkpoint-every",
            "1",
            "--factors",
            "2",
            "--batch-size",
            "2",
            "--k",
            "2",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    return sorted((tmp_path / "runs").iterdir())[-1]


def test_seed_configs_replay_and_future_labels_do_not_select_epochs(tmp_path):
    data = tmp_path / "u.data"
    original = (
        "1\t1\t5\t1\n1\t2\t1\t2\n2\t2\t5\t3\n2\t3\t4\t4\n"
        "1\t3\t5\t86401\n3\t4\t5\t86402\n"
        "1\t4\t5\t172801\n3\t1\t5\t172802\n"
    )
    data.write_text(original)
    first, second = run_cli(tmp_path, data), run_cli(tmp_path, data)
    a, b = [json.loads((run / "report.json").read_text()) for run in [first, second]]
    assert a["summary"] == b["summary"]
    assert a["config"]["seeds"] == [7, 8]
    assert a["config"]["std_ddof"] == 1
    for seed in [7, 8]:
        assert (first / f"seed-{seed}/predictions.json").read_bytes() == (
            second / f"seed-{seed}/predictions.json"
        ).read_bytes()
        for model in ["mf", "bpr"]:
            checkpoint = torch.load(first / f"seed-{seed}/{model}_test.pt", weights_only=True)
            assert checkpoint["config"]["seed"] == seed
            assert checkpoint["state_dict"]["item_vectors.weight"].shape == (4, 2)
    selection = (first / "selection.json").read_bytes()
    data.write_text(original.replace("1\t4\t5\t172801", "1\t4\t1\t172801"))
    changed = run_cli(tmp_path, data)
    assert (changed / "selection.json").read_bytes() == selection
    for seed in [7, 8]:
        before = json.loads((first / f"seed-{seed}/predictions.json").read_text())
        after = json.loads((changed / f"seed-{seed}/predictions.json").read_text())
        assert before["test"]["predictions"] == after["test"]["predictions"]
    assert (first / "protocol.json").stat().st_mtime_ns <= (
        first / "selection.json"
    ).stat().st_mtime_ns
    assert (first / "selection.json").stat().st_mtime_ns <= (
        first / "seed-7/mf_test.pt"
    ).stat().st_mtime_ns
