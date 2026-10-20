"""真实执行小型训练入口，检查未来标签独立性及检查点可追溯。"""

import json
import subprocess
import sys
from hashlib import sha256
from pathlib import Path

import torch
from test_products_cli import prepared_fixture

ROOT = Path(__file__).resolve().parents[1]


def test_future_labels_do_not_change_selection_predictions_or_fitted_weights(tmp_path):
    data, output = tmp_path / "data", tmp_path / "runs"
    prepared_fixture(data)
    command = [
        sys.executable,
        str(ROOT / "scripts/train_products.py"),
        "--data",
        str(data),
        "--output-root",
        str(output),
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
    ]
    first = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    assert first.returncode == 0, first.stderr
    first_run = next(output.iterdir())
    a = json.loads((first_run / "selection.json").read_text())
    assert set(a) == {"mf", "bpr"}
    assert all(row["best_epoch"] in [1, 2] for row in a.values())
    prepared_fixture(data, test_rating=1)
    second = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    assert second.returncode == 0, second.stderr
    second_run = next(p for p in output.iterdir() if p != first_run)
    b = json.loads((second_run / "selection.json").read_text())
    assert a == b
    left = json.loads((first_run / "predictions.json").read_text())
    right = json.loads((second_run / "predictions.json").read_text())
    assert left["test"]["predictions"] == right["test"]["predictions"]
    assert set(left["test"]["predictions"]) == {"popularity", "mf", "bpr"}
    for model in ["mf", "bpr"]:
        for stage in ["validation", "test"]:
            x = torch.load(first_run / f"{model}_{stage}.pt", weights_only=True)
            y = torch.load(second_run / f"{model}_{stage}.pt", weights_only=True)
            assert x["users"] == y["users"] and x["items"] == y["items"]
            assert all(
                torch.equal(value, y["state_dict"][key]) for key, value in x["state_dict"].items()
            )
    report = json.loads((first_run / "report.json").read_text())
    assert set(report["checkpoint_sha256"]) == {
        "mf_validation.pt",
        "mf_test.pt",
        "bpr_validation.pt",
        "bpr_test.pt",
    }
    assert report["config"]["bpr"]["materialize_pools"] is False
    assert report["stages"]["test"]["stats"]["n_positive_targets_outside_catalog"] == 1


def test_selection_audit_includes_users_with_only_seen_repeat_events(tmp_path):
    data, output = tmp_path / "data", tmp_path / "runs"
    prepared_fixture(data)
    # v在训练中已评价B；验证只重复评价B，属于窗口用户但没有新商品答案。
    events_path = data / "events.jsonl"
    with events_path.open("a") as stream:
        stream.write(
            json.dumps(
                {
                    "user_id": "v",
                    "item_id": "B",
                    "variant_id": "Bv",
                    "rating": 5,
                    "timestamp_ms": 1640995200003,
                    "source": "synthetic",
                    "line_number": 9,
                }
            )
            + "\n"
        )
    execution_path = data / "execution.json"
    execution = json.loads(execution_path.read_text())
    execution["output_sha256"]["events.jsonl"] = sha256(events_path.read_bytes()).hexdigest()
    execution_path.write_text(json.dumps(execution))
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/train_products.py"),
            "--data",
            str(data),
            "--output-root",
            str(output),
            "--max-epochs",
            "1",
            "--checkpoint-every",
            "1",
            "--factors",
            "2",
            "--k",
            "2",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    run = next(output.iterdir())
    selection = json.loads((run / "selection.json").read_text())
    report = json.loads((run / "report.json").read_text())
    for name in ["mf", "bpr"]:
        checkpoint = selection[name]["checkpoints"][0]["validation_metrics"]
        assert checkpoint["n_users"] == 3
        assert checkpoint["n_without_relevant"] == 1
        assert checkpoint["n_evaluated"] == 2
        assert checkpoint == report["stages"]["validation"]["metrics"][name]
        assert selection[name]["cohort"] == {
            "n_window_users": 3,
            "n_target_users": 2,
            "n_repeat_only_users": 1,
        }
