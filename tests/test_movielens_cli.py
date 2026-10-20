"""Run the actual offline CLI against a small timestamped data file."""

import hashlib
import json
import runpy
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def fixture_data(path):
    path.write_text(
        "1\t101\t5\t883612800\n1\t103\t5\t883699200\n"
        "2\t101\t5\t883785600\n2\t102\t5\t883872000\n"
        "3\t101\t5\t883958400\n3\t103\t5\t889488000\n"
        "3\t102\t5\t892166400\n"
    )


def test_cli_saves_reproducible_results_in_distinct_run_directories(tmp_path):
    data = tmp_path / "u.data"
    fixture_data(data)
    output = tmp_path / "runs"
    cmd = [
        sys.executable,
        str(ROOT / "scripts/evaluate_movielens.py"),
        "--data",
        str(data),
        "--output-root",
        str(output),
        "--k",
        "2",
    ]
    first = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
    assert first.returncode == 0, first.stderr
    second = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
    assert second.returncode == 0, second.stderr
    runs = sorted(output.iterdir())
    assert len(runs) == 2
    reports = [json.loads((p / "report.json").read_text()) for p in runs]
    assert reports[0]["stages"] == reports[1]["stages"]
    assert (runs[0] / "predictions.json").read_bytes() == (
        runs[1] / "predictions.json"
    ).read_bytes()
    report = reports[0]
    assert report["data"]["sha256"] == hashlib.sha256(data.read_bytes()).hexdigest()
    assert report["split_sizes"] == {"train": 5, "validation": 1, "test": 1}
    assert report["stages"]["test"]["metrics"]["itemcf"]["mean"]["recall"] == 1.0
    with zipfile.ZipFile(runs[0] / "source_snapshot.zip") as archive:
        relative = "src/memrecagent/evaluation.py"
        content = archive.read(relative)
        assert hashlib.sha256(content).hexdigest() == report["code"]["files"][relative]
        assert not any(name.startswith("data/") or ".env" in name for name in archive.namelist())
    config = json.loads((runs[0] / "config.json").read_text())
    assert config["threshold"] == 4.0
    assert config["zero_scores"] == "rank_in_both_models"


def test_cli_invalid_data_fails_without_success_report(tmp_path):
    data = tmp_path / "u.data"
    data.write_text("bad input\n")
    output = tmp_path / "runs"
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/evaluate_movielens.py"),
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
    assert "Invalid rating record" in result.stderr
    assert not list(output.glob("*/report.json"))


def test_source_snapshot_works_with_git_without_local_rtk_wrapper(tmp_path, monkeypatch):
    # 发布仓库的用户只需Git，不应依赖维护者机器上的命令包装工具。
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "git").symlink_to(shutil.which("git"))
    monkeypatch.setenv("PATH", str(bin_dir))
    output = tmp_path / "run"
    output.mkdir()
    capture = runpy.run_path(str(ROOT / "scripts/evaluate_movielens.py"))["capture_source"]
    result = capture(output)
    with zipfile.ZipFile(output / "source_snapshot.zip") as archive:
        source = archive.read("src/memrecagent/evaluation.py")
    assert hashlib.sha256(source).hexdigest() == result["files"]["src/memrecagent/evaluation.py"]
    assert isinstance(result["git_status"], list)
