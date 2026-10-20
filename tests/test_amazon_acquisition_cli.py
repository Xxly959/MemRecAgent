"""验证命令默认不取数、预算不足时不产生文件或网络请求。"""

import json
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/acquire_amazon.py"


def test_default_command_only_reports_pinned_plan_without_creating_destination(tmp_path):
    target = tmp_path / "not-created"
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--destination", str(target)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    plan = json.loads(result.stdout)
    assert plan["revision"] == "2b6d039ed471f2ba5fd2acb718bf33b0a7e5598e"
    assert plan["total_bytes"] == 2189723077
    assert plan["fetch"] is False
    assert not target.exists()


def test_insufficient_total_budget_rejects_fetch_before_creating_destination(tmp_path):
    target = tmp_path / "not-created"
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--fetch", "--max-bytes", "1", "--destination", str(target)],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "预算" in result.stderr
    assert not target.exists()
