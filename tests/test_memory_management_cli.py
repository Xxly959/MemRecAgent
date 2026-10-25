"""实际执行本地管理命令，验证查看、明确纠正和物理删除。"""

import json
import subprocess
import sys
from datetime import timedelta
from pathlib import Path

from test_memory import T0, sample_memory
from test_memory_store import AFTER, file_for

from memrecagent.memory_store import JsonlMemoryStore

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/manage_memories.py"


def run(root, *args):
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--root", str(root), "--user-id", "demo-user", *args],
        text=True,
        capture_output=True,
    )


def test_cli_view_correct_history_and_delete_across_processes(tmp_path):
    root = tmp_path / "store"
    JsonlMemoryStore(root).add(sample_memory())
    view = run(root, "list", "--as-of", AFTER.isoformat())
    assert view.returncode == 0, view.stderr
    assert json.loads(view.stdout)[0]["relation"] == "exclude"
    data = {
        "category": "phone",
        "entity": "小米",
        "expected_head": "m1",
        "relation": "neutral",
        "evidence_ref": "S2:T5",
        "evidence_text": "以后不用排除小米了（合成）",
        "occurred_at": (T0 + timedelta(hours=1)).isoformat(),
        "available_at": (T0 + timedelta(hours=2)).isoformat(),
    }
    source = tmp_path / "correction.json"
    source.write_text(json.dumps(data))
    corrected = run(root, "correct", "--input", str(source))
    assert corrected.returncode == 0, corrected.stderr
    assert json.loads(corrected.stdout)["status"] == "corrected"
    history = run(root, "list", "--history", "--entity", "小米", "--as-of", AFTER.isoformat())
    assert [r["relation"] for r in json.loads(history.stdout)] == ["exclude", "neutral"]
    current = run(root, "list", "--as-of", AFTER.isoformat())
    assert [r["relation"] for r in json.loads(current.stdout)] == ["neutral"]
    deleted = run(
        root, "delete", "--category", "phone", "--entity", "小米", "--at", AFTER.isoformat()
    )
    assert deleted.returncode == 0, deleted.stderr
    assert json.loads(deleted.stdout) == {"deleted_records": 2}
    assert (
        json.loads(run(root, "list", "--as-of", (AFTER + timedelta(days=1)).isoformat()).stdout)
        == []
    )


def test_cli_invalid_input_does_not_echo_evidence_or_modify_store(tmp_path):
    JsonlMemoryStore(tmp_path / "store").add(sample_memory())
    before = file_for(tmp_path / "store").read_bytes()
    source = tmp_path / "bad.json"
    source.write_text('{"evidence_text":"private-synthetic", "unknown":true}')
    result = run(tmp_path / "store", "correct", "--input", str(source))
    assert result.returncode == 2
    assert json.loads(result.stderr) == {"error": "invalid_input_or_store"}
    assert "private-synthetic" not in result.stdout + result.stderr
    assert "Traceback" not in result.stderr
    assert file_for(tmp_path / "store").read_bytes() == before


def test_cli_query_missing_store_does_not_create_files(tmp_path):
    root = tmp_path / "missing"
    result = run(root, "list", "--as-of", AFTER.isoformat())
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == []
    assert not root.exists()


def test_out_of_range_utc_conversion_has_sanitized_error_and_no_write(tmp_path):
    root = tmp_path / "store"
    result = run(
        root,
        "delete",
        "--category",
        "phone",
        "--entity",
        "小米",
        "--at",
        "0001-01-01T00:00:00+01:00",
    )
    assert result.returncode == 2
    assert json.loads(result.stderr) == {"error": "invalid_input_or_store"}
    assert not root.exists()
