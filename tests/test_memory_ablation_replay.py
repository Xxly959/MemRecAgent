"""真实冻结包、磁盘偏好、离线CLI及不可覆盖的完整结果。"""

import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

from test_memory_ablation import fitted_ranker_inputs
from test_memory_products import request

from memrecagent.product_bundle import save_bundle


def replay_fixture(root):
    base, _, facts = fitted_ranker_inputs(root / "memory")
    manifest = save_bundle(base, root / "bundle", provenance={"kind": "synthetic-ablation-test"})
    (root / "facts.json").write_text(
        json.dumps(
            {
                "bundle_id": manifest["bundle_id"],
                "facts": [asdict(f) | {"available_at": f.available_at.isoformat()} for f in facts],
            },
            ensure_ascii=False,
        )
    )
    query = request() | {"user_id": "u", "strategy": "bpr", "k": 1}
    query["as_of"] = query["as_of"].isoformat()
    suite = {
        "schema_version": 1,
        "evidence_kind": "structured_oracle",
        "label_source": "author-written synthetic diagnostic; not independent or real-user",
        "cases": [
            {
                "case_id": "warm-1",
                "cluster_id": "u",
                "stratum": "warm",
                "request": query,
                "relevant": ["B", "OUTSIDE"],
                "excluded": ["A"],
            }
        ],
    }
    (root / "suite.json").write_text(json.dumps(suite, ensure_ascii=False))
    return suite


def command(root, *extra):
    return [
        sys.executable,
        str(Path(__file__).resolve().parents[1] / "scripts/evaluate_ranking_ablation.py"),
        "--bundle",
        str(root / "bundle"),
        "--facts",
        str(root / "facts.json"),
        "--memory-root",
        str(root / "memory"),
        "--suite",
        str(root / "suite.json"),
        "--output-dir",
        str(root / "output"),
        *extra,
    ]


def test_cli_default_is_plan_then_run_freezes_and_scores_real_four_groups(tmp_path):
    replay_fixture(tmp_path)
    planned = subprocess.run(command(tmp_path), capture_output=True, timeout=15)
    assert planned.returncode == 0, planned.stderr.decode()
    assert json.loads(planned.stdout)["api_calls"] == 0
    assert not (tmp_path / "output").exists()
    run = subprocess.run(command(tmp_path, "--run-offline"), capture_output=True, timeout=15)
    assert run.returncode == 0, run.stderr.decode()
    report = json.loads((tmp_path / "output/report.json").read_text())
    assert report["summary"]["groups"]["G11"]["recall"] == 0.5
    assert report["summary"]["groups"]["G01"]["recall"] == 0
    assert report["evidence_kind"] == "structured_oracle"
    assert report["n_execution_errors"] == 0
    assert report["inputs_unchanged"] is True
    assert report["api_calls"] == 0
    assert (tmp_path / "output/source/memory_products.py").is_file()
    assert (tmp_path / "output/manifest.json").is_file()
    assert len((tmp_path / "output/cases.jsonl").read_text().splitlines()) == 1
    again = subprocess.run(command(tmp_path, "--run-offline"), capture_output=True, timeout=15)
    assert again.returncode != 0
    assert json.loads((tmp_path / "output/report.json").read_text()) == report


def test_execution_failures_keep_all_cases_in_denominator(tmp_path):
    suite = replay_fixture(tmp_path)
    # 包截止之前的查询必须失败，而非从未来模型取得有用排序。
    suite["cases"][0]["request"]["as_of"] = "2000-01-01T00:00:00+00:00"
    (tmp_path / "suite.json").write_text(json.dumps(suite))
    run = subprocess.run(command(tmp_path, "--run-offline"), capture_output=True, timeout=15)
    assert run.returncode == 1
    report = json.loads((tmp_path / "output/report.json").read_text())
    assert report["summary"]["n_cases"] == 1
    assert report["summary"]["groups"]["G11"]["n_errors"] == 1
    assert report["summary"]["groups"]["G11"]["recall"] == 0


def test_labels_cannot_be_smuggled_into_ranking_request(tmp_path):
    suite = replay_fixture(tmp_path)
    suite["cases"][0]["request"]["relevant"] = ["B"]
    (tmp_path / "suite.json").write_text(json.dumps(suite))
    run = subprocess.run(command(tmp_path, "--run-offline"), capture_output=True, timeout=15)
    assert run.returncode == 2
    assert json.loads(run.stderr)["error"] == "invalid_ablation_input"
    assert not (tmp_path / "output").exists()


def test_same_user_cannot_be_split_into_independent_bootstrap_clusters(tmp_path):
    suite = replay_fixture(tmp_path)
    second = suite["cases"][0] | {"case_id": "another-turn", "cluster_id": "pretend-new-user"}
    suite["cases"].append(second)
    (tmp_path / "suite.json").write_text(json.dumps(suite))
    run = subprocess.run(command(tmp_path, "--run-offline"), capture_output=True, timeout=15)
    assert run.returncode == 2
    assert json.loads(run.stderr)["error"] == "invalid_ablation_input"
    assert not (tmp_path / "output").exists()
