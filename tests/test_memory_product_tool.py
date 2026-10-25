"""可信路径绑定、跨进程真实排名与发布前失效检查。"""

import json
import subprocess
import sys

import pytest
from test_memory import sample_memory
from test_memory_products import AT, engine

from memrecagent.memory_store import JsonlMemoryStore
from memrecagent.memory_use_policy import JsonMemoryUseStore, MemoryUsePolicy
from memrecagent.product_bundle import save_bundle


def fixture(tmp_path):
    ranker = engine(tmp_path / "memory")
    manifest = save_bundle(ranker._service, tmp_path / "bundle", provenance={"kind": "synthetic"})
    facts = {
        "bundle_id": manifest["bundle_id"],
        "facts": [
            {
                "item_id": f.item_id,
                "category": f.category,
                "entity": f.entity,
                "evidence_ref": f.evidence_ref,
                "available_at": f.available_at.isoformat(),
            }
            for f in ranker._facts
        ],
    }
    request = {
        "user_id": "demo-user",
        "session_id": "S2",
        "beneficiary_id": "self",
        "category": "phone",
        "as_of": AT.isoformat(),
        "strategy": "popularity",
        "k": 3,
    }
    for name, value in (("facts", facts), ("request", request)):
        (tmp_path / f"{name}.json").write_text(json.dumps(value, ensure_ascii=False))
    return tmp_path


def run_tool(path, *extra):
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "memrecagent.memory_product_tool",
            "--bundle",
            str(path / "bundle"),
            "--memory-root",
            str(path / "memory"),
            "--facts",
            str(path / "facts.json"),
            "--request",
            str(path / "request.json"),
            *extra,
        ],
        capture_output=True,
        timeout=15,
    )


def test_real_subprocess_reads_disk_memory_and_token_without_writing(tmp_path):
    fixture(tmp_path)
    JsonlMemoryStore(tmp_path / "memory").add(sample_memory(entity="品牌A"))
    before = {
        str(p.relative_to(tmp_path)): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()
    }
    process = run_tool(tmp_path)
    assert process.returncode == 0 and process.stderr == b""
    result = json.loads(process.stdout)
    assert [p["item_id"] for p in result["items"]] == ["B", "C"]
    checked = run_tool(tmp_path, "--check-token", result["delivery_token"])
    assert checked.returncode == 0 and json.loads(checked.stdout) == {"valid": True}
    assert {
        str(p.relative_to(tmp_path)): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()
    } == before


@pytest.mark.parametrize("change", ["delete", "policy-aba", "facts", "request", "bundle"])
def test_snapshot_cannot_be_delivered_after_relevant_state_changes(tmp_path, change):
    fixture(tmp_path)
    store = JsonlMemoryStore(tmp_path / "memory")
    store.add(sample_memory(entity="品牌A"))
    token = json.loads(run_tool(tmp_path).stdout)["delivery_token"]
    if change == "delete":
        store.delete_entity("demo-user", category="phone", entity="品牌A", deleted_at=AT)
    elif change == "policy-aba":
        policy = JsonMemoryUseStore(store.root / "use-policy")
        policy.set("demo-user", MemoryUsePolicy(personalisation=False))
        policy.set("demo-user", MemoryUsePolicy())
    else:
        path = (
            tmp_path
            / {"facts": "facts.json", "request": "request.json", "bundle": "bundle/service.json"}[
                change
            ]
        )
        path.write_bytes(path.read_bytes() + b" ")
    process = run_tool(tmp_path, "--check-token", token)
    assert process.returncode == 2 and process.stdout == b""
    assert process.stderr == b"recommendation_unavailable\n"


@pytest.mark.parametrize(
    "raw",
    [
        '{"user_id":"x","user_id":"y"}',
        '{"k":NaN}',
        '{"k":1e999}',
        "[" * 2000 + "]" * 2000,
        "x" * 65537,
    ],
)
def test_malformed_request_has_no_raw_echo_or_traceback(tmp_path, raw):
    fixture(tmp_path)
    (tmp_path / "request.json").write_text(raw)
    process = run_tool(tmp_path)
    assert process.returncode == 2 and process.stdout == b""
    assert process.stderr == b"recommendation_unavailable\n"


def test_unexpected_request_fields_cannot_override_bound_paths(tmp_path):
    fixture(tmp_path)
    path = tmp_path / "request.json"
    value = json.loads(path.read_text()) | {"bundle": "/private/unknown"}
    path.write_text(json.dumps(value))
    process = run_tool(tmp_path)
    assert process.returncode == 2 and process.stdout == b""


@pytest.mark.parametrize("restriction", ["excluded_entity", "requested_entity"])
def test_large_legal_pool_keeps_all_candidates_but_bounds_filter_trace(tmp_path, restriction):
    from dataclasses import replace

    from test_product_service import service

    from memrecagent.mf import MatrixFactorization

    base = service()
    base.counts = {f"P{i:04d}": 1000 - i for i in range(1000)}
    base.seen = {}
    for name, fitted in base.models.items():
        base.models[name] = replace(
            fitted,
            model=MatrixFactorization(1, 1000, factors=2, seed=42),
            items={k: i for i, k in enumerate(base.counts)},
            seen={},
        )
    manifest = save_bundle(base, tmp_path / "bundle", provenance={"kind": "synthetic-large"})
    facts = {
        "bundle_id": manifest["bundle_id"],
        "facts": [
            {
                "item_id": item,
                "category": "phone",
                "entity": "品牌C" if i == 999 else "品牌A",
                "evidence_ref": "synthetic:large",
                "available_at": "2026-10-01T00:00:00+00:00",
            }
            for i, item in enumerate(base.counts)
        ],
    }
    request = {
        "user_id": "demo-user",
        "session_id": "S2",
        "beneficiary_id": "self",
        "category": "phone",
        "as_of": AT.isoformat(),
        "strategy": "popularity",
        "k": 1,
    }
    if restriction == "excluded_entity":
        JsonlMemoryStore(tmp_path / "memory").add(sample_memory(entity="品牌A"))
    else:
        request["requested_entity"] = "品牌C"
    (tmp_path / "facts.json").write_text(json.dumps(facts, ensure_ascii=False))
    (tmp_path / "request.json").write_text(json.dumps(request, ensure_ascii=False))
    process = run_tool(tmp_path)
    assert process.returncode == 0, process.stderr
    assert len(process.stdout) <= 65536
    result = json.loads(process.stdout)
    assert [p["item_id"] for p in result["items"]] == ["P0999"]
    assert result["n_candidates"] == 1000 and result["n_filtered"] == 999
    assert result["filter_counts"] == {restriction: 999}
    assert result["filtered_truncated"] is True and len(result["filtered"]) == 20
