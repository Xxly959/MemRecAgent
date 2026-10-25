"""四组开关必须贯通实际Python工具，且不绕过删除、许可和发布快照。"""

import json
from datetime import timedelta

import pytest
from test_memory import T0, sample_memory
from test_memory_product_tool import fixture, run_tool
from test_memory_products import engine, request

from memrecagent.memory_policy import ContextPreference
from memrecagent.memory_store import JsonlMemoryStore
from memrecagent.memory_use_policy import JsonMemoryUseStore, MemoryUsePolicy


def test_long_term_off_preserves_same_source_deletion_and_current_preferences(tmp_path):
    ranker = engine(tmp_path)
    store = JsonlMemoryStore(tmp_path)
    store.add(sample_memory(entity="品牌A"))
    store.add(sample_memory(memory_id="old-c", entity="品牌C", relation="prefer"))
    store.delete_entity(
        "demo-user", category="phone", entity="品牌C", deleted_at=T0 + timedelta(hours=2)
    )
    contexts = (
        ContextPreference(
            "demo-user", "S2", "self", "phone", "品牌C", "exclude", "old", "合成旧证据", T0
        ),
        ContextPreference(
            "demo-user", "S2", "self", "phone", "品牌B", "prefer", "now", "合成本次偏好", T0
        ),
    )
    result = ranker.recommend(**request(context_preferences=contexts), use_long_term_memory=False)
    assert [i["item_id"] for i in result["items"]] == ["B", "A", "C"]
    assert result["use_long_term_memory"] is False
    assert all(s["scope"] == "current" for i in result["items"] for s in i["signals"])
    assert store.active_memories("demo-user", as_of=T0 + timedelta(days=1))[0].relation == "exclude"


def test_off_does_not_query_active_preference_records(tmp_path, monkeypatch):
    ranker = engine(tmp_path)

    def forbidden(*args, **kwargs):
        raise AssertionError("memory-off不得查询长期偏好")

    monkeypatch.setattr(JsonlMemoryStore, "active_memories", forbidden)
    result = ranker.recommend(**request(), use_long_term_memory=False)
    assert [i["item_id"] for i in result["items"]] == ["A", "B", "C"]


@pytest.mark.parametrize("flag", [None, 0, 1, "false", [], {}])
def test_memory_flag_is_explicit_boolean_not_truthy_conversion(tmp_path, flag):
    with pytest.raises(ValueError, match="布尔"):
        engine(tmp_path).recommend(**request(), use_long_term_memory=flag)


def test_tool_honours_off_and_rank_and_binds_them_to_delivery_token(tmp_path):
    fixture(tmp_path)
    store = JsonlMemoryStore(tmp_path / "memory")
    store.add(sample_memory(entity="品牌A"))
    path = tmp_path / "request.json"
    query = json.loads(path.read_text()) | {"use_long_term_memory": False, "normalisation": "rank"}
    path.write_text(json.dumps(query))
    off = run_tool(tmp_path)
    assert off.returncode == 0, off.stderr
    result = json.loads(off.stdout)
    assert [i["item_id"] for i in result["items"]] == ["A", "B", "C"]
    assert [i["normalised_base_score"] for i in result["items"]] == [1, 0.5, 0]
    assert result["use_long_term_memory"] is False
    assert run_tool(tmp_path, "--check-token", result["delivery_token"]).returncode == 0
    path.write_text(json.dumps(query | {"use_long_term_memory": True}))
    stale = run_tool(tmp_path, "--check-token", result["delivery_token"])
    assert stale.returncode == 2 and stale.stdout == b""
    on = json.loads(run_tool(tmp_path).stdout)
    assert [i["item_id"] for i in on["items"]] == ["B", "C"]


def test_memory_on_cannot_override_existing_personalisation_denial(tmp_path):
    ranker = engine(tmp_path)
    JsonlMemoryStore(tmp_path).add(sample_memory(entity="品牌A"))
    JsonMemoryUseStore(tmp_path / "use-policy").set(
        "demo-user", MemoryUsePolicy(personalisation=False)
    )
    result = ranker.recommend(**request(), use_long_term_memory=True)
    assert [i["item_id"] for i in result["items"]] == ["A", "B", "C"]
    assert not any(i["signals"] for i in result["items"])
