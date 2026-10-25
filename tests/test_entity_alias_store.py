"""别名命名空间绑定；存储原文不改，逻辑键必须修订/删除一致。"""

import json
from dataclasses import replace
from datetime import timedelta

import pytest
from test_entity_aliases import document
from test_memory import T0, sample_memory

from memrecagent.entity_aliases import install_entity_aliases
from memrecagent.memory_store import JsonlMemoryStore

AT = T0 + timedelta(days=1)


def configured(root):
    table = install_entity_aliases(root, document())
    return JsonlMemoryStore(root), table


def test_alias_storage_uses_canonical_key_and_preserves_original_evidence(tmp_path):
    store, table = configured(tmp_path)
    store.add(sample_memory(entity="BrandA", evidence_text="合成：以后别推荐BrandA"))
    loaded = store.list_memories("demo-user", as_of=AT)[0]
    assert loaded.entity == "品牌A" and loaded.evidence_text == "合成：以后别推荐BrandA"
    assert store.canonical_entity("phone", "ＢＲＡＮＤＡ") == "品牌A"
    first = json.loads(store._path("demo-user").read_text().splitlines()[0])
    assert first["record_type"] == "entity_registry" and first["registry_digest"] == table.digest


def test_other_alias_requires_same_explicit_revision_chain(tmp_path):
    store, _ = configured(tmp_path)
    first = sample_memory(entity="BrandA")
    store.add(first)
    with pytest.raises(ValueError):
        store.add(replace(first, memory_id="bad", entity="A牌"))
    second = replace(
        first,
        memory_id="new",
        entity="A牌",
        relation="prefer",
        supersedes="m1",
        occurred_at=T0 + timedelta(hours=1),
        available_at=T0 + timedelta(hours=1),
    )
    store.add(second)
    active = store.active_memories("demo-user", entity="BrandA", as_of=AT)
    assert len(active) == 1 and active[0].memory_id == "new" and active[0].entity == "品牌A"


def test_delete_alias_covers_canonical_history_and_old_alias_replay(tmp_path):
    store, _ = configured(tmp_path)
    old = sample_memory(entity="BrandA")
    store.add(old)
    assert (
        store.delete_entity(
            "demo-user", category="phone", entity="A牌", deleted_at=T0 + timedelta(hours=1)
        )
        == 1
    )
    assert store.list_memories("demo-user", as_of=AT) == ()
    assert store.evidence_is_deleted(
        "demo-user", category="phone", entity="ＢＲＡＮＤＡ", available_at=T0
    )
    with pytest.raises(ValueError):
        store.add(old)
    store.add(
        replace(
            old,
            memory_id="fresh",
            available_at=T0 + timedelta(hours=2),
            occurred_at=T0 + timedelta(hours=2),
        )
    )
    assert store.active_memories("demo-user", as_of=AT)[0].entity == "品牌A"


@pytest.mark.parametrize("change", ["remove", "source"])
def test_registry_change_never_reinterprets_saved_data(tmp_path, change):
    store, _ = configured(tmp_path)
    store.add(sample_memory(entity="BrandA"))
    path = tmp_path / "entity-aliases.json"
    if change == "remove":
        path.unlink()
    else:
        raw = json.loads(path.read_text())
        raw["entries"][0]["source"] = "different-version"
        path.write_text(json.dumps(raw))
    with pytest.raises(ValueError):
        store.snapshot_token("demo-user")
    with pytest.raises(ValueError):
        JsonlMemoryStore(tmp_path).snapshot_token("demo-user")


def test_legacy_records_cannot_be_silently_adopted_by_dropped_in_table(tmp_path):
    store = JsonlMemoryStore(tmp_path)
    store.add(sample_memory(entity="BrandA"))
    original = store._path("demo-user").read_bytes()
    (tmp_path / "entity-aliases.json").write_text(json.dumps(document()))
    with pytest.raises(ValueError):
        JsonlMemoryStore(tmp_path).list_memories("demo-user", as_of=AT)
    assert store._path("demo-user").read_bytes() == original


def test_store_opened_before_alias_install_must_be_reopened(tmp_path):
    old = JsonlMemoryStore(tmp_path)
    install_entity_aliases(tmp_path, document())
    with pytest.raises(ValueError):
        old.snapshot_token("demo-user")
    assert JsonlMemoryStore(tmp_path).canonical_entity("phone", "BrandA") == "品牌A"


def test_feedback_brand_target_shares_canonical_delete_scope(tmp_path):
    from memrecagent.memory_feedback import FeedbackTarget, HostFeedbackSession
    from memrecagent.memory_feedback_store import JsonFeedbackStore

    install_entity_aliases(tmp_path, document())
    store = JsonFeedbackStore(tmp_path / "feedback")
    host = HostFeedbackSession(store, user_id="demo-user", session_id="S1")
    display = host.register_display(FeedbackTarget("brand", "phone", "BrandA"), displayed_at=T0)
    host.submit(display, event_id="block", at=T0)
    assert store.blocked("demo-user", FeedbackTarget("brand", "phone", "A牌"), AT)
    assert host.delete(FeedbackTarget("brand", "phone", "A牌"), at=AT) == 1
    with pytest.raises(ValueError):
        host.submit(display, event_id="replay", at=AT)
    data = json.loads(store._path("demo-user").read_text())
    assert data["schema_version"] == 2 and len(data["entity_registry_digest"]) == 64


def test_feedback_cannot_reopen_under_changed_alias_definition(tmp_path):
    from test_feedback_ranking import block

    from memrecagent.memory_feedback_store import JsonFeedbackStore

    install_entity_aliases(tmp_path, document())
    store, _ = block(tmp_path, kind="brand", target="BrandA")
    (tmp_path / "entity-aliases.json").unlink()
    with pytest.raises(ValueError):
        store.snapshot_token("demo-user")
    with pytest.raises(ValueError):
        JsonFeedbackStore(tmp_path / "feedback").snapshot_token("demo-user")


def test_alias_rank_request_and_unified_delete_share_one_logical_entity(tmp_path):
    from test_feedback_ranking import block
    from test_memory_products import engine, request

    from memrecagent.memory_management import MemoryManager

    store, _ = configured(tmp_path)
    store.add(sample_memory(entity="BrandA"))
    block(tmp_path, kind="brand", target="A牌")
    ranker = engine(tmp_path)
    assert [p["item_id"] for p in ranker.recommend(**request())["items"]] == ["B", "C"]
    assert [
        p["item_id"] for p in ranker.recommend(**request(requested_entity="ＢＲＡＮＤＡ"))["items"]
    ] == ["A"]
    result = MemoryManager(tmp_path).forget(
        "demo-user", kind="brand", category="phone", target="BrandA", at=AT
    )
    assert result == dict(deleted_memories=1, deleted_feedback=1)
    assert store.list_memories("demo-user", as_of=AT) == ()


def test_alias_management_interruption_recovers_only_under_same_table(tmp_path, monkeypatch):
    from test_feedback_ranking import block

    import memrecagent.memory_management as module
    from memrecagent.memory_management import MemoryManager
    from memrecagent.memory_management_guard import management_marker

    store, _ = configured(tmp_path)
    store.add(sample_memory(entity="BrandA"))
    block(tmp_path, kind="brand", target="A牌")
    manager = MemoryManager(tmp_path)
    actual = module.os.replace

    def fail(source, target):
        if target == manager.feedback._path("demo-user"):
            raise OSError("interrupted")
        return actual(source, target)

    monkeypatch.setattr(module.os, "replace", fail)
    with pytest.raises(OSError):
        manager.forget("demo-user", kind="brand", category="phone", target="A牌", at=AT)
    monkeypatch.setattr(module.os, "replace", actual)
    marker = management_marker(tmp_path, "demo-user")
    config = tmp_path / "entity-aliases.json"
    original = config.read_bytes()
    changed = json.loads(original)
    changed["entries"][0]["source"] = "tampered"
    config.write_text(json.dumps(changed))
    with pytest.raises(ValueError):
        MemoryManager(tmp_path).recover("demo-user")
    assert marker.exists()
    config.write_bytes(original)
    assert MemoryManager(tmp_path).recover("demo-user")
    assert store.list_memories("demo-user", as_of=AT) == ()
