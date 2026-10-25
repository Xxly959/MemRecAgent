"""统一删除真实双文件，失败不发布可读的半个用户状态。"""

import json
from datetime import timedelta

import pytest
from test_attribute_memory_store import attribute
from test_feedback_ranking import block
from test_memory import T0, sample_memory
from test_memory_products import AT, engine, request

from memrecagent.memory_feedback import FeedbackTarget
from memrecagent.memory_feedback_store import JsonFeedbackStore
from memrecagent.memory_management import MemoryManager
from memrecagent.memory_store import JsonlMemoryStore


def setup(root):
    store = JsonlMemoryStore(root)
    store.add(sample_memory(entity="品牌A"))
    store.add(attribute())
    block(root, kind="brand", target="品牌A")
    return store, MemoryManager(root)


def test_brand_forget_removes_both_histories_and_rejects_old_evidence(tmp_path):
    store, manager = setup(tmp_path)
    old = store.list_memories("demo-user", as_of=AT)[0]
    before = engine(tmp_path).state_token("demo-user")
    result = manager.forget("demo-user", kind="brand", category="phone", target="品牌A", at=AT)
    assert result == dict(deleted_memories=1, deleted_feedback=1)
    assert store.list_memories("demo-user", as_of=AT + timedelta(seconds=1)) == ()
    assert len(store.active_attribute_memories("demo-user", category="phone", as_of=AT)) == 1
    assert manager.list_feedback("demo-user", as_of=AT + timedelta(seconds=1)) == []
    assert engine(tmp_path).state_token("demo-user") != before
    with pytest.raises(ValueError):
        store.add(old)
    assert not JsonFeedbackStore(tmp_path / "feedback").blocked(
        "demo-user", FeedbackTarget("brand", "phone", "品牌A"), AT + timedelta(seconds=1)
    )


def test_product_forget_does_not_delete_brand_preferences_or_brand_feedback(tmp_path):
    store, manager = setup(tmp_path)
    block(tmp_path, event="product")
    manager.forget("demo-user", kind="product", category="phone", target="A", at=AT)
    assert len(store.list_memories("demo-user", as_of=AT)) == 1
    rows = manager.list_feedback("demo-user", as_of=AT)
    assert len(rows) == 1 and rows[0]["target"] == dict(
        kind="brand", category="phone", target_id="品牌A"
    )


def test_attribute_forget_covers_all_predicates_without_touching_brand(tmp_path):
    store, manager = setup(tmp_path)
    result = manager.forget(
        "demo-user", kind="attribute", category="phone", target="price_cny", at=AT
    )
    assert result == dict(deleted_memories=1, deleted_feedback=0)
    assert store.active_attribute_memories("demo-user", category="phone", as_of=AT) == ()
    with pytest.raises(ValueError):
        store.add(attribute(memory_id="replay"))
    assert len(store.list_memories("demo-user", as_of=AT)) == 1


def test_bad_time_and_corrupt_second_store_leave_original_files_unchanged(tmp_path):
    store, manager = setup(tmp_path)
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    with pytest.raises(ValueError):
        manager.forget(
            "demo-user", kind="brand", category="phone", target="品牌A", at=T0 - timedelta(days=1)
        )
    assert {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()} == before
    feedback = JsonFeedbackStore(tmp_path / "feedback")._path("demo-user")
    feedback.write_text("bad")
    before = store.snapshot_token("demo-user")
    with pytest.raises(ValueError):
        manager.forget("demo-user", kind="brand", category="phone", target="品牌A", at=AT)
    assert store.snapshot_token("demo-user") == before and feedback.read_text() == "bad"


def test_partial_publish_blocks_both_read_paths_until_verified_recovery(tmp_path, monkeypatch):
    import memrecagent.memory_management as module

    store, manager = setup(tmp_path)
    actual = module.os.replace
    once = False

    def fail_second(source, target):
        nonlocal once
        if target == JsonFeedbackStore(tmp_path / "feedback")._path("demo-user") and not once:
            once = True
            raise OSError("simulated second publication failure")
        return actual(source, target)

    monkeypatch.setattr(module.os, "replace", fail_second)
    with pytest.raises(OSError):
        manager.forget("demo-user", kind="brand", category="phone", target="品牌A", at=AT)
    with pytest.raises(ValueError, match="management_pending"):
        store.snapshot_token("demo-user")
    with pytest.raises(ValueError, match="management_pending"):
        JsonFeedbackStore(tmp_path / "feedback").active_targets("demo-user", as_of=AT)
    with pytest.raises(ValueError, match="management_pending"):
        engine(tmp_path).recommend(**request())
    monkeypatch.setattr(module.os, "replace", actual)
    assert manager.recover("demo-user") is True
    assert manager.recover("demo-user") is False
    assert store.list_memories("demo-user", as_of=AT + timedelta(seconds=1)) == ()
    assert manager.list_feedback("demo-user", as_of=AT + timedelta(seconds=1)) == []


def test_feedback_view_has_no_raw_user_text_or_display_tokens(tmp_path):
    _, manager = setup(tmp_path)
    rows = manager.list_feedback("demo-user", as_of=AT)
    assert rows[0].keys() == {"target", "event_digest", "at"}
    assert rows[0]["target"]["target_id"] == "品牌A"
    assert len(rows[0]["event_digest"]) == 64


def test_feedback_only_forget_keeps_same_brand_preference(tmp_path):
    store, manager = setup(tmp_path)
    result = manager.forget(
        "demo-user", kind="brand", category="phone", target="品牌A", at=AT, feedback_only=True
    )
    assert result == dict(deleted_memories=0, deleted_feedback=1)
    assert len(store.list_memories("demo-user", as_of=AT)) == 1
    assert manager.list_feedback("demo-user", as_of=AT) == []


def test_recovery_rejects_stage_tampering_without_releasing_read_guard(tmp_path, monkeypatch):
    import memrecagent.memory_management as module
    from memrecagent.memory_management_guard import management_marker

    _, manager = setup(tmp_path)
    actual = module.os.replace

    def fail(source, target):
        if target == JsonFeedbackStore(tmp_path / "feedback")._path("demo-user"):
            raise OSError("interrupted")
        return actual(source, target)

    monkeypatch.setattr(module.os, "replace", fail)
    with pytest.raises(OSError):
        manager.forget("demo-user", kind="brand", category="phone", target="品牌A", at=AT)
    monkeypatch.setattr(module.os, "replace", actual)
    marker = management_marker(tmp_path, "demo-user")
    journal = json.loads(marker.read_text())
    staged = (
        marker.parent
        / journal["stage"]
        / "feedback"
        / JsonFeedbackStore(tmp_path / "feedback")._path("demo-user").name
    )
    staged.write_text("corrupt")
    with pytest.raises(ValueError, match="漂移"):
        manager.recover("demo-user")
    assert marker.exists()
    with pytest.raises(ValueError, match="management_pending"):
        manager.list_feedback("demo-user", as_of=AT)


def test_cli_unified_inspect_and_feedback_only_forget_across_processes(tmp_path):
    from test_memory_management_cli import run

    setup(tmp_path)
    view = run(tmp_path, "inspect", "--as-of", AT.isoformat())
    assert view.returncode == 0, view.stderr
    result = json.loads(view.stdout)
    assert len(result["preferences"]) == len(result["attributes"]) == len(result["feedback"]) == 1
    deleted = run(
        tmp_path,
        "forget-feedback",
        "--kind",
        "brand",
        "--category",
        "phone",
        "--target",
        "品牌A",
        "--at",
        AT.isoformat(),
    )
    assert deleted.returncode == 0, deleted.stderr
    assert json.loads(deleted.stdout) == dict(deleted_memories=0, deleted_feedback=1)
    recovery = run(tmp_path, "recover")
    assert recovery.returncode == 0 and json.loads(recovery.stdout) == {"recovered": False}


def test_source_change_during_preparation_cannot_be_overwritten(tmp_path, monkeypatch):
    store, manager = setup(tmp_path)
    original = JsonlMemoryStore._write
    changed = False

    def mutate(candidate, user, records, deleted):
        nonlocal changed
        if candidate.root != tmp_path and not changed:
            changed = True
            store.add(sample_memory(memory_id="new-b", entity="品牌B", evidence_ref="S9:T1"))
        return original(candidate, user, records, deleted)

    monkeypatch.setattr(JsonlMemoryStore, "_write", mutate)
    with pytest.raises(ValueError, match="漂移"):
        manager.forget("demo-user", kind="brand", category="phone", target="品牌A", at=AT)
    assert {m.entity for m in store.list_memories("demo-user", as_of=AT)} == {"品牌A", "品牌B"}


def test_pending_delete_clears_live_rpc_without_extra_analysis(tmp_path, monkeypatch):
    from test_conversation_receipt_privacy import setup as conversation

    import memrecagent.memory_management as module

    h, bridge, client, _, old_query, _, t = conversation(tmp_path)
    block(tmp_path / "memory", user="u", kind="brand", target="品牌C")
    manager = MemoryManager(tmp_path / "memory")
    actual = module.os.replace

    def fail(source, target):
        if target == manager.feedback._path("u"):
            raise OSError("interrupted")
        return actual(source, target)

    monkeypatch.setattr(module.os, "replace", fail)
    with pytest.raises(OSError):
        manager.forget("u", kind="brand", category="phone", target="品牌C", at=AT)
    replay = bridge.handle(old_query)
    assert not replay["ok"] and replay["error"]["code"] == "conversation_state"
    assert h._memory.turns == () and len(client.seen) == 4
    assert h._last_request is None and h._feedback_display is None
    monkeypatch.setattr(module.os, "replace", actual)
    assert manager.recover("u")
