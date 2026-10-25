"""用真实临时文件验证纠正、删除、重启与旧证据重放边界。"""

import json
from dataclasses import replace
from datetime import timedelta

import pytest
from test_memory import T0, sample_memory
from test_memory_store import AFTER, file_for, revision

from memrecagent.memory_policy import ContextPreference, resolve_recommendation
from memrecagent.memory_store import JsonlMemoryStore

DELETED = T0 + timedelta(hours=4)


def correct(store, **changes):
    values = dict(
        category="phone",
        entity="小米",
        expected_head="m1",
        relation="neutral",
        evidence_ref="S2:T5",
        evidence_text="以后不用排除小米了（合成）",
        occurred_at=T0 + timedelta(hours=1),
        available_at=T0 + timedelta(hours=2),
    )
    return store.correct("demo-user", **(values | changes))


def delete(store, **changes):
    return store.delete_entity(
        "demo-user", **(dict(category="phone", entity="小米", deleted_at=DELETED) | changes)
    )


def test_view_filters_entity_and_distinguishes_history_from_current(tmp_path):
    store = JsonlMemoryStore(tmp_path)
    old = sample_memory()
    store.add(old)
    store.add(revision(old))
    store.add(sample_memory(memory_id="b", entity="三星"))
    assert [m.memory_id for m in store.list_memories("demo-user", entity="小米", as_of=AFTER)] == [
        "m1",
        "m2",
    ]
    assert [
        m.memory_id
        for m in store.active_memories("demo-user", category="phone", entity="小米", as_of=AFTER)
    ] == ["m2"]


def test_explicit_correction_preserves_evidence_and_survives_restart(tmp_path):
    store = JsonlMemoryStore(tmp_path)
    old = sample_memory()
    store.add(old)
    new = correct(store)
    reopened = JsonlMemoryStore(tmp_path)
    assert new.supersedes == "m1" and new.relation == "neutral"
    assert new.evidence_text == "以后不用排除小米了（合成）"
    assert new.confidence is None
    assert reopened.list_memories("demo-user", as_of=AFTER) == (old, new)
    assert reopened.active_memories("demo-user", as_of=T0 + timedelta(minutes=30)) == (old,)
    assert reopened.active_memories("demo-user", as_of=AFTER) == (new,)


@pytest.mark.parametrize(
    "changes",
    [
        {"expected_head": "wrong"},
        {"entity": "未知品牌"},
        {"category": "other"},
        {"occurred_at": T0},
        {"relation": "none"},
        {"evidence_text": ""},
    ],
)
def test_invalid_or_stale_correction_keeps_original_file(tmp_path, changes):
    store = JsonlMemoryStore(tmp_path)
    store.add(sample_memory())
    before = file_for(tmp_path).read_bytes()
    with pytest.raises(ValueError):
        correct(store, **changes)
    assert file_for(tmp_path).read_bytes() == before


def test_delete_erases_entire_chain_but_preserves_other_entities_categories_users(tmp_path):
    store = JsonlMemoryStore(tmp_path)
    old = sample_memory()
    for row in [
        old,
        revision(old),
        sample_memory(memory_id="b", entity="三星", evidence_text="其他品牌证据"),
        sample_memory(memory_id="c", category="instrument", evidence_text="其他类别证据"),
        sample_memory(user_id="another-user"),
    ]:
        store.add(row)
    assert delete(store) == 2
    reopened = JsonlMemoryStore(tmp_path)
    assert [m.memory_id for m in reopened.list_memories("demo-user", as_of=AFTER)] == ["b", "c"]
    assert len(reopened.list_memories("another-user", as_of=AFTER)) == 1
    content = file_for(tmp_path).read_text()
    assert "这是合成表达" not in content and "以后可以推荐" not in content
    assert '"m1"' not in content and '"m2"' not in content
    assert reopened.list_memories("demo-user", category="phone", entity="小米", as_of=AFTER) == ()
    # 删除是隐私操作，不允许用历史时点恢复已删除原文。
    assert reopened.list_memories("demo-user", category="phone", entity="小米", as_of=DELETED) == ()


def test_delete_last_chain_retains_no_raw_identity_entity_or_evidence(tmp_path):
    store = JsonlMemoryStore(tmp_path)
    store.add(sample_memory())
    delete(store)
    content = file_for(tmp_path).read_text()
    for raw in ["demo-user", "小米", "S1:T3", "以后买手机", '"m1"']:
        assert raw not in content
    # 删除水位在其他品牌新增之后仍要保留。
    store.add(sample_memory(memory_id="other", entity="三星"))
    with pytest.raises(ValueError):
        JsonlMemoryStore(tmp_path).add(sample_memory())


def test_correction_rejects_a_head_that_was_current_before_another_correction(tmp_path):
    store = JsonlMemoryStore(tmp_path)
    store.add(sample_memory())
    first = correct(store)
    before = file_for(tmp_path).read_bytes()
    with pytest.raises(ValueError):
        correct(store, relation="prefer", occurred_at=DELETED, available_at=DELETED)
    assert file_for(tmp_path).read_bytes() == before
    assert store.active_memories("demo-user", as_of=AFTER) == (first,)


def test_deletion_blocks_old_and_regenerated_proposals_but_allows_new_evidence(tmp_path):
    store = JsonlMemoryStore(tmp_path)
    old = sample_memory()
    store.add(old)
    delete(store)
    reopened = JsonlMemoryStore(tmp_path)
    for candidate in [
        old,
        replace(old, memory_id="regenerated"),
        replace(old, memory_id="late", available_at=AFTER),
        replace(old, memory_id="boundary", occurred_at=DELETED, available_at=DELETED),
    ]:
        with pytest.raises(ValueError):
            reopened.add(candidate)
    fresh = replace(
        old,
        memory_id="fresh",
        relation="prefer",
        evidence_ref="S3:T1",
        evidence_text="现在开始我喜欢小米（合成）",
        occurred_at=AFTER,
        available_at=AFTER,
    )
    reopened.add(fresh)
    assert reopened.active_memories("demo-user", as_of=AFTER + timedelta(seconds=1)) == (fresh,)


def test_delete_without_saved_record_also_invalidates_pending_old_proposals(tmp_path):
    store = JsonlMemoryStore(tmp_path)
    assert delete(store) == 0
    with pytest.raises(ValueError):
        JsonlMemoryStore(tmp_path).add(sample_memory())
    before = file_for(tmp_path).read_bytes()
    assert delete(store) == 0
    assert file_for(tmp_path).read_bytes() == before


@pytest.mark.parametrize("bad_time", [T0, DELETED.replace(tzinfo=None)])
def test_delete_cannot_precede_known_evidence_or_use_naive_time(tmp_path, bad_time):
    store = JsonlMemoryStore(tmp_path)
    store.add(sample_memory())
    before = file_for(tmp_path).read_bytes()
    with pytest.raises(ValueError):
        delete(store, deleted_at=bad_time)
    assert file_for(tmp_path).read_bytes() == before


@pytest.mark.parametrize("operation", ["delete", "correct"])
def test_failed_management_publish_keeps_all_old_bytes(tmp_path, monkeypatch, operation):
    store = JsonlMemoryStore(tmp_path)
    store.add(sample_memory())
    before = file_for(tmp_path).read_bytes()

    def fail(*args):
        raise OSError("模拟原子替换失败")

    monkeypatch.setattr("memrecagent.memory_store.os.replace", fail)
    with pytest.raises(OSError):
        delete(store) if operation == "delete" else correct(store)
    assert file_for(tmp_path).read_bytes() == before
    assert list(tmp_path.iterdir()) == [file_for(tmp_path)]


def test_deleted_self_context_cannot_reactivate_exclusion_but_other_person_is_retained(tmp_path):
    store = JsonlMemoryStore(tmp_path)
    store.add(sample_memory())
    delete(store)
    context = ContextPreference(
        "demo-user",
        "S2",
        "self",
        "phone",
        "小米",
        "exclude",
        "S1:T3",
        "合成本人旧排除",
        T0 + timedelta(minutes=1),
    )

    def resolve(who, row):
        return resolve_recommendation(
            store,
            user_id="demo-user",
            session_id="S2",
            beneficiary_id=who,
            category="phone",
            as_of=AFTER,
            context_preferences=(row,),
        )

    assert resolve("self", context).excluded_entities == ()
    other = replace(context, beneficiary_id="person:dad")
    assert resolve("person:dad", other).excluded_entities == ("小米",)


def test_malformed_or_cross_owner_deletion_marker_is_not_ignored(tmp_path):
    store = JsonlMemoryStore(tmp_path)
    delete(store)
    original = json.loads(file_for(tmp_path).read_text())
    for changes in [
        {"deleted_at": "private-bad-time"},
        {"deleted_at": "0001-01-01T00:00:00+01:00"},
        {"owner_digest": "0" * 64},
        {"scope_digest": "no-digest"},
        {"schema_version": True},
        {"unexpected": "private-content"},
    ]:
        file_for(tmp_path).write_text(json.dumps(original | changes) + "\n")
        with pytest.raises(ValueError) as error:
            store.list_memories("demo-user", as_of=AFTER)
        assert "private" not in str(error.value)


def test_delete_does_not_hide_corrupt_unrelated_records(tmp_path):
    store = JsonlMemoryStore(tmp_path)
    store.add(sample_memory())
    with file_for(tmp_path).open("ab") as f:
        f.write(b"{}\n")
    before = file_for(tmp_path).read_bytes()
    with pytest.raises(ValueError):
        delete(store)
    assert file_for(tmp_path).read_bytes() == before
