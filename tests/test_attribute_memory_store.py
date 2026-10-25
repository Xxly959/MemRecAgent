"""混合单文件必须原子、隔离类型修订与删除，不影响旧品牌接口。"""

from datetime import timedelta

import pytest
from test_memory import T0, sample_memory

from memrecagent.attribute_memory import AttributeMemory
from memrecagent.memory_store import JsonlMemoryStore


def attribute(**changes):
    return AttributeMemory(
        **(
            dict(
                memory_id="a1",
                user_id="demo-user",
                category="phone",
                attribute="price_cny",
                operator="<=",
                value=3000,
                unit="CNY",
                is_hard=True,
                active=True,
                evidence_ref="S1:T1",
                evidence_text="以后预算三千。",
                occurred_at=T0,
                available_at=T0,
            )
            | changes
        )
    )


def test_mixed_batch_survives_restart_and_brand_api_stays_brand_only(tmp_path):
    store = JsonlMemoryStore(tmp_path)
    brand, budget = sample_memory(), attribute()
    store.add_many((brand, budget))
    reopened = JsonlMemoryStore(tmp_path)
    assert reopened.list_memories("demo-user", as_of=T0 + timedelta(minutes=2)) == (brand,)
    assert reopened.active_attribute_memories("demo-user", as_of=T0 + timedelta(seconds=1)) == (
        budget,
    )
    assert len(list(tmp_path.glob("*.jsonl"))) == 1
    assert AttributeMemory.from_dict(budget.to_dict()) == budget
    assert "三千" not in repr(budget) and "3000" not in repr(budget)


def test_failed_mixed_atomic_replace_keeps_both_old_types_unchanged(tmp_path, monkeypatch):
    store = JsonlMemoryStore(tmp_path)
    store.add_many((sample_memory(), attribute()))
    path = next(tmp_path.glob("*.jsonl"))
    before = path.read_bytes()

    def fail(*args):
        raise OSError("synthetic replace failure")

    monkeypatch.setattr("memrecagent.memory_store.os.replace", fail)
    with pytest.raises(OSError):
        store.add_many(
            (
                sample_memory(memory_id="brand-new", entity="品牌B"),
                attribute(
                    memory_id="a2",
                    value=4000,
                    occurred_at=T0 + timedelta(seconds=1),
                    available_at=T0 + timedelta(seconds=1),
                    supersedes="a1",
                ),
            )
        )
    assert path.read_bytes() == before
    assert list(tmp_path.glob(".memory-*.tmp")) == []


@pytest.mark.parametrize("parent", ["brand", "another-operator", "soft"])
def test_revision_cannot_cross_record_type_operator_or_hardness(tmp_path, parent):
    store = JsonlMemoryStore(tmp_path)
    old = (
        sample_memory()
        if parent == "brand"
        else attribute(
            operator=">=" if parent == "another-operator" else "<=", is_hard=parent != "soft"
        )
    )
    store.add_many((old,))
    with pytest.raises(ValueError):
        store.add_many(
            (
                attribute(
                    memory_id="next",
                    supersedes=old.memory_id,
                    occurred_at=T0 + timedelta(seconds=1),
                    available_at=T0 + timedelta(seconds=1),
                ),
            )
        )


def test_late_revision_and_retraction_only_apply_after_available_time(tmp_path):
    store = JsonlMemoryStore(tmp_path)
    first = attribute()
    next_ = attribute(
        memory_id="a2",
        value=4000,
        supersedes="a1",
        occurred_at=T0 + timedelta(hours=1),
        available_at=T0 + timedelta(hours=3),
    )
    cancelled = attribute(
        memory_id="a3",
        value=4000,
        active=False,
        supersedes="a2",
        occurred_at=T0 + timedelta(hours=4),
        available_at=T0 + timedelta(hours=4),
    )
    store.add_many((first, next_, cancelled))
    assert store.active_attribute_memories("demo-user", as_of=T0 + timedelta(hours=2)) == (first,)
    assert store.active_attribute_memories("demo-user", as_of=T0 + timedelta(hours=3)) == (first,)
    assert store.active_attribute_memories(
        "demo-user", as_of=T0 + timedelta(hours=3, seconds=1)
    ) == (next_,)
    assert store.active_attribute_memories("demo-user", as_of=T0 + timedelta(hours=5)) == ()
    assert (
        store.active_attribute_memories(
            "demo-user", as_of=T0 + timedelta(hours=6), max_memory_age=timedelta(minutes=1)
        )
        == ()
    )
    assert store.active_attribute_memories(
        "demo-user", as_of=T0 + timedelta(hours=5), include_inactive=True
    ) == (cancelled,)


def test_attribute_delete_preserves_brand_but_blocks_all_old_modes_and_operators(tmp_path):
    store = JsonlMemoryStore(tmp_path)
    brand = sample_memory(entity="price_cny")
    store.add_many((brand, attribute(), attribute(memory_id="soft", is_hard=False, value=2500)))
    before = store.snapshot_token("demo-user")
    assert (
        store.delete_attribute(
            "demo-user", category="phone", attribute="price_cny", deleted_at=T0 + timedelta(hours=1)
        )
        == 2
    )
    assert store.snapshot_token("demo-user") != before
    assert store.list_memories("demo-user", as_of=T0 + timedelta(days=1)) == (brand,)
    assert store.list_attribute_memories("demo-user", as_of=T0 + timedelta(days=1)) == ()
    for op in ("<=", ">=", "=="):
        for hard in (True, False):
            with pytest.raises(ValueError):
                store.add_many((attribute(operator=op, is_hard=hard),))
    fresh = attribute(
        memory_id="fresh", occurred_at=T0 + timedelta(hours=2), available_at=T0 + timedelta(hours=2)
    )
    store.add_many((fresh,))
    assert store.active_attribute_memories("demo-user", as_of=T0 + timedelta(days=1)) == (fresh,)


def test_brand_delete_and_correction_do_not_touch_attribute_records(tmp_path):
    store = JsonlMemoryStore(tmp_path)
    store.add_many((sample_memory(entity="price_cny"), attribute()))
    store.delete_entity(
        "demo-user", category="phone", entity="price_cny", deleted_at=T0 + timedelta(hours=1)
    )
    assert store.active_attribute_memories("demo-user", as_of=T0 + timedelta(days=1)) == (
        attribute(),
    )
    updated = store.correct_attribute(
        "demo-user",
        category="phone",
        attribute="price_cny",
        operator="<=",
        is_hard=True,
        expected_head="a1",
        value=4500,
        unit="CNY",
        active=True,
        evidence_ref="S2:T1",
        evidence_text="以后预算四千五。",
        occurred_at=T0 + timedelta(hours=2),
        available_at=T0 + timedelta(hours=2),
    )
    assert updated.supersedes == "a1" and updated.value == 4500
    assert len(store.list_attribute_memories("demo-user", as_of=T0 + timedelta(days=1))) == 2


@pytest.mark.parametrize(
    "change",
    [
        dict(subject_id="uncle"),
        dict(is_hard=1),
        dict(active=1),
        dict(value=True),
        dict(unit="USD"),
        dict(available_at=T0 - timedelta(seconds=1)),
    ],
)
def test_attribute_record_rejects_owner_unit_type_and_time_errors(change):
    with pytest.raises(ValueError):
        attribute(**change)


def test_unknown_record_fields_fail_closed_without_hiding_valid_brand_rows(tmp_path):
    store = JsonlMemoryStore(tmp_path)
    store.add_many((sample_memory(), attribute()))
    row = attribute().to_dict() | {"approved": True}
    with pytest.raises(ValueError):
        AttributeMemory.from_dict(row)
    import json

    path = next(tmp_path.glob("*.jsonl"))
    with path.open("a") as stream:
        stream.write(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="记忆记录无效"):
        store.list_memories("demo-user", as_of=T0 + timedelta(days=1))


def test_attribute_mutation_invalidates_old_brand_batch_token(tmp_path):
    store = JsonlMemoryStore(tmp_path)
    token = store.snapshot_token("demo-user")
    store.add_many((attribute(),))
    with pytest.raises(ValueError):
        store.add_many((sample_memory(),), expected_token=token)
