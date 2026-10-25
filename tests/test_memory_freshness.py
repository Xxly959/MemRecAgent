"""可选年龄策略只控制激活，不能删除历史或复活被修订的旧态度。"""

from dataclasses import replace
from datetime import timedelta

import pytest
from test_memory import T0, sample_memory
from test_memory_extraction import NOW, decode, output_for, turns_for
from test_memory_store import revision

from memrecagent.memory_commit import MemoryCommitSession
from memrecagent.memory_policy import resolve_recommendation
from memrecagent.memory_store import JsonlMemoryStore


def test_default_stable_preference_has_no_automatic_expiry(tmp_path):
    store = JsonlMemoryStore(tmp_path)
    old = sample_memory()
    store.add(old)
    assert store.active_memories("demo-user", as_of=T0 + timedelta(days=365)) == (old,)
    assert (
        store.active_memories(
            "demo-user", as_of=T0 + timedelta(days=1), max_memory_age=timedelta(days=1)
        )
        == ()
    )
    assert store.list_memories("demo-user", as_of=T0 + timedelta(days=365)) == (old,)


def test_exact_age_boundary_and_late_arrival_do_not_refresh_evidence(tmp_path):
    store = JsonlMemoryStore(tmp_path)
    old = sample_memory(available_at=T0 + timedelta(hours=20))
    store.add(old)
    assert store.active_memories(
        "demo-user", as_of=T0 + timedelta(hours=23), max_memory_age=timedelta(days=1)
    ) == (old,)
    assert (
        store.active_memories(
            "demo-user", as_of=T0 + timedelta(days=1), max_memory_age=timedelta(days=1)
        )
        == ()
    )


def test_expiry_is_after_revision_resolution_and_never_revives_old_exclusion(tmp_path):
    store = JsonlMemoryStore(tmp_path)
    old = sample_memory()
    new = revision(old)
    store.add(old)
    store.add(new)
    assert (
        store.active_memories(
            "demo-user", as_of=T0 + timedelta(days=2), max_memory_age=timedelta(days=1)
        )
        == ()
    )
    assert store.list_memories("demo-user", as_of=T0 + timedelta(days=2)) == (old, new)
    assert store.active_memories(
        "demo-user", as_of=T0 + timedelta(hours=3), max_memory_age=timedelta(days=1)
    ) == (new,)


def test_expired_head_is_still_superseded_by_fresh_revision(tmp_path):
    store = JsonlMemoryStore(tmp_path)
    old = sample_memory()
    store.add(old)
    turns = tuple(
        replace(
            t,
            occurred_at=t.occurred_at + timedelta(days=2),
            available_at=t.available_at + timedelta(days=2),
        )
        for t in turns_for("以后我喜欢小米")
    )
    data = output_for(turns[0])
    data["assertions"][0]["relation"] = "prefer"
    session = MemoryCommitSession(
        store, user_id="demo-user", session_id="S2", max_memory_age=timedelta(days=1)
    )
    now = NOW + timedelta(days=2, hours=1)
    preview = session.prepare(decode(data, turns), turns=turns, as_of=now)
    assert preview.proposal.proposed_memories[0].supersedes == "m1"
    session.commit(preview.ticket_id, as_of=now)
    assert [
        m.relation
        for m in store.active_memories("demo-user", as_of=now, max_memory_age=timedelta(days=1))
    ] == ["prefer"]


def test_policy_consumes_only_fresh_heads_when_explicitly_configured(tmp_path):
    store = JsonlMemoryStore(tmp_path)
    store.add(sample_memory())
    decision = resolve_recommendation(
        store,
        user_id="demo-user",
        session_id="S2",
        beneficiary_id="self",
        category="phone",
        as_of=T0 + timedelta(days=2),
        max_memory_age=timedelta(days=1),
    )
    assert decision.memories == () and decision.excluded_entities == ()


@pytest.mark.parametrize("age", [timedelta(0), timedelta(seconds=-1), True, 1, "1"])
def test_invalid_age_configuration_is_rejected_without_writing(tmp_path, age):
    store = JsonlMemoryStore(tmp_path)
    with pytest.raises(ValueError):
        store.active_memories("demo-user", as_of=T0, max_memory_age=age)
    assert list(tmp_path.iterdir()) == []
