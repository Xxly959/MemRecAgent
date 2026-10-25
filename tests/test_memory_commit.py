"""合成候选与真实临时文件验证宿主票据、确认回复和原子提交。"""

from datetime import timedelta

import pytest
from test_memory import sample_memory
from test_memory_extraction import NOW, decode, output_for, span, subject, turns_for
from test_memory_store import file_for

from memrecagent.memory_commit import CommitError, MemoryCommitSession
from memrecagent.memory_extraction import EvidenceSpan, Turn
from memrecagent.memory_store import JsonlMemoryStore

AS_OF = NOW + timedelta(minutes=10)
LATER = NOW + timedelta(minutes=20)


def setup(tmp_path, text="以后排除小米", *, old=False, edit=None):
    store = JsonlMemoryStore(tmp_path)
    if old:
        store.add(sample_memory())
    turns = turns_for(text)
    data = output_for(turns[0])
    if edit:
        edit(data, turns)
    session = MemoryCommitSession(store, user_id="demo-user", session_id="S2")
    preview = session.prepare(decode(data, turns), turns=turns, as_of=AS_OF)
    return session, store, preview


def ambiguous(tmp_path, *, unknown=False):
    def edit(data, turns):
        data["assertions"][0].update(
            relation="prefer",
            scope="unclear" if not unknown else "long_term",
            subject=subject("unknown" if unknown else "self"),
        )

    return setup(tmp_path, "我喜欢小米（合成）", old=True, edit=edit)


def reply(text="是我自己，以后长期按这个偏好", **changes):
    return Turn(
        **(
            dict(
                turn_id="R1",
                role="user",
                content=text,
                occurred_at=NOW + timedelta(minutes=11),
                available_at=NOW + timedelta(minutes=11),
            )
            | changes
        )
    )


def test_clear_long_term_is_previewed_then_committed_once(tmp_path):
    session, store, preview = setup(tmp_path)
    assert preview.action == "apply"
    assert store.list_memories("demo-user", as_of=LATER) == ()
    first = session.commit(preview.ticket_id, as_of=LATER)
    assert first.status == "applied" and len(first.memory_ids) == 1
    saved = JsonlMemoryStore(tmp_path).active_memories("demo-user", as_of=LATER)
    assert [r.relation for r in saved] == ["exclude"]
    before = file_for(tmp_path).read_bytes()
    assert session.commit(preview.ticket_id, as_of=LATER).status == "already_applied"
    assert file_for(tmp_path).read_bytes() == before


def test_clarify_cannot_commit_and_current_confirmation_does_not_change_long_term(tmp_path):
    session, store, preview = ambiguous(tmp_path)
    before = file_for(tmp_path).read_bytes()
    assert preview.action == "clarify"
    with pytest.raises(CommitError, match="confirmation_required"):
        session.commit(preview.ticket_id, as_of=LATER)
    current = session.confirm(
        preview.ticket_id,
        reply=reply("是我自己，只用于这次"),
        subject="self",
        scope="current",
        as_of=LATER,
    )
    assert current.action == "current"
    assert session.commit(current.ticket_id, as_of=LATER).status == "not_applied"
    assert file_for(tmp_path).read_bytes() == before
    assert store.active_memories("demo-user", as_of=LATER)[0].relation == "exclude"


def test_long_term_confirmation_preserves_both_original_and_reply_evidence(tmp_path):
    session, store, preview = ambiguous(tmp_path)
    confirmed = session.confirm(
        preview.ticket_id, reply=reply(), subject="self", scope="long_term", as_of=LATER
    )
    assert confirmed.action == "apply"
    assert store.active_memories("demo-user", as_of=LATER)[0].relation == "exclude"
    session.commit(confirmed.ticket_id, as_of=LATER)
    current = JsonlMemoryStore(tmp_path).active_memories("demo-user", as_of=LATER)[0]
    assert current.relation == "prefer" and current.supersedes == "m1"
    assert "我喜欢小米（合成）" in current.evidence_text
    assert "是我自己，以后长期按这个偏好" in current.evidence_text
    assert "R1[" in current.evidence_ref
    assert len(store.list_memories("demo-user", as_of=LATER)) == 2


def test_unknown_subject_confirms_other_only_in_session(tmp_path):
    session, store, preview = ambiguous(tmp_path, unknown=True)
    answer = reply("是我爸喜欢，不是我")
    confirmed = session.confirm(
        preview.ticket_id,
        reply=answer,
        subject="other",
        scope="current",
        person_reference=EvidenceSpan("R1", 1, 3),
        as_of=LATER,
    )
    assert confirmed.action == "current"
    assert confirmed.proposal.proposed_memories == ()
    assert confirmed.proposal.session_preferences[0].beneficiary_id.startswith("person:")
    session.commit(confirmed.ticket_id, as_of=LATER)
    assert store.active_memories("demo-user", as_of=LATER)[0].relation == "exclude"


def test_unknown_without_conflict_does_not_guess_self_or_ask(tmp_path):
    session, store, preview = setup(
        tmp_path, edit=lambda d, t: d["assertions"][0].update(subject=subject("unknown"))
    )
    assert preview.action == "current"
    assert session.commit(preview.ticket_id, as_of=LATER).status == "not_applied"
    assert store.list_memories("demo-user", as_of=LATER) == ()


def test_neutral_without_prior_attitude_is_noop(tmp_path):
    session, store, preview = setup(
        tmp_path, edit=lambda d, t: d["assertions"][0].update(relation="neutral")
    )
    assert preview.action == "noop" and preview.proposal.proposed_memories == ()
    assert session.commit(preview.ticket_id, as_of=LATER).status == "not_applied"
    assert list(tmp_path.iterdir()) == []


def test_deleted_unclear_assertion_does_not_reappear_as_a_confirmation_question(tmp_path):
    store = JsonlMemoryStore(tmp_path)
    store.delete_entity("demo-user", category="phone", entity="小米", deleted_at=AS_OF)
    turns = turns_for("我喜欢小米")
    data = output_for(turns[0])
    data["assertions"][0].update(relation="prefer", scope="unclear")
    session = MemoryCommitSession(store, user_id="demo-user", session_id="S2")
    preview = session.prepare(decode(data, turns), turns=turns, as_of=LATER)
    assert preview.action == "noop"
    assert preview.proposal.decision.clarification is None


def test_unclear_self_scope_without_prior_memory_requires_confirmation(tmp_path):
    session, store, preview = setup(
        tmp_path, edit=lambda d, t: d["assertions"][0].update(scope="unclear")
    )
    assert preview.action == "clarify"
    with pytest.raises(CommitError, match="confirmation_required"):
        session.commit(preview.ticket_id, as_of=LATER)
    assert store.list_memories("demo-user", as_of=LATER) == ()


def test_confirmation_of_attitude_subject_preserves_explicit_recommendation_beneficiary(tmp_path):
    def edit(data, turns):
        data["assertions"][0].update(subject=subject("unknown"), relation="prefer")
        data["request"].update(
            is_recommendation=True, beneficiary=subject("self"), evidence=[span(turns[0])]
        )

    session, store, preview = setup(tmp_path, "喜欢小米，给我推荐手机", old=True, edit=edit)
    confirmed = session.confirm(
        preview.ticket_id,
        reply=reply("是我爸喜欢"),
        subject="other",
        scope="current",
        person_reference=EvidenceSpan("R1", 1, 3),
        as_of=LATER,
    )
    assert confirmed.proposal.decision.beneficiary_id == "self"
    assert confirmed.proposal.decision.excluded_entities == ("小米",)
    assert confirmed.proposal.proposed_memories == ()


def test_brand_view_request_cannot_be_confirmed_into_long_term_preference(tmp_path):
    def edit(data, turns):
        data["assertions"] = []
        data["request"].update(
            is_recommendation=True,
            beneficiary=subject("unknown"),
            requested_entity="小米",
            evidence=[span(turns[0])],
        )

    session, store, preview = setup(tmp_path, "这次看看小米", old=True, edit=edit)
    assert preview.action == "clarify"
    with pytest.raises(CommitError, match="clarification_target"):
        session.confirm(
            preview.ticket_id, reply=reply(), subject="self", scope="long_term", as_of=LATER
        )
    current = session.confirm(
        preview.ticket_id,
        reply=reply("是我自己，只看看"),
        subject="self",
        scope="current",
        as_of=LATER,
    )
    assert current.action == "current" and current.proposal.proposed_memories == ()
    assert current.proposal.decision.excluded_entities == ()
    assert store.active_memories("demo-user", as_of=LATER)[0].relation == "exclude"


@pytest.mark.parametrize("operation", ["commit", "confirm"])
def test_external_deletion_invalidates_pending_ticket(tmp_path, operation):
    session, store, preview = ambiguous(tmp_path) if operation == "confirm" else setup(tmp_path)
    store.delete_entity(
        "demo-user", category="phone", entity="小米", deleted_at=NOW + timedelta(minutes=12)
    )
    before = file_for(tmp_path).read_bytes()
    with pytest.raises(CommitError, match="stale"):
        if operation == "commit":
            session.commit(preview.ticket_id, as_of=LATER)
        else:
            session.confirm(
                preview.ticket_id, reply=reply(), subject="self", scope="long_term", as_of=LATER
            )
    assert file_for(tmp_path).read_bytes() == before


def test_duplicate_after_delete_does_not_restore_applied_records(tmp_path):
    session, store, preview = setup(tmp_path)
    session.commit(preview.ticket_id, as_of=LATER)
    store.delete_entity("demo-user", category="phone", entity="小米", deleted_at=LATER)
    with pytest.raises(CommitError, match="stale"):
        session.commit(preview.ticket_id, as_of=LATER)
    assert store.list_memories("demo-user", as_of=LATER) == ()


@pytest.mark.parametrize("owner", ["another-user", "demo-user"])
def test_ticket_is_not_accepted_by_another_session_instance(tmp_path, owner):
    session, store, preview = setup(tmp_path)
    another = MemoryCommitSession(store, user_id=owner, session_id="other-session")
    with pytest.raises(CommitError, match="ticket"):
        another.commit(preview.ticket_id, as_of=LATER)
    assert store.list_memories("demo-user", as_of=LATER) == ()


def test_root_rebinding_is_rejected_without_creating_other_store(tmp_path):
    session, store, preview = setup(tmp_path / "original")
    store.root = tmp_path / "other"
    with pytest.raises(CommitError, match="binding"):
        session.commit(preview.ticket_id, as_of=LATER)
    assert not store.root.exists()


def test_cancel_and_new_preview_each_invalidate_previous_ticket(tmp_path):
    session, store, old = setup(tmp_path)
    turns = turns_for("以后排除小米")
    new = session.prepare(decode(output_for(turns[0]), turns), turns=turns, as_of=AS_OF)
    with pytest.raises(CommitError, match="ticket"):
        session.commit(old.ticket_id, as_of=LATER)
    session.cancel(new.ticket_id)
    with pytest.raises(CommitError, match="ticket"):
        session.commit(new.ticket_id, as_of=LATER)


@pytest.mark.parametrize(
    "changes",
    [
        {"role": "assistant"},
        {"turn_id": "T1"},
        {"occurred_at": NOW, "available_at": NOW},
        {"occurred_at": LATER, "available_at": LATER},
    ],
)
def test_confirmation_requires_new_available_user_reply_after_question(tmp_path, changes):
    session, store, preview = ambiguous(tmp_path)
    before = file_for(tmp_path).read_bytes()
    with pytest.raises(CommitError, match="reply"):
        session.confirm(
            preview.ticket_id,
            reply=reply(**changes),
            subject="self",
            scope="long_term",
            as_of=LATER,
        )
    assert file_for(tmp_path).read_bytes() == before


@pytest.mark.parametrize(
    "kwargs",
    [
        {"subject": "other", "scope": "long_term", "person_reference": EvidenceSpan("R1", 1, 3)},
        {"subject": "other", "scope": "current"},
        {"subject": "other", "scope": "current", "person_reference": EvidenceSpan("T1", 0, 1)},
        {"subject": "unknown", "scope": "long_term"},
    ],
)
def test_confirmation_does_not_create_long_term_other_or_unanchored_subject(tmp_path, kwargs):
    session, store, preview = ambiguous(tmp_path, unknown=True)
    with pytest.raises(CommitError):
        session.confirm(preview.ticket_id, reply=reply("是我爸喜欢"), as_of=LATER, **kwargs)
    assert store.active_memories("demo-user", as_of=LATER)[0].relation == "exclude"


def test_one_confirmation_cannot_resolve_multiple_targets_or_opposite_attitudes(tmp_path):
    def edit(data, turns):
        data["assertions"][0]["scope"] = "unclear"
        data["assertions"].append(data["assertions"][0] | {"relation": "prefer"})

    session, _, preview = setup(tmp_path, edit=edit)
    assert preview.action == "clarify"
    with pytest.raises(CommitError, match="clarification_target"):
        session.confirm(
            preview.ticket_id, reply=reply(), subject="self", scope="long_term", as_of=LATER
        )


def test_multiple_clear_preferences_commit_in_one_atomic_batch(tmp_path, monkeypatch):
    def edit(data, turns):
        data["assertions"].append(data["assertions"][0] | {"entity": "三星", "relation": "prefer"})

    session, store, preview = setup(tmp_path, "以后排除小米，喜欢三星", edit=edit)
    assert len(preview.proposal.proposed_memories) == 2

    def fail(*args):
        raise OSError("合成发布失败")

    with monkeypatch.context() as patch:
        patch.setattr("memrecagent.memory_store.os.replace", fail)
        with pytest.raises(CommitError, match="storage"):
            session.commit(preview.ticket_id, as_of=LATER)
    assert store.list_memories("demo-user", as_of=LATER) == ()
    session.commit(preview.ticket_id, as_of=LATER)
    assert {
        (m.entity, m.relation)
        for m in JsonlMemoryStore(tmp_path).active_memories("demo-user", as_of=LATER)
    } == {("小米", "exclude"), ("三星", "prefer")}


def test_one_confirmation_cannot_merge_self_and_other_attitude_evidence(tmp_path):
    def edit(data, turns):
        data["assertions"][0].update(relation="prefer", scope="unclear")
        data["assertions"].append(
            data["assertions"][0]
            | {
                "subject": subject("other", [span(turns[0], "我爸")]),
                "scope": "current",
            }
        )

    session, store, preview = setup(tmp_path, "我爸喜欢小米，我也喜欢小米", old=True, edit=edit)
    assert preview.action == "clarify"
    before = file_for(tmp_path).read_bytes()
    with pytest.raises(CommitError, match="clarification_target"):
        session.confirm(
            preview.ticket_id, reply=reply(), subject="self", scope="long_term", as_of=LATER
        )
    assert file_for(tmp_path).read_bytes() == before


def test_scope_confirmation_does_not_rewrite_unrelated_explicit_long_term_assertion(tmp_path):
    def edit(data, turns):
        data["assertions"][0].update(entity="三星")
        data["request"].update(
            is_recommendation=True,
            requested_entity="小米",
            change_scope="unclear",
            evidence=[span(turns[0])],
        )

    session, store, preview = setup(tmp_path, "以后排除三星，看看小米", old=True, edit=edit)
    assert preview.action == "clarify"
    with pytest.raises(CommitError, match="clarification_target"):
        session.confirm(
            preview.ticket_id,
            reply=reply("只是这次看看"),
            subject="self",
            scope="current",
            as_of=LATER,
        )


def test_confirmation_cannot_silently_ignore_a_changed_explicit_request_subject(tmp_path):
    def edit(data, turns):
        data["assertions"] = []
        data["request"].update(
            is_recommendation=True,
            requested_entity="小米",
            change_scope="unclear",
            evidence=[span(turns[0])],
        )

    session, store, preview = setup(tmp_path, "给我看看小米", old=True, edit=edit)
    assert preview.action == "clarify"
    with pytest.raises(CommitError, match="clarification_target"):
        session.confirm(
            preview.ticket_id,
            reply=reply("改成替我爸看"),
            subject="other",
            scope="current",
            person_reference=EvidenceSpan("R1", 3, 5),
            as_of=LATER,
        )


@pytest.mark.parametrize("field", ["entity", "relation"])
def test_unresolved_other_part_of_turn_blocks_entire_commit(tmp_path, field):
    def edit(data, turns):
        data["assertions"][0]["evidence"] = [span(turns[0], "以后排除小米")]
        data["unresolved"] = [{"field": field, "evidence": [span(turns[0], "另外那个品牌也不要")]}]

    session, store, preview = setup(tmp_path, "以后排除小米，另外那个品牌也不要", edit=edit)
    assert preview.action == "clarify"
    assert preview.proposal.proposed_memories == ()
    with pytest.raises(CommitError, match="confirmation_required"):
        session.commit(preview.ticket_id, as_of=LATER)
    assert store.list_memories("demo-user", as_of=LATER) == ()
    with pytest.raises(CommitError, match="clarification_target"):
        session.confirm(
            preview.ticket_id, reply=reply(), subject="self", scope="long_term", as_of=LATER
        )
