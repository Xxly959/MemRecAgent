"""使用临时文件验证只读提案，明确区分本人长期与他人会话。"""

from dataclasses import replace
from datetime import timedelta

import pytest
from test_memory import sample_memory
from test_memory_extraction import NOW, decode, output_for, span, subject, turns_for
from test_memory_store import file_for

from memrecagent.memory_extraction import ExtractionError
from memrecagent.memory_proposals import propose
from memrecagent.memory_store import JsonlMemoryStore

AS_OF = NOW + timedelta(days=1)


def test_deleted_evidence_replay_creates_neither_proposal_nor_session_preference(tmp_path):
    result, store, data, turns = run_proposal(tmp_path, ["以后排除小米"])
    store.add(result.proposed_memories[0])
    store.delete_entity("demo-user", category="phone", entity="小米", deleted_at=AS_OF)
    replay = propose(
        decode(data, turns),
        store=JsonlMemoryStore(tmp_path),
        user_id="demo-user",
        session_id="S2",
        turns=turns,
        people=(),
        as_of=AS_OF + timedelta(seconds=1),
    )
    assert replay.proposed_memories == ()
    assert replay.session_preferences == ()
    assert replay.decision.excluded_entities == ()


@pytest.mark.parametrize("merged_span", [False, True])
def test_deleted_old_evidence_cannot_be_laundered_by_new_matching_assertion(tmp_path, merged_span):
    turns = turns_for("以后排除小米，旧私密合成证据", "以后排除小米，这是新表达")
    old_data = output_for(turns[0])
    new_data = output_for(turns[1])
    store = JsonlMemoryStore(tmp_path)
    store.delete_entity(
        "demo-user",
        category="phone",
        entity="小米",
        deleted_at=NOW + timedelta(minutes=1, seconds=30),
    )
    if merged_span:
        new_data["assertions"][0]["evidence"].insert(0, span(turns[0]))
    else:
        new_data["assertions"].insert(0, old_data["assertions"][0])

    def preview():
        return propose(
            decode(new_data, turns),
            store=store,
            user_id="demo-user",
            session_id="S2",
            turns=turns,
            people=(),
            as_of=AS_OF,
        )

    if merged_span:
        with pytest.raises(ExtractionError, match="deleted_evidence"):
            preview()
    else:
        result = preview()
        assert len(result.proposed_memories) == 1
        store.add(result.proposed_memories[0])
        assert "旧私密合成证据" not in file_for(tmp_path).read_text()
        assert result.proposed_memories[0].evidence_text == "以后排除小米，这是新表达"


def test_old_request_cannot_override_new_preference_after_deletion(tmp_path):
    turns = turns_for("这次看看小米")
    data = output_for(turns[0])
    data["assertions"] = []
    data["request"].update(
        is_recommendation=True, requested_entity="小米", evidence=[span(turns[0])]
    )
    store = JsonlMemoryStore(tmp_path)
    store.delete_entity(
        "demo-user", category="phone", entity="小米", deleted_at=NOW + timedelta(hours=1)
    )
    store.add(
        sample_memory(
            memory_id="fresh",
            occurred_at=NOW + timedelta(hours=2),
            available_at=NOW + timedelta(hours=2),
        )
    )
    with pytest.raises(ExtractionError, match="deleted_request"):
        propose(
            decode(data, turns),
            store=store,
            user_id="demo-user",
            session_id="S2",
            turns=turns,
            people=(),
            as_of=AS_OF,
        )


def run_proposal(tmp_path, texts, edit=None, old=None, people=()):
    turns = turns_for(*texts)
    data = output_for(turns[-1])
    if edit:
        edit(data, turns)
    store = JsonlMemoryStore(tmp_path)
    if old:
        store.add(old)
    result = propose(
        decode(data, turns),
        store=store,
        user_id="demo-user",
        session_id="S2",
        turns=turns,
        people=people,
        as_of=AS_OF,
    )
    return result, store, data, turns


def test_other_like_self_request_does_not_personalize_from_other(tmp_path):
    def edit(data, turns):
        data["assertions"][0].update(
            relation="prefer", subject=subject("other", [span(turns[0], "我爸")])
        )
        data["request"].update(is_recommendation=True, evidence=[span(turns[0])])

    result, store, _, _ = run_proposal(
        tmp_path, ["我爸喜欢小米，给我推荐手机"], edit, sample_memory()
    )
    assert result.proposed_memories == ()
    assert result.decision.beneficiary_id == "self"
    assert result.decision.excluded_entities == ("小米",)
    assert result.decision.context_preferences == ()
    assert result.session_preferences[0].beneficiary_id != "self"
    assert len(store.list_memories("demo-user", as_of=AS_OF)) == 1


def test_self_dislike_other_request_not_filtered(tmp_path):
    def edit(data, turns):
        data["assertions"][0]["relation"] = "dislike"
        data["request"].update(
            is_recommendation=True,
            beneficiary=subject("other", [span(turns[0], "我爸")]),
            evidence=[span(turns[0])],
        )

    result, _, _, _ = run_proposal(tmp_path, ["我讨厌小米，给我爸推荐手机"], edit)
    assert [m.relation for m in result.proposed_memories] == ["dislike"]
    assert result.decision.excluded_entities == ()
    assert result.decision.memories == ()


def test_ambiguous_opposition_clarifies_and_does_not_propose_memory(tmp_path):
    def edit(data, turns):
        data["assertions"][0].update(subject=subject("unknown"), relation="prefer")
        data["request"].update(beneficiary=subject("unknown"))

    result, _, _, _ = run_proposal(tmp_path, ["喜欢小米"], edit, sample_memory())
    assert result.decision.status == "clarify"
    assert result.proposed_memories == ()


def test_unclear_attitude_subject_clarifies_even_when_request_is_for_self(tmp_path):
    def edit(data, turns):
        data["assertions"][0].update(subject=subject("unknown"), relation="prefer")
        data["request"].update(is_recommendation=True, evidence=[span(turns[0])])

    result, _, _, _ = run_proposal(tmp_path, ["给我推荐手机，喜欢小米"], edit, sample_memory())
    assert result.decision.status == "clarify"
    assert result.proposed_memories == ()


def test_subject_flag_inconsistent_with_confident_self_never_proposes(tmp_path):
    def edit(data, turns):
        data["unresolved"] = [{"field": "subject", "evidence": [span(turns[0])]}]

    with pytest.raises(ExtractionError):
        run_proposal(tmp_path, ["以后排除小米"], edit)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("field", ["relation", "entity"])
def test_unresolved_relation_or_entity_blocks_overlapping_confident_assertion(tmp_path, field):
    def edit(data, turns):
        data["unresolved"] = [{"field": field, "evidence": [span(turns[0])]}]

    with pytest.raises(ExtractionError):
        run_proposal(tmp_path, ["我对小米的长期态度还没想好"], edit)
    assert list(tmp_path.iterdir()) == []


def test_no_request_placeholder_scope_cannot_override_explicit_long_term_change(tmp_path):
    def edit(data, turns):
        data["request"].update(
            change_scope="unclear", requested_entity="小米", evidence=[span(turns[0])]
        )
        data["assertions"][0]["relation"] = "prefer"

    result, _, _, _ = run_proposal(
        tmp_path, ["我以后喜欢小米手机了，长期偏好改成喜欢"], edit, sample_memory()
    )
    assert result.decision.status == "ready"
    assert result.decision.requested_entity is None
    assert [m.relation for m in result.proposed_memories] == ["prefer"]


def test_replaying_superseded_evidence_does_not_reactivate_old_exclusion(tmp_path):
    result, store, data, turns = run_proposal(tmp_path, ["我以后排除小米"])
    original = result.proposed_memories[0]
    store.add(original)
    cancel = replace(
        original,
        memory_id="new-neutral",
        relation="neutral",
        supersedes=original.memory_id,
        occurred_at=original.occurred_at + timedelta(hours=1),
        available_at=original.available_at + timedelta(hours=1),
    )
    store.add(cancel)
    before = file_for(tmp_path).read_bytes()
    replay = propose(
        decode(data, turns),
        store=store,
        user_id="demo-user",
        session_id="S2",
        turns=turns,
        people=(),
        as_of=AS_OF,
    )
    assert replay.proposed_memories == ()
    assert replay.decision.excluded_entities == ()
    assert [m.relation for m in replay.decision.memories] == ["neutral"]
    assert file_for(tmp_path).read_bytes() == before


def test_context_only_scope_unclear_clarifies(tmp_path):
    def edit(data, turns):
        data["assertions"][0].update(relation="prefer", scope="unclear")

    result, _, _, _ = run_proposal(
        tmp_path, ["我好像喜欢小米，但不确定长期还是本次"], edit, sample_memory()
    )
    assert result.decision.status == "clarify"
    assert result.proposed_memories == ()


def test_neutral_revision_is_not_like_and_proposal_never_writes(tmp_path):
    def edit(data, turns):
        data["assertions"][0]["relation"] = "neutral"

    result, store, data, turns = run_proposal(
        tmp_path, ["我以后不排除小米了"], edit, sample_memory()
    )
    before = file_for(tmp_path).read_bytes()
    assert result.decision.status == "ready" and result.decision.excluded_entities == ()
    assert result.proposed_memories[0].relation == "neutral"
    assert result.proposed_memories[0].supersedes == "m1"
    again = propose(
        decode(data, turns),
        store=store,
        user_id="demo-user",
        session_id="S2",
        turns=turns,
        people=(),
        as_of=AS_OF,
    )
    assert again.proposed_memories == result.proposed_memories
    assert file_for(tmp_path).read_bytes() == before


def test_apply_from_trusted_caller_reopens_and_repeated_proposal_does_not_append(tmp_path):
    result, store, data, turns = run_proposal(tmp_path, ["以后别给我推荐小米"])
    assert not tmp_path.exists() or list(tmp_path.iterdir()) == []
    store.add(result.proposed_memories[0])
    reopened = JsonlMemoryStore(tmp_path)
    assert reopened.active_memories("demo-user", as_of=AS_OF)[0].relation == "exclude"
    assert (
        propose(
            decode(data, turns),
            store=reopened,
            user_id="demo-user",
            session_id="S2",
            turns=turns,
            people=(),
            as_of=AS_OF,
        ).proposed_memories
        == ()
    )


def test_duplicate_same_key_assertions_produce_one_writable_proposal(tmp_path):
    def edit(data, turns):
        data["assertions"].append(data["assertions"][0].copy())

    result, store, _, _ = run_proposal(tmp_path, ["我以后排除小米"], edit)
    assert len(result.proposed_memories) == 1
    for memory in result.proposed_memories:
        store.add(memory)
    assert len(store.list_memories("demo-user", as_of=AS_OF)) == 1


def test_same_attitude_evidence_fragments_merge_into_one_proposal(tmp_path):
    def edit(data, turns):
        data["assertions"][0]["evidence"] = [span(turns[0], "我排除小米")]
        data["assertions"].append(
            data["assertions"][0] | {"evidence": [span(turns[0], "以后排除小米")]}
        )

    result, _, _, _ = run_proposal(tmp_path, ["我排除小米，以后排除小米"], edit)
    assert len(result.proposed_memories) == 1
    assert "我排除小米" in result.proposed_memories[0].evidence_text
    assert "以后排除小米" in result.proposed_memories[0].evidence_text


def test_current_request_temporarily_overrides_and_comparison_does_not_write(tmp_path):
    def edit(data, turns):
        data["assertions"] = []
        data["request"].update(
            is_recommendation=True, requested_entity="小米", evidence=[span(turns[0])]
        )

    result, store, _, _ = run_proposal(tmp_path, ["这次我想看看小米"], edit, sample_memory())
    assert result.decision.overridden_memory_ids == ("m1",)
    assert result.proposed_memories == ()
    assert store.active_memories("demo-user", as_of=AS_OF)[0].relation == "exclude"


def test_same_pronoun_uses_explicit_antecedent_and_session_people_remain_separate(tmp_path):
    def edit(data, turns):
        data["assertions"][0].update(
            relation="prefer", subject=subject("other", [span(turns[0], "我爸")])
        )
        data["request"].update(
            is_recommendation=True,
            beneficiary=subject("other", [span(turns[0], "我爸")]),
            evidence=[span(turns[-1])],
        )

    result, store, data, turns = run_proposal(
        tmp_path, ["我爸和我哥都想换手机", "他喜欢小米，推荐一款"], edit
    )
    assert result.decision.context_preferences[0] == result.session_preferences[0]
    assert len(result.people) == 1
    data["assertions"][0]["subject"]["references"] = [span(turns[0], "我哥")]
    data["request"]["beneficiary"]["references"] = [span(turns[0], "我哥")]
    second = propose(
        decode(data, turns),
        store=store,
        user_id="demo-user",
        session_id="S2",
        turns=turns,
        people=result.people,
        as_of=AS_OF,
    )
    assert len(second.people) == 2
    assert second.decision.beneficiary_id != result.decision.beneficiary_id


@pytest.mark.parametrize("mode", ["future", "old_revision", "two_opposed", "scope_unresolved"])
def test_unsafe_or_conflicting_proposals_never_write(tmp_path, mode):
    def edit(data, turns):
        data["assertions"][0]["relation"] = "prefer"
        if mode == "two_opposed":
            data["assertions"].append(data["assertions"][0] | {"relation": "exclude"})
        if mode == "scope_unresolved":
            data["unresolved"] = [{"field": "scope", "evidence": [span(turns[0])]}]

    old = (
        sample_memory(occurred_at=NOW + timedelta(hours=1), available_at=NOW + timedelta(hours=1))
        if mode == "old_revision"
        else sample_memory()
    )
    store = JsonlMemoryStore(tmp_path)
    store.add(old)
    before = file_for(tmp_path).read_bytes()
    turns = turns_for("我以后喜欢小米")
    data = output_for(turns[0])
    edit(data, turns)
    kwargs = dict(
        store=store,
        user_id="demo-user",
        session_id="S2",
        turns=turns,
        people=(),
        as_of=NOW if mode == "future" else AS_OF,
    )
    if mode in {"future", "old_revision"}:
        with pytest.raises(ExtractionError):
            propose(decode(data, turns), **kwargs)
    else:
        result = propose(decode(data, turns), **kwargs)
        assert result.decision.status == "clarify" and result.proposed_memories == ()
    assert file_for(tmp_path).read_bytes() == before
