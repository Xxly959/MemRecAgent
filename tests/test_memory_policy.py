"""人物由调用方提供：测试策略边界，不冒充自然语言主语识别评测。"""

from datetime import timedelta

import pytest
from test_memory import T0, sample_memory
from test_memory_store import AFTER, file_for, revision

from memrecagent.memory_policy import ContextPreference, resolve_recommendation
from memrecagent.memory_store import JsonlMemoryStore


def relative_preference(**changes):
    fields = dict(
        user_id="demo-user",
        session_id="S2",
        beneficiary_id="relative:uncle",
        category="phone",
        entity="小米",
        relation="prefer",
        evidence_ref="S2:T1",
        evidence_text="给亲戚买，他喜欢小米",
        available_at=T0 + timedelta(hours=1),
    )
    return ContextPreference(**(fields | changes))


def resolve(store, **changes):
    fields = dict(
        user_id="demo-user", session_id="S2", beneficiary_id="self", category="phone", as_of=AFTER
    )
    return resolve_recommendation(store, **(fields | changes))


def test_self_explicit_request_temporarily_overrides_old_negative_without_writing(tmp_path):
    store = JsonlMemoryStore(tmp_path)
    old = sample_memory()
    store.add(old)
    before = file_for(tmp_path).read_bytes()
    decision = resolve(store, requested_entity="小米")
    assert decision.status == "ready" and decision.clarification is None
    assert decision.requested_entity == "小米"
    assert decision.excluded_entities == ()
    assert decision.overridden_memory_ids == ("m1",)
    assert resolve(store).excluded_entities == ("小米",)
    assert file_for(tmp_path).read_bytes() == before


def test_relative_preference_stays_in_current_context_and_never_changes_user_memory(tmp_path):
    store = JsonlMemoryStore(tmp_path)
    old = sample_memory()
    store.add(old)
    context = relative_preference()
    before = file_for(tmp_path).read_bytes()
    decision = resolve(store, beneficiary_id="relative:uncle", context_preferences=(context,))
    assert decision.status == "ready"
    assert decision.memories == () and decision.excluded_entities == ()
    assert decision.context_preferences == (context,)
    assert resolve(store).excluded_entities == ("小米",)
    assert (
        resolve(store, session_id="S3", beneficiary_id="relative:uncle").context_preferences == ()
    )
    with pytest.raises(ValueError):
        store.add(context)
    assert store.list_memories("demo-user", as_of=AFTER) == (old,)
    assert file_for(tmp_path).read_bytes() == before


def test_unclear_subject_and_conflicting_request_asks_instead_of_guessing(tmp_path):
    store = JsonlMemoryStore(tmp_path)
    store.add(sample_memory())
    before = file_for(tmp_path).read_bytes()
    decision = resolve(store, beneficiary_id=None, requested_entity="小米")
    assert decision.status == "clarify" and decision.clarification
    assert decision.memories == () and decision.context_preferences == ()
    assert file_for(tmp_path).read_bytes() == before


def test_unclear_subject_without_clear_opposition_continues_without_personalizing(tmp_path):
    store = JsonlMemoryStore(tmp_path)
    store.add(sample_memory(entity="三星", relation="prefer"))
    decision = resolve(store, beneficiary_id=None, requested_entity="小米")
    assert decision.status == "ready" and decision.clarification is None
    assert decision.beneficiary_id is None
    assert decision.memories == () and decision.excluded_entities == ()


def test_ambiguous_negative_statement_opposing_known_like_also_asks(tmp_path):
    store = JsonlMemoryStore(tmp_path)
    store.add(sample_memory(relation="prefer"))
    context = relative_preference(beneficiary_id=None, relation="dislike")
    decision = resolve(store, beneficiary_id=None, context_preferences=(context,))
    assert decision.status == "clarify"


def test_explicit_long_term_revision_removes_exclusion_without_inventing_like(tmp_path):
    store = JsonlMemoryStore(tmp_path)
    old = sample_memory()
    store.add(old)
    store.add(revision(old))
    decision = resolve(store)
    assert decision.excluded_entities == ()
    assert [m.relation for m in decision.memories] == ["neutral"]
    earlier = resolve(store, as_of=T0 + timedelta(hours=1))
    assert earlier.excluded_entities == ("小米",)


def test_uncertain_temporary_versus_permanent_change_asks_and_does_not_write(tmp_path):
    store = JsonlMemoryStore(tmp_path)
    store.add(sample_memory())
    decision = resolve(store, requested_entity="小米", change_scope="unclear")
    assert decision.status == "clarify"
    assert len(store.list_memories("demo-user", as_of=AFTER)) == 1


@pytest.mark.parametrize(
    ("old_relation", "new_relation"), [("exclude", "prefer"), ("prefer", "exclude")]
)
def test_context_only_change_with_unclear_scope_asks_before_overriding(
    tmp_path, old_relation, new_relation
):
    store = JsonlMemoryStore(tmp_path)
    store.add(sample_memory(relation=old_relation))
    before = file_for(tmp_path).read_bytes()
    context = relative_preference(beneficiary_id="self", relation=new_relation)
    decision = resolve(store, context_preferences=(context,), change_scope="unclear")
    assert decision.status == "clarify" and decision.clarification
    assert decision.memories == () and decision.context_preferences == ()
    assert file_for(tmp_path).read_bytes() == before


@pytest.mark.parametrize("changes", [{"user_id": "someone-else"}, {"session_id": "S3"}])
def test_other_user_or_session_context_is_rejected(tmp_path, changes):
    context = relative_preference(**changes)
    with pytest.raises(ValueError):
        resolve(
            JsonlMemoryStore(tmp_path),
            beneficiary_id="relative:uncle",
            context_preferences=(context,),
        )


def test_future_context_is_not_used_and_other_person_is_not_applied_to_self(tmp_path):
    store = JsonlMemoryStore(tmp_path)
    context = relative_preference(available_at=AFTER)
    assert (
        resolve(
            store, beneficiary_id="relative:uncle", context_preferences=(context,)
        ).context_preferences
        == ()
    )
    assert resolve(store, context_preferences=(relative_preference(),)).context_preferences == ()


def test_contradictory_same_time_context_is_clarified_instead_of_input_order_wins(tmp_path):
    first = relative_preference()
    second = relative_preference(relation="exclude")
    decision = resolve(
        JsonlMemoryStore(tmp_path),
        beneficiary_id="relative:uncle",
        context_preferences=(first, second),
    )
    assert decision.status == "clarify"
