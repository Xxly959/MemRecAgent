"""验证ISS-038：宿主接受后的稳定他人偏好仅在本会话接续。"""

import json
from copy import deepcopy
from datetime import timedelta

import pytest
from test_memory_extraction import NOW, output_for, span, subject, turns_for

from memrecagent.memory_commit import CommitError, MemoryCommitSession
from memrecagent.memory_extraction import decode_candidate
from memrecagent.memory_store import JsonlMemoryStore


def prepare(
    host,
    turns,
    *,
    relation=None,
    scope="long_term",
    person="我爸",
    anchor=None,
    requested=None,
    asking=False,
    unresolved=False,
    category="phone",
):
    current = turns[-1]
    source = next(t for t in turns if person in t.content) if person != "self" else None
    who = subject("self") if source is None else subject("other", [span(source, anchor or person)])
    data = output_for(current)
    data["assertions"] = []
    if relation:
        data["assertions"] = [
            {
                "subject": who,
                "category": category,
                "entity": "小米",
                "relation": relation,
                "scope": scope,
                "evidence": [span(current)],
            }
        ]
    data["request"].update(
        is_recommendation=asking,
        beneficiary=who,
        category=category,
        requested_entity=requested,
        evidence=[span(current)] if asking else [],
    )
    if unresolved:
        data["unresolved"] = [{"field": "scope", "evidence": [span(current)]}]
    return host.prepare(
        decode_candidate(json.dumps(data), turns=turns),
        turns=turns,
        as_of=current.available_at + timedelta(seconds=1),
    )


def accepted(host, preview, turns):
    return host.commit(preview.ticket_id, as_of=turns[-1].available_at + timedelta(seconds=1))


def fixture(tmp_path, *, relation="prefer", scope="long_term"):
    store = JsonlMemoryStore(tmp_path)
    host = MemoryCommitSession(store, user_id="synthetic-user", session_id="S1")
    turns = turns_for("我爸长期喜欢小米手机。", "这次给我爸推荐手机。")
    first = prepare(host, turns[:1], relation=relation, scope=scope)
    return host, store, turns, first


def used(preview):
    return [(r.entity, r.relation) for r in preview.proposal.decision.context_preferences]


def test_committed_stable_other_preference_is_used_in_later_request_without_disk_write(tmp_path):
    host, store, turns, first = fixture(tmp_path)
    assert used(first) == []
    assert accepted(host, first, turns[:1]).status == "not_applied"
    second = prepare(host, turns, asking=True)
    assert second.action == "current"
    assert used(second) == [("小米", "prefer")]
    assert second.proposal.session_preferences == ()
    assert store.list_memories("synthetic-user", as_of=NOW + timedelta(hours=1)) == ()


@pytest.mark.parametrize("cancel", [False, True])
def test_uncommitted_or_cancelled_preview_does_not_enter_future_session_state(tmp_path, cancel):
    host, _, turns, first = fixture(tmp_path)
    if cancel:
        host.cancel(first.ticket_id)
    assert used(prepare(host, turns, asking=True)) == []


def test_other_explicit_current_preference_is_not_carried_to_later_request(tmp_path):
    host, _, turns, first = fixture(tmp_path, scope="current")
    accepted(host, first, turns[:1])
    assert used(prepare(host, turns, asking=True)) == []


def test_latest_accepted_stable_attitude_replaces_prior_attitude(tmp_path):
    host, _, _, _ = fixture(tmp_path)
    turns = turns_for("我爸长期喜欢小米手机。", "我爸以后长期排除小米手机。", "给我爸推荐手机。")
    first = prepare(host, turns[:1], relation="prefer")
    accepted(host, first, turns[:1])
    change = prepare(host, turns[:2], relation="exclude")
    accepted(host, change, turns[:2])
    result = prepare(host, turns, asking=True)
    assert used(result) == [("小米", "exclude")]
    assert result.proposal.decision.excluded_entities == ("小米",)


def test_new_person_does_not_use_other_persons_context(tmp_path):
    host, _, _, _ = fixture(tmp_path)
    turns = turns_for("我爸长期喜欢小米手机。", "这次给我哥推荐手机。", "给我爸推荐手机。")
    first = prepare(host, turns[:1], relation="prefer")
    accepted(host, first, turns[:1])
    assert used(prepare(host, turns[:2], person="我哥", asking=True)) == []
    assert used(prepare(host, turns, asking=True)) == [("小米", "prefer")]


@pytest.mark.parametrize(
    "user,session", [("synthetic-user", "S2"), ("different-user", "S1"), ("synthetic-user", "S1")]
)
def test_new_host_session_or_owner_never_inherits_in_memory_context(tmp_path, user, session):
    host, store, turns, first = fixture(tmp_path)
    accepted(host, first, turns[:1])
    other = MemoryCommitSession(store, user_id=user, session_id=session)
    assert used(prepare(other, turns, asking=True)) == []


def test_current_brand_request_temporarily_overrides_old_session_exclusion(tmp_path):
    host, _, _, _ = fixture(tmp_path)
    turns = turns_for("我爸长期排除小米手机。", "这次给我爸看看小米手机。", "再给我爸推荐手机。")
    first = prepare(host, turns[:1], relation="exclude")
    accepted(host, first, turns[:1])
    current = prepare(host, turns[:2], asking=True, requested="小米")
    assert current.action == "current" and current.proposal.decision.excluded_entities == ()
    accepted(host, current, turns[:2])
    later = prepare(host, turns, asking=True)
    assert later.proposal.decision.excluded_entities == ("小米",)


def test_temporary_other_attitude_only_overrides_for_current_request(tmp_path):
    host, _, _, _ = fixture(tmp_path)
    turns = turns_for(
        "我爸长期喜欢小米手机。", "这次替我爸选手机但只限本次排除小米。", "再给我爸推荐手机。"
    )
    first = prepare(host, turns[:1], relation="prefer")
    accepted(host, first, turns[:1])
    current = prepare(host, turns[:2], relation="exclude", scope="current", asking=True)
    assert current.proposal.decision.excluded_entities == ("小米",)
    accepted(host, current, turns[:2])
    assert used(prepare(host, turns, asking=True)) == [("小米", "prefer")]


def test_clarification_or_failed_commit_does_not_replace_accepted_context(tmp_path):
    host, _, _, _ = fixture(tmp_path)
    turns = turns_for(
        "我爸长期喜欢小米手机。", "我爸排除小米但没说本次还是长期。", "给我爸推荐手机。"
    )
    first = prepare(host, turns[:1], relation="prefer")
    accepted(host, first, turns[:1])
    pending = prepare(host, turns[:2], relation="exclude", scope="unclear", unresolved=True)
    assert pending.action == "clarify"
    with pytest.raises(CommitError, match="confirmation_required"):
        accepted(host, pending, turns[:2])
    assert used(prepare(host, turns, asking=True)) == [("小米", "prefer")]


def test_self_current_assertion_never_leaks_to_later_requests(tmp_path):
    host, store, _, _ = fixture(tmp_path)
    turns = turns_for("这次我排除小米。", "再给我推荐手机。")
    first = prepare(host, turns[:1], relation="exclude", scope="current", person="self")
    accepted(host, first, turns[:1])
    second = prepare(host, turns, asking=True, person="self")
    assert used(second) == [] and second.proposal.decision.excluded_entities == ()
    assert store.list_memories("synthetic-user", as_of=NOW + timedelta(hours=1)) == ()


def test_same_turn_exclusion_and_brand_request_still_requires_clarification(tmp_path):
    host, _, _, _ = fixture(tmp_path)
    turns = turns_for("我爸长期喜欢小米手机。", "给我爸看小米手机，但本次也排除小米。")
    first = prepare(host, turns[:1], relation="prefer")
    accepted(host, first, turns[:1])
    result = prepare(
        host, turns, relation="exclude", scope="current", requested="小米", asking=True
    )
    assert result.action == "clarify"
    with pytest.raises(CommitError, match="confirmation_required"):
        accepted(host, result, turns)


def test_stale_commit_cannot_publish_new_session_attitude(tmp_path):
    host, store, turns, first = fixture(tmp_path)
    store.delete_entity("synthetic-user", category="phone", entity="苹果", deleted_at=NOW)
    with pytest.raises(CommitError, match="stale"):
        accepted(host, first, turns[:1])
    assert used(prepare(host, turns, asking=True)) == []


def test_self_deletion_does_not_erase_independent_other_person_preference(tmp_path):
    host, store, turns, first = fixture(tmp_path)
    accepted(host, first, turns[:1])
    store.delete_entity("synthetic-user", category="phone", entity="小米", deleted_at=NOW)
    assert used(prepare(host, turns, asking=True)) == [("小米", "prefer")]
    assert store.list_memories("synthetic-user", as_of=NOW + timedelta(hours=1)) == ()


def test_root_rebinding_clears_accepted_context_even_if_original_root_is_restored(tmp_path):
    host, store, turns, first = fixture(tmp_path)
    accepted(host, first, turns[:1])
    root = store.root
    store.root = tmp_path / "different"
    with pytest.raises(CommitError, match="binding"):
        prepare(host, turns, asking=True)
    store.root = root
    assert used(prepare(host, turns, asking=True)) == []


def test_background_without_request_does_not_activate_accepted_context(tmp_path):
    host, _, turns, first = fixture(tmp_path)
    accepted(host, first, turns[:1])
    result = prepare(host, turns)
    assert result.action == "noop" and used(result) == []


@pytest.mark.parametrize("boundary", ["keep", "reset_session", "restart"])
def test_real_dialogue_runner_carries_accepted_context_and_clears_on_boundary(tmp_path, boundary):
    from test_memory_dialogue import FIXTURE, empty, faithful_fake, model

    from memrecagent.memory_dialogue import evaluate_dialogues, load_dialogues

    cases = deepcopy(load_dialogues(FIXTURE))
    first, later = cases[0]["steps"][1:3]
    first["text"] = "我爸长期排除小米手机，这次替我爸选手机。"
    first["expected"]["excluded"] = ["小米"]
    later["text"] = "再给我爸推荐手机。"
    later["reset_session"] = boundary == "reset_session"
    later["restart"] = boundary == "restart"
    later["expected"].update(
        actions=["current"] if boundary == "keep" else ["noop"],
        excluded=["小米"] if boundary == "keep" else [],
        beneficiary="other",
        person="我爸",
    )

    def complete(messages):
        # 合成客户端只解释本轮文字，不重发历史态度，也不读取验收答案。
        payload = json.loads(messages[1]["content"])
        if "reply" in payload:
            return faithful_fake(messages)
        turns = payload["turns"]
        current = turns[-1]
        if current["content"] not in (first["text"], later["text"]):
            return faithful_fake(messages)
        source = next(t for t in turns if "我爸" in t["content"])
        start = source["content"].index("我爸")
        who = subject("other", [{"turn_id": source["turn_id"], "start": start, "end": start + 2}])
        evidence = [{"turn_id": current["turn_id"], "start": 0, "end": len(current["content"])}]
        data = empty()
        if "长期排除" in current["content"]:
            data["assertions"] = [
                {
                    "subject": who,
                    "category": "phone",
                    "entity": "小米",
                    "relation": "exclude",
                    "scope": "long_term",
                    "evidence": evidence,
                }
            ]
        data["request"].update(
            is_recommendation=True, beneficiary=who, category="phone", evidence=evidence
        )
        return model(data)

    report = evaluate_dialogues(cases, complete=complete, output_dir=tmp_path / "run")
    assert report["model_calls"] == 20 and report["strict_steps"] == 18
    rows = [json.loads(s) for s in (tmp_path / "run/steps.jsonl").read_text().splitlines()]
    assert rows[2]["excluded"] == (["小米"] if boundary == "keep" else [])
    assert rows[1]["written"] == rows[2]["written"] == []


def test_other_category_does_not_activate_phone_context(tmp_path):
    host, _, turns, first = fixture(tmp_path)
    accepted(host, first, turns[:1])
    assert used(prepare(host, turns, asking=True, category="laptop")) == []
    assert used(prepare(host, turns, asking=True)) == [("小米", "prefer")]


def test_future_accepted_context_is_invisible_to_earlier_query(tmp_path):
    host, _, _, _ = fixture(tmp_path)
    turns = turns_for("我爸准备换手机。", "我爸长期喜欢小米手机。", "给我爸推荐手机。")
    first = prepare(host, turns[:2], relation="prefer")
    accepted(host, first, turns[:2])
    assert used(prepare(host, turns[:1], asking=True)) == []
    assert used(prepare(host, turns, asking=True)) == [("小米", "prefer")]


def test_accepted_neutral_cancels_old_other_exclusion_only_in_session(tmp_path):
    host, store, _, _ = fixture(tmp_path)
    turns = turns_for("我爸长期排除小米。", "我爸长期取消对小米的态度。", "给我爸推荐手机。")
    first = prepare(host, turns[:1], relation="exclude")
    accepted(host, first, turns[:1])
    change = prepare(host, turns[:2], relation="neutral")
    accepted(host, change, turns[:2])
    current = prepare(host, turns, asking=True)
    assert used(current) == [("小米", "neutral")]
    assert current.proposal.decision.excluded_entities == ()
    assert store.list_memories("synthetic-user", as_of=NOW + timedelta(hours=1)) == ()
