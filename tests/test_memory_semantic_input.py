"""已接受人物来源模板在模型边界透传，不自动合并人物或填偏好。"""

import json
from dataclasses import replace
from datetime import timedelta

import pytest
from test_memory_analyst import task_for
from test_memory_interaction import turn, value
from test_memory_interaction_extraction import decode, quote

from memrecagent.deepseek_client import ModelReply
from memrecagent.memory_analyst import analyze_draft, build_analyst_messages
from memrecagent.memory_extraction import EvidenceSpan, ExtractionError, Subject
from memrecagent.memory_interaction import ConversationMemorySession
from memrecagent.memory_interaction_extraction import build_interaction_messages
from memrecagent.memory_store import JsonlMemoryStore


def registered(first):
    start = first.content.index("我舅舅")
    return Subject("other", (EvidenceSpan(first.turn_id, start, start + 3),))


def payload(messages):
    return json.loads(messages[-1]["content"])["output_contract"]


def test_known_person_is_original_user_source_not_latest_same_name():
    first = turn(1, "我舅舅一直喜欢荣耀。")
    latest = turn(2, "给我舅舅推荐手机。")
    contract = payload(
        build_interaction_messages((first, latest), (), known_subjects=(registered(first),))
    )
    assert contract["known_other_subjects"] == [
        {"kind": "other", "references": [quote(first, "我舅舅")]}
    ]
    assert "relation" not in str(contract["known_other_subjects"])
    assert contract["latest_user_evidence_quote"] == quote(latest)


@pytest.mark.parametrize(
    "bad", ["self", "unknown", "empty", "assistant", "latest", "missing", "ambiguous", "non_tuple"]
)
def test_illegal_known_person_source_rejected_before_model(bad):
    first = turn(1, "我舅舅一直喜欢荣耀。")
    latest = turn(2, "给我舅舅推荐手机。")
    person = registered(first)
    if bad in ("self", "unknown"):
        person = Subject(bad, ())
    elif bad == "empty":
        person = Subject("other", ())
    elif bad == "assistant":
        first = replace(first, role="assistant")
    elif bad == "latest":
        person = registered(latest)
    elif bad == "missing":
        person = Subject("other", (EvidenceSpan("missing", 0, 3),))
    elif bad == "ambiguous":
        first = replace(first, content="我舅舅说我舅舅喜欢荣耀。")
    known = [person] if bad == "non_tuple" else (person,)
    with pytest.raises(ExtractionError):
        build_interaction_messages((first, latest), (), known_subjects=known)


def test_default_empty_and_two_stage_payloads_agree():
    first = turn(1, "我舅舅一直喜欢荣耀。")
    latest = turn(2, "给我舅舅推荐手机。")
    people = (registered(first),)
    direct = payload(build_interaction_messages((first, latest), (), known_subjects=people))
    seen = []

    def complete(messages):
        seen.append(payload(messages))
        return ModelReply(json.dumps(value(latest, "consult", entity=None)), "fake", None, {}, 0)

    analyze_draft((first, latest), complete=complete, known_subjects=people)
    task = replace(task_for(latest), turns=(first, latest), known_subjects=people)
    final = payload(build_analyst_messages(task))
    assert direct == seen[0] == final
    assert payload(build_interaction_messages((latest,), ()))["known_other_subjects"] == []


@pytest.mark.parametrize("mode", ["direct", "analyst"])
@pytest.mark.parametrize("clue", [False, True])
def test_only_accepted_person_templates_flow_to_both_modes(tmp_path, mode, clue):
    host = ConversationMemorySession(
        JsonlMemoryStore(tmp_path), user_id="synthetic", session_id="s"
    )
    first = turn(1, "我舅舅一直喜欢荣耀。" if not clue else "我舅舅说荣耀这台用着顺手。")
    data = value(
        first,
        "share_experience" if clue else "express_preference",
        entity="荣耀",
        relation=None if clue else "prefer",
        clue=clue,
    )
    row = data["clues"][0] if clue else data["candidate"]["assertions"][0]
    row["subject"] = {"kind": "other", "references": [quote(first, "我舅舅")]}
    seen = []

    def complete(messages):
        seen.append(payload(messages))
        if len(host.turns) == 1:
            result = data
        else:
            result = value(host.turns[-1], "consult", entity=None)
        return ModelReply(json.dumps(result, ensure_ascii=False), "fake", None, {}, 0)

    preview = host.prepare_user(
        first,
        complete=complete,
        as_of=first.available_at + timedelta(seconds=1),
        analysis_mode=mode,
    )
    assert all(not p["known_other_subjects"] for p in seen)
    host.commit(preview.ticket_id, as_of=first.available_at + timedelta(seconds=1))
    latest = turn(2, "再谈谈其他手机。")
    seen.clear()
    host.prepare_user(
        latest,
        complete=complete,
        as_of=latest.available_at + timedelta(seconds=1),
        analysis_mode=mode,
    )
    assert len(seen) == (2 if mode == "analyst" else 1)
    assert all(
        p["known_other_subjects"] == [{"kind": "other", "references": [quote(first, "我舅舅")]}]
        for p in seen
    )
    new = ConversationMemorySession(host._store, user_id="synthetic", session_id="new")
    seen.clear()
    new.prepare_user(
        first,
        complete=lambda messages: (
            seen.append(payload(messages))
            or ModelReply(json.dumps(value(first, "consult", entity=None)), "fake", None, {}, 0)
        ),
        as_of=first.available_at + timedelta(seconds=1),
        analysis_mode=mode,
    )
    assert all(not p["known_other_subjects"] for p in seen)


def test_long_term_other_assertion_has_direction_but_no_own_disk_write(tmp_path):
    first = turn(1, "我舅舅一直喜欢荣耀，不代表我也喜欢。")
    data = value(first, entity="荣耀", relation="prefer")
    data["candidate"]["assertions"][0]["subject"] = {
        "kind": "other",
        "references": [quote(first, "我舅舅")],
    }
    parsed = decode(data, (first,))
    assert parsed.candidate.assertions[0].scope == "long_term"
    store = JsonlMemoryStore(tmp_path)
    host = ConversationMemorySession(store, user_id="synthetic", session_id="s")
    p = host.prepare_user(
        first,
        complete=lambda _: ModelReply(json.dumps(data), "fake", None, {}, 0),
        as_of=first.available_at + timedelta(seconds=1),
    )
    host.commit(p.ticket_id, as_of=first.available_at + timedelta(seconds=1))
    assert not store.list_memories("synthetic", as_of=first.available_at + timedelta(hours=1))


@pytest.mark.parametrize("mode", ["direct", "analyst"])
def test_cancelled_preview_does_not_register_person(tmp_path, mode):
    host = ConversationMemorySession(
        JsonlMemoryStore(tmp_path), user_id="synthetic", session_id="s"
    )
    first = turn(1, "我舅舅一直喜欢荣耀。")
    data = value(first, entity="荣耀", relation="prefer")
    data["candidate"]["assertions"][0]["subject"] = {
        "kind": "other",
        "references": [quote(first, "我舅舅")],
    }
    p = host.prepare_user(
        first,
        complete=lambda _: ModelReply(json.dumps(data), "fake", None, {}, 0),
        as_of=first.available_at + timedelta(seconds=1),
        analysis_mode=mode,
    )
    host.cancel(p.ticket_id)
    seen = []
    latest = turn(2, "今天聊点别的。")
    host.prepare_user(
        latest,
        complete=lambda m: (
            seen.append(payload(m))
            or ModelReply(json.dumps(value(latest, "unrelated")), "fake", None, {}, 0)
        ),
        as_of=latest.available_at + timedelta(seconds=1),
        analysis_mode=mode,
    )
    assert all(p["known_other_subjects"] == [] for p in seen)


@pytest.mark.parametrize("mode", ["direct", "analyst"])
def test_external_change_discards_person_templates_before_next_send(tmp_path, mode):
    from test_memory import sample_memory

    from memrecagent.memory_commit import CommitError

    store = JsonlMemoryStore(tmp_path)
    host = ConversationMemorySession(store, user_id="synthetic", session_id="s")
    first = turn(1, "我舅舅一直喜欢荣耀。")
    data = value(first, entity="荣耀", relation="prefer")
    data["candidate"]["assertions"][0]["subject"] = {
        "kind": "other",
        "references": [quote(first, "我舅舅")],
    }
    p = host.prepare_user(
        first,
        complete=lambda _: ModelReply(json.dumps(data), "fake", None, {}, 0),
        as_of=first.available_at + timedelta(seconds=1),
        analysis_mode=mode,
    )
    host.commit(p.ticket_id, as_of=first.available_at + timedelta(seconds=1))
    store.add(sample_memory(user_id="synthetic", entity="苹果"))
    latest = turn(2, "今天聊点别的。")
    seen = []

    def complete(messages):
        seen.append(payload(messages))
        return ModelReply(json.dumps(value(latest, "unrelated")), "fake", None, {}, 0)

    with pytest.raises(CommitError, match="stale"):
        host.prepare_user(
            latest,
            complete=complete,
            as_of=latest.available_at + timedelta(seconds=1),
            analysis_mode=mode,
        )
    assert not seen and not host.turns
    host.prepare_user(
        latest,
        complete=complete,
        as_of=latest.available_at + timedelta(seconds=1),
        analysis_mode=mode,
    )
    assert all(p["known_other_subjects"] == [] for p in seen)


def test_two_similar_people_are_not_merged_by_display_name():
    first = turn(1, "朋友甲有荣耀，朋友乙有苹果。")
    latest = turn(2, "还没指定推荐给谁。")
    subjects = tuple(
        Subject("other", (EvidenceSpan("T1", start, start + 3),))
        for start in (first.content.index("朋友甲"), first.content.index("朋友乙"))
    )
    contract = payload(build_interaction_messages((first, latest), (), known_subjects=subjects))
    assert contract["known_other_subjects"] == [
        {"kind": "other", "references": [quote(first, "朋友甲")]},
        {"kind": "other", "references": [quote(first, "朋友乙")]},
    ]
