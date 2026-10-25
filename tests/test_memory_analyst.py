"""专职角色的离线契约测试；固定模型返回不代表自然语言准确率。"""

import hashlib
import json
from dataclasses import FrozenInstanceError, replace

import pytest
from test_memory import sample_memory
from test_memory_extraction import subject
from test_memory_interaction import after, turn, value
from test_memory_interaction_extraction import envelope, quote

from memrecagent.deepseek_client import ModelReply, ProviderError
from memrecagent.memory_analyst import (
    AnalystError,
    AnalystTask,
    MemoryAnalyst,
    MemoryView,
    build_analyst_messages,
    build_memory_view,
)
from memrecagent.memory_interaction_extraction import decode_interaction


def raw(data):
    return json.dumps(data, ensure_ascii=False)


def memory(entity="小米", **changes):
    return sample_memory(**(dict(user_id="u", entity=entity) | changes))


def view_for(t, data, memories=()):
    return build_memory_view(
        user_id="u",
        session_id="s",
        as_of=after(t),
        snapshot_token="snapshot-private",
        active_memories=memories,
        draft=decode_interaction(raw(data), turns=(t,)),
    )


def task_for(t, data=None, memories=()):
    data = value(t) if data is None else data
    return AnalystTask("u", "s", after(t), (t,), raw(data), view_for(t, data, memories))


def reply(data, **changes):
    return ModelReply(
        **(
            dict(
                content=raw(data),
                model="fixed-model",
                fingerprint="fp",
                usage={"total_tokens": 5, "prompt_tokens": 3},
                latency_seconds=0.25,
            )
            | changes
        )
    )


@pytest.mark.parametrize("kind,expected", [("self", ["小米"]), ("other", []), ("unknown", [])])
def test_view_selects_only_authorized_self_entities(kind, expected):
    t = turn(1, "我妈讨厌小米，给她推荐三星")
    data = value(t, entity="小米", relation="dislike")
    data["candidate"]["assertions"][0]["subject"] = subject(
        kind, [quote(t, "我妈")] if kind == "other" else []
    )
    view = view_for(t, data, (memory(), memory("三星", memory_id="m2")))
    assert [m.entity for m in view.memories] == expected


def test_self_request_and_self_clue_authorize_only_their_own_keys():
    t = turn(1, "给我推荐三星，小米用着顺手")
    for data in [value(t, "recommend", entity="三星"), value(t, entity="小米", clue=True)]:
        view = view_for(t, data, (memory(), memory("三星", memory_id="m2")))
        assert [m.entity for m in view.memories] == (
            ["三星"] if data["candidate"]["request"]["is_recommendation"] else ["小米"]
        )


@pytest.mark.parametrize(
    "changes",
    [
        {"user_id": "wrong-owner"},
        {
            "occurred_at": turn(2, "future").occurred_at,
            "available_at": turn(2, "future").available_at,
        },
        {"available_at": after(turn(1, "x"))},
    ],
)
def test_view_rejects_wrong_owner_and_future_before_filtering(changes):
    t = turn(1, "今天下雨")
    with pytest.raises(AnalystError, match="view"):
        view_for(t, value(t, "unrelated"), (memory(**changes),))


@pytest.mark.parametrize(
    "topic,keys,expected",
    [
        ("荣耀", ["荣耀"], ["荣耀"]),
        ("荣耀手机", ["荣耀"], ["荣耀"]),
        ("荣耀手机", ["荣耀", "荣耀手机"], ["荣耀手机"]),
        ("荣耀手机壳", ["荣耀"], []),
        ("荣", ["荣耀"], []),
        ("HONOR", ["荣耀"], []),
    ],
)
def test_view_exact_phone_alignment_never_matches_longer_product_phrase(topic, keys, expected):
    t = turn(1, f"我想了解{topic}")
    data = value(t, entity=topic)
    data["candidate"]["request"]["beneficiary"] = subject()
    memories = tuple(memory(k, memory_id=f"m{i}") for i, k in enumerate(keys))
    assert [m.entity for m in view_for(t, data, memories).memories] == expected


def test_unknown_consult_and_other_request_do_not_authorize_topics():
    t = turn(1, "我妈问三星怎么样")
    for data in [value(t, entity="三星"), value(t, "recommend", entity="三星")]:
        if "recommend" in data["intents"]:
            data["candidate"]["request"]["beneficiary"] = subject("other", [quote(t, "我妈")])
        assert view_for(t, data, (memory("三星"),)).memories == ()


def test_view_limits_sorting_and_excerpt_do_not_mutate_memories():
    names = [f"品牌{i}" for i in range(9)]
    t = turn(1, "我喜欢" + "、".join(names))
    data = envelope()
    data["intents"] = ["express_preference"]
    data["candidate"]["assertions"] = [
        {
            "subject": subject(),
            "category": "phone",
            "entity": n,
            "relation": "prefer",
            "scope": "long_term",
            "evidence": [quote(t)],
        }
        for n in names
    ]
    memories = tuple(
        memory(n, memory_id=f"m{i}", evidence_text="🙂" * 161)
        for i, n in reversed(list(enumerate(names)))
    )
    view = view_for(t, data, memories)
    assert [m.entity for m in view.memories] == [f"品牌{i}" for i in range(8)]
    assert view.truncated is True
    task = AnalystTask("u", "s", after(t), (t,), raw(data), view)
    messages = build_analyst_messages(task)
    payload = json.loads(messages[1]["content"])
    assert [m["evidence_excerpt"] for m in payload["memory_view"]["entries"]] == ["🙂" * 160] * 8
    assert all(m.evidence_text == "🙂" * 161 for m in memories)
    assert len(messages) == 2 and [m["role"] for m in messages] == ["system", "user"]
    assert set(payload) == {
        "latest_user_turn_id",
        "turns",
        "draft",
        "memory_view",
        "pending_question",
        "output_contract",
    }
    assert set(payload["memory_view"]) == {"truncated", "entries"}
    assert set(payload["memory_view"]["entries"][0]) == {
        "memory_id",
        "category",
        "entity",
        "relation",
        "occurred_at",
        "available_at",
        "evidence_ref",
        "evidence_excerpt",
    }
    assert "snapshot-private" not in raw(messages)
    assert "user_id" not in raw(messages) and "session_id" not in raw(messages)
    for obj in (task, view):
        assert "品牌" not in repr(obj) and "snapshot-private" not in repr(obj)
    with pytest.raises(FrozenInstanceError):
        view.truncated = False


def test_view_does_not_mix_duplicate_heads():
    t = turn(1, "我喜欢小米")
    with pytest.raises(AnalystError, match="view"):
        view_for(t, value(t, entity="小米", relation="prefer"), (memory(), memory(memory_id="m2")))


def test_view_omits_neutral_and_nonphone_heads():
    t = turn(1, "给我推荐小米")
    for m in (memory(relation="neutral"), memory(category="book")):
        assert not view_for(t, value(t, "recommend", entity="小米"), (m,)).memories


@pytest.mark.parametrize(
    "change",
    [
        {"user_id": "wrong"},
        {"session_id": "wrong"},
        {"as_of": after(turn(2, "future"))},
        {"pending_question": turn(0, "不是实际问题", "assistant")},
        {"turns": [turn(1, "私密内容")]},
        {"as_of": turn(1, "私密内容").available_at},
    ],
)
def test_task_rejects_mismatched_bindings_and_invalid_pending(change):
    t = turn(1, "私密内容")
    calls = []
    with pytest.raises(AnalystError) as error:
        task = replace(task_for(t), **change)
        MemoryAnalyst(lambda m: calls.append(m)).analyze(task)
    assert error.value.call is None and calls == []
    assert "私密内容" not in str(error.value)


def test_manually_constructed_view_cannot_disclose_unrelated_memory():
    t = turn(1, "今天下雨")
    view = MemoryView("u", "s", after(t), "snap", (memory(),), False)
    with pytest.raises(AnalystError, match="view"):
        build_analyst_messages(AnalystTask("u", "s", after(t), (t,), raw(value(t)), view))


def test_role_returns_final_candidate_instead_of_draft():
    t = turn(1, "我一直讨厌小米")
    task = task_for(t, value(t, entity="小米"))
    final = value(t, entity="小米", relation="dislike")
    calls = []

    def complete(messages):
        calls.append(json.loads(json.dumps(messages)))
        return reply(final)

    result = MemoryAnalyst(complete).analyze(task)
    assert result.analysis.candidate.assertions[0].relation == "dislike"
    assert result.analysis.candidate.assertions[0].scope == "long_term"
    assert len(calls) == 1 and result.call.stage == "analyst"
    assert result.call.status == "ok" and result.call.error_code is None
    assert result.call.usage == (("prompt_tokens", 3), ("total_tokens", 5))
    canonical = json.dumps(calls[0], ensure_ascii=False, sort_keys=True, allow_nan=False)
    assert result.call.request_sha256 == hashlib.sha256(canonical.encode()).hexdigest()
    assert result.call.latency_seconds == 0.25 and result.call.model == "fixed-model"


@pytest.mark.parametrize("clue", [False, True])
def test_role_accepts_final_noop_or_clue_without_long_term_attitude(clue):
    t = turn(1, "小米这台用着顺手" if clue else "今天下雨")
    data = value(t, entity="小米", clue=True) if clue else value(t, "unrelated")
    result = MemoryAnalyst(lambda _: reply(data)).analyze(task_for(t))
    assert result.analysis.candidate.assertions == ()
    assert len(result.analysis.clues) == int(clue)


def test_final_cannot_cite_memory_as_latest_user_evidence():
    t = turn(1, "我喜欢这个")
    task = task_for(t)
    final = value(t, relation="prefer", entity="三星")
    with pytest.raises(AnalystError) as error:
        MemoryAnalyst(lambda _: reply(final)).analyze(task)
    assert error.value.code == "entity_evidence"
    assert error.value.call.status == "processing_error"


def test_final_still_requires_actual_question_and_latest_evidence():
    first, latest = turn(1, "小米不错"), turn(3, "今天下雨")
    task = replace(task_for(latest), turns=(first, latest))
    old = value(first, entity="小米", clue=True)
    with pytest.raises(AnalystError, match="interaction_latest"):
        MemoryAnalyst(lambda _: reply(old)).analyze(task)
    confirm = envelope()
    confirm["confirmation"] = {
        "question_turn_id": "invented",
        "outcome": "cancel",
        "subject": None,
        "scope": None,
        "person_reference": None,
        "evidence": [quote(latest)],
    }
    with pytest.raises(AnalystError, match="interaction_question"):
        MemoryAnalyst(lambda _: reply(confirm)).analyze(task)


def test_actual_pending_question_contract_is_reused():
    question, latest = turn(1, "你是说长期吗？", "assistant"), turn(2, "不是")
    task = replace(task_for(latest), turns=(question, latest), pending_question=question)
    data = json.loads(build_analyst_messages(task)[1]["content"])
    assert data["pending_question"]["turn_id"] == "T1"
    assert data["output_contract"]["confirmation_question_turn_id"] == "T1"
    late_question = replace(question, available_at=latest.occurred_at)
    with pytest.raises(AnalystError):
        build_analyst_messages(
            replace(task, turns=(late_question, latest), pending_question=late_question)
        )


def test_final_cannot_confirm_an_assistant_turn_that_is_not_pending():
    question, latest = turn(1, "普通助手回答", "assistant"), turn(2, "取消")
    task = replace(task_for(latest), turns=(question, latest))
    data = envelope()
    data["confirmation"] = {
        "question_turn_id": "T1",
        "outcome": "cancel",
        "subject": None,
        "scope": None,
        "person_reference": None,
        "evidence": [quote(latest)],
    }
    with pytest.raises(AnalystError, match="interaction_question"):
        MemoryAnalyst(lambda _: reply(data)).analyze(task)
    bound = replace(task, pending_question=question)
    result = MemoryAnalyst(lambda _: reply(data)).analyze(bound)
    assert result.analysis.confirmation.choice.outcome == "cancel"


@pytest.mark.parametrize(
    "bad,code,status",
    [
        (ProviderError("transport"), "transport", "provider_error"),
        (RuntimeError("私密异常"), "client_error", "provider_error"),
        (None, "reply_type", "processing_error"),
        (ModelReply("私密坏JSON", "fixed", None, None, 0), "interaction_json", "processing_error"),
        (reply(envelope(), usage={"total_tokens": True}), "reply_metadata", "processing_error"),
        (reply(envelope(), usage={"私密字段": 1}), "reply_metadata", "processing_error"),
        (reply(envelope(), latency_seconds=float("nan")), "reply_metadata", "processing_error"),
    ],
)
def test_provider_and_invalid_output_keep_attempt_without_fallback(bad, code, status):
    t = turn(1, "私密输入")
    calls = []

    def complete(messages):
        calls.append(messages)
        if isinstance(bad, Exception):
            raise bad
        return bad

    with pytest.raises(AnalystError) as error:
        MemoryAnalyst(complete).analyze(task_for(t))
    assert len(calls) == 1
    assert error.value.code == code
    assert error.value.call.status == status and error.value.call.stage == "analyst"
    assert error.value.call.error_code == code
    assert "私密" not in str(error.value) + repr(error.value.call)
    assert error.value.__suppress_context__ is True


def test_invalid_draft_never_calls_provider():
    task = task_for(turn(1, "正常输入"))
    with pytest.raises(AnalystError) as error:
        MemoryAnalyst(lambda _: pytest.fail("不得调用")).analyze(replace(task, draft_raw="{}"))
    assert error.value.call is None


def test_reused_role_has_no_cross_task_history():
    a, b = turn(1, "甲轮私密内容"), turn(2, "乙轮公开背景")
    seen = []

    def complete(messages):
        seen.append(json.loads(messages[1]["content"]))
        messages.clear()
        return reply(value(b, "unrelated"))

    role = MemoryAnalyst(complete)
    task = task_for(a)
    role.analyze(task)
    role.analyze(task_for(b))
    assert "甲轮私密内容" not in raw(seen[1])
    assert seen[1]["turns"] == [{"turn_id": "T2", "role": "user", "content": "乙轮公开背景"}]
    assert task.turns == (a,) and "甲轮私密内容" in build_analyst_messages(task)[1]["content"]


@pytest.mark.parametrize(
    "changes",
    [
        {"memories": [memory()]},
        {"memories": tuple(memory(f"品牌{i}", memory_id=f"m{i}") for i in range(9))},
        {"memories": (memory("苹果", memory_id="m2"), memory("三星"))},
        {"memories": (memory(relation="neutral"),)},
        {"truncated": 1},
        {"snapshot_token": ""},
    ],
)
def test_direct_view_constructor_enforces_the_same_boundaries(changes):
    fields = dict(
        user_id="u",
        session_id="s",
        as_of=after(turn(1, "x")),
        snapshot_token="snap",
        memories=(),
        truncated=False,
    )
    with pytest.raises(AnalystError, match="view"):
        MemoryView(**(fields | changes))


@pytest.mark.parametrize("latency", [-1, True, float("inf"), 10**400])
def test_invalid_latency_is_a_recorded_processing_failure(latency):
    task = task_for(turn(1, "普通背景"))
    with pytest.raises(AnalystError) as error:
        MemoryAnalyst(lambda _: reply(envelope(), latency_seconds=latency)).analyze(task)
    assert error.value.code == "reply_metadata"
    assert error.value.call.status == "processing_error"


def test_call_usage_is_an_immutable_snapshot():
    metadata = {"total_tokens": 2}
    result = MemoryAnalyst(lambda _: reply(envelope(), usage=metadata)).analyze(
        task_for(turn(1, "普通背景"))
    )
    metadata["total_tokens"] = 999
    assert result.call.usage == (("total_tokens", 2),)


def test_provider_cannot_leak_arbitrary_error_code():
    def fail(_):
        raise ProviderError("私密异常原文")

    with pytest.raises(AnalystError) as error:
        MemoryAnalyst(fail).analyze(task_for(turn(1, "普通背景")))
    assert error.value.code == "client_error"
    assert "私密" not in str(error.value) + repr(error.value.call)


@pytest.mark.parametrize("key", ["model", "fingerprint"])
def test_provider_metadata_cannot_echo_private_text_in_error_repr(key):
    private = "这是一段合成私密表达"
    with pytest.raises(AnalystError) as error:
        MemoryAnalyst(lambda _: reply(envelope(), content="{}", **{key: private})).analyze(
            task_for(turn(1, private))
        )
    assert error.value.call.status == "processing_error"
    assert private not in str(error.value) + repr(error.value.call)
