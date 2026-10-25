"""确认候选必须来自实际最新用户回复，程序契约测试不证明语义准确率。"""

import json
from dataclasses import replace
from datetime import timedelta

import pytest
from test_memory_commit import ambiguous
from test_memory_extraction import NOW, span

from memrecagent.memory_confirmation import build_confirmation_messages, decode_confirmation
from memrecagent.memory_extraction import ExtractionError, Turn


def answer(text="是我爸，这次给他挑"):
    return Turn("R1", "user", text, NOW + timedelta(minutes=11), NOW + timedelta(minutes=11))


def data(reply, **changes):
    return (
        dict(
            outcome="confirm",
            subject="self",
            scope="current",
            person_reference=None,
            evidence=[span(reply)],
        )
        | changes
    )


def test_decodes_self_other_cancel_and_unclear_without_persisting():
    reply = answer()
    own = decode_confirmation(json.dumps(data(reply)), reply=reply)
    assert (own.outcome, own.subject, own.scope) == ("confirm", "self", "current")
    other = decode_confirmation(
        json.dumps(
            data(reply, subject="other", scope="long_term", person_reference=span(reply, "我爸"))
        ),
        reply=reply,
    )
    assert other.person_reference.turn_id == "R1" and other.scope == "long_term"
    for outcome in ("cancel", "unclear"):
        value = decode_confirmation(
            json.dumps(data(reply, outcome=outcome, subject=None, scope=None)), reply=reply
        )
        assert value.outcome == outcome and value.subject is None


@pytest.mark.parametrize(
    "mutation",
    [
        {"evidence": [{"turn_id": "T0", "start": 0, "end": 1}]},
        {"evidence": [{"turn_id": "R1", "start": False, "end": 1}]},
        {"evidence": []},
        {"subject": "unknown"},
        {"scope": "unknown"},
        {"subject": "other", "person_reference": None},
        {"outcome": "unclear"},
        {"entity": "新品牌"},
    ],
)
def test_invalid_or_invented_confirmation_is_rejected(mutation):
    reply = answer()
    with pytest.raises(ExtractionError):
        decode_confirmation(json.dumps(data(reply, **mutation)), reply=reply)


def test_assistant_reply_and_pronoun_only_anchor_are_not_user_confirmation():
    reply = answer("他喜欢")
    with pytest.raises(ExtractionError):
        decode_confirmation(json.dumps(data(reply)), reply=replace(reply, role="assistant"))
    with pytest.raises(ExtractionError):
        decode_confirmation(
            json.dumps(data(reply, subject="other", person_reference=span(reply, "他"))),
            reply=reply,
        )


def test_prompt_only_contains_actual_question_and_reply_not_future_or_oracle(tmp_path):
    _, _, preview = ambiguous(tmp_path)
    reply = answer()
    messages = build_confirmation_messages(preview, reply=reply)
    assert [m["role"] for m in messages] == ["system", "user"]
    payload = json.loads(messages[1]["content"])
    assert payload["reply"]["content"] == reply.content
    assert payload["question"] == preview.proposal.decision.clarification
    assert "expected" not in payload and "future" not in payload


def test_ready_preview_cannot_fabricate_a_confirmation_question(tmp_path):
    from test_memory_commit import setup

    _, _, preview = setup(tmp_path)
    with pytest.raises(ValueError):
        build_confirmation_messages(preview, reply=answer())


def test_time_scope_word_is_not_a_person_anchor():
    reply = answer("这次给他挑")
    with pytest.raises(ExtractionError):
        decode_confirmation(
            json.dumps(data(reply, subject="other", person_reference=span(reply, "这次"))),
            reply=reply,
        )
