"""新交互契约的真实解析/提案测试，不充当模型理解准确率。"""

import json
from dataclasses import replace

import pytest
from test_memory import sample_memory
from test_memory_extraction import NOW, subject, turns_for

from memrecagent.memory_extraction import EvidenceSpan, ExtractionError
from memrecagent.memory_proposals import propose
from memrecagent.memory_store import JsonlMemoryStore


def quote(turn, text=None):
    return {"turn_id": turn.turn_id, "quote": turn.content if text is None else text}


def envelope():
    return {
        "schema_version": 2,
        "intents": ["consult"],
        "topics": [],
        "clues": [],
        "candidate": {
            "request": {
                "is_recommendation": False,
                "beneficiary": subject("unknown"),
                "category": None,
                "requested_entity": None,
                "change_scope": "request",
                "evidence": [],
            },
            "assertions": [],
            "unresolved": [],
        },
        "confirmation": None,
    }


def decode(value, turns):
    from memrecagent.memory_interaction_extraction import decode_interaction

    return decode_interaction(json.dumps(value, ensure_ascii=False), turns=turns)


def test_o3_quote_locates_real_person_without_counting_digits():
    from memrecagent.memory_interaction_extraction import locate_quote

    turns = turns_for("我讨厌小米，给我爸推荐手机")
    assert locate_quote(turns, turn_id="T1", quote="我爸", role="user") == EvidenceSpan("T1", 7, 9)
    value = envelope()
    value["intents"] = ["recommend", "express_preference"]
    value["candidate"]["request"].update(
        is_recommendation=True,
        category="phone",
        beneficiary=subject("other", [quote(turns[0], "我爸")]),
        evidence=[quote(turns[0], "给我爸推荐手机")],
    )
    value["candidate"]["assertions"] = [
        {
            "subject": subject(),
            "category": "phone",
            "entity": "小米",
            "relation": "dislike",
            "scope": "long_term",
            "evidence": [quote(turns[0], "我讨厌小米")],
        }
    ]
    parsed = decode(value, turns)
    assert parsed.candidate.request.beneficiary.references == (EvidenceSpan("T1", 7, 9),)
    assert parsed.candidate.assertions[0].evidence == (EvidenceSpan("T1", 0, 5),)


@pytest.mark.parametrize(
    "text,word,turn_id,role,code",
    [
        ("我爸说我爸选", "我爸", "T1", "user", "quote_ambiguous"),
        ("哈哈哈", "哈哈", "T1", "user", "quote_ambiguous"),
        ("我爸", "我哥", "T1", "user", "quote_missing"),
        ("我爸", "我爸", "T9", "user", "quote_missing"),
        ("我爸", "我爸 ", "T1", "user", "quote_missing"),
        ("我爸", "", "T1", "user", "quote_missing"),
        ("我爸", "我爸", "T1", "assistant", "quote_role"),
    ],
)
def test_duplicate_missing_assistant_and_fabricated_quote_are_rejected(
    text, word, turn_id, role, code
):
    from memrecagent.memory_interaction_extraction import locate_quote

    with pytest.raises(ExtractionError, match=code):
        locate_quote(turns_for(text), turn_id=turn_id, quote=word, role=role)


def test_quote_uses_requested_turn_and_unicode_codepoints():
    from memrecagent.memory_interaction_extraction import locate_quote

    turns = turns_for("我爸喜欢三星", "🙂我哥喜欢苹果")
    assert locate_quote(turns, turn_id="T2", quote="我哥", role="user") == EvidenceSpan("T2", 1, 3)
    with pytest.raises(ExtractionError, match="quote_missing"):
        locate_quote(turns, turn_id="T2", quote="我爸", role="user")


def test_topic_and_clue_do_not_become_durable_assertion(tmp_path):
    turns = turns_for("你怎么看三星？", "三星系统我用着挺顺手。")
    value = envelope()
    value["topics"] = [quote(turns[0], "三星")]
    value["clues"] = [
        {
            "subject": subject(),
            "category": "phone",
            "entity": "三星",
            "evidence": [quote(turns[1], "三星系统我用着挺顺手。")],
        }
    ]
    parsed = decode(value, turns)
    assert parsed.intents == ("consult",)
    assert parsed.topic_entities == ("三星",)
    assert len(parsed.clues) == 1 and parsed.clues[0].entity == "三星"
    assert parsed.candidate.request.requested_entity is None
    store = JsonlMemoryStore(tmp_path)
    result = propose(
        parsed.candidate,
        store=store,
        user_id="synthetic-user",
        session_id="S",
        people=(),
        turns=turns,
        as_of=NOW.replace(day=21),
    )
    assert result.proposed_memories == ()
    assert store.list_memories("synthetic-user", as_of=NOW.replace(day=21)) == ()


def test_topics_are_deduplicated_in_original_context_order():
    turns = turns_for("三星和苹果怎么比", "还想比较苹果与华为")
    value = envelope()
    value["intents"] = ["compare", "consult"]
    value["topics"] = [
        quote(turns[1], "华为"),
        quote(turns[1], "苹果"),
        quote(turns[0], "三星"),
        quote(turns[0], "苹果"),
    ]
    assert decode(value, turns).topic_entities == ("三星", "苹果", "华为")


@pytest.mark.parametrize(
    "mutate",
    [
        lambda d: d.update(schema_version=True),
        lambda d: d.update(schema_version=1),
        lambda d: d.update(schema_version="2"),
        lambda d: d.update(user_id="forged"),
        lambda d: d.update(intents=[]),
        lambda d: d.update(intents=["unknown"]),
        lambda d: d.update(intents=["consult", "consult"]),
        lambda d: d.update(intents=["unrelated", "consult"]),
        lambda d: d.update(topics=[{"turn_id": "T1", "quote": "三星", "start": 0}]),
        lambda d: d.update(topics=[{"turn_id": "T1", "start": 0, "end": 2}]),
        lambda d: d.update(topics=[{"turn_id": "T1", "quote": "三星"}] * 33),
        lambda d: d.update(
            clues=[
                {
                    "subject": subject(),
                    "category": "phone",
                    "entity": "三星",
                    "evidence": [],
                    "relation": "prefer",
                }
            ]
        ),
        lambda d: d["candidate"]["request"].update(change_scope="current"),
        lambda d: d["candidate"].update(supersedes="forged"),
        lambda d: d["candidate"]["request"].update(
            requested_entity="苹果", evidence=[{"turn_id": "T1", "quote": "三星"}]
        ),
        lambda d: d["candidate"]["request"].update(
            is_recommendation=True, evidence=[{"turn_id": "T1", "quote": "三星"}]
        ),
    ],
)
def test_invalid_contract_is_not_silently_repaired(mutate):
    value = envelope()
    mutate(value)
    with pytest.raises(ExtractionError):
        decode(value, turns_for("三星"))


@pytest.mark.parametrize(
    "raw",
    ["{", "[]", '{"schema_version":2,"schema_version":2}', '{"schema_version":NaN}', "x" * 65537],
    ids=["invalid", "array", "duplicate", "nan", "size"],
)
def test_bad_json_duplicate_keys_and_oversize_are_rejected(raw):
    from memrecagent.memory_interaction_extraction import decode_interaction

    with pytest.raises(ExtractionError):
        decode_interaction(raw, turns=turns_for("三星"))


def test_assistant_praise_cannot_supply_user_topic_or_clue_evidence():
    turns = turns_for("三星不错", "随便看看")
    turns = (replace(turns[0], role="assistant"), turns[1])
    value = envelope()
    value["topics"] = [quote(turns[0], "三星")]
    with pytest.raises(ExtractionError, match="quote_role"):
        decode(value, turns)
    value["topics"] = []
    value["clues"] = [
        {"subject": subject(), "category": "phone", "entity": "三星", "evidence": [quote(turns[0])]}
    ]
    with pytest.raises(ExtractionError, match="quote_role"):
        decode(value, turns)


def test_entity_proof_and_latest_expression_still_required():
    turns = turns_for("我喜欢三星", "今天讲Python吧")
    value = envelope()
    value["intents"] = ["express_preference"]
    value["candidate"]["assertions"] = [
        {
            "subject": subject(),
            "category": "phone",
            "entity": "三星",
            "relation": "prefer",
            "scope": "long_term",
            "evidence": [quote(turns[0])],
        }
    ]
    with pytest.raises(ExtractionError, match="interaction_latest"):
        decode(value, turns)
    value["candidate"]["assertions"][0]["evidence"] = [quote(turns[1])]
    with pytest.raises(ExtractionError, match="entity_evidence"):
        decode(value, turns)


def confirmed_turns():
    turns = turns_for(
        "我好像喜欢三星，但不清楚本次还是长期", "只限本次还是以后都如此？", "是我自己的长期偏好"
    )
    return (turns[0], replace(turns[1], role="assistant"), turns[2])


def confirmation_value(turns):
    value = envelope()
    value["intents"] = ["express_preference"]
    value["confirmation"] = {
        "question_turn_id": "T2",
        "outcome": "confirm",
        "subject": "self",
        "scope": "long_term",
        "person_reference": None,
        "evidence": [quote(turns[2])],
    }
    return value


def test_confirmation_has_actual_question_reference_but_no_commit_permission():
    turns = confirmed_turns()
    value = confirmation_value(turns)
    parsed = decode(value, turns)
    assert parsed.confirmation.question_turn_id == "T2"
    assert parsed.confirmation.choice.scope == "long_term"
    assert parsed.confirmation.choice.evidence == (EvidenceSpan("T3", 0, 9),)
    assert parsed.candidate.assertions == ()


@pytest.mark.parametrize(
    "kind", ["missing", "user_question", "old_reply", "mixed_assertion", "mixed_clue"]
)
def test_confirmation_invalid_binding_and_mixed_new_changes_are_rejected(kind):
    turns = confirmed_turns()
    value = confirmation_value(turns)
    if kind == "missing":
        value["confirmation"]["question_turn_id"] = "T9"
    elif kind == "user_question":
        value["confirmation"]["question_turn_id"] = "T1"
    elif kind == "old_reply":
        value["confirmation"]["evidence"] = [quote(turns[0])]
    elif kind == "mixed_clue":
        value["clues"] = [
            {
                "subject": subject(),
                "category": "phone",
                "entity": "三星",
                "evidence": [quote(turns[0])],
            }
        ]
    else:
        value["candidate"]["assertions"] = [
            {
                "subject": subject(),
                "category": "phone",
                "entity": "三星",
                "relation": "prefer",
                "scope": "long_term",
                "evidence": [quote(turns[0])],
            }
        ]
    with pytest.raises(ExtractionError):
        decode(value, turns)


def test_builder_includes_actual_assistant_context_but_no_codepoint_or_answers():
    from memrecagent.memory_interaction_extraction import build_interaction_messages

    turns = confirmed_turns()
    messages = build_interaction_messages(turns, (sample_memory(),), pending_question=turns[1])
    data = json.loads(messages[1]["content"])
    assert data["latest_user_turn_id"] == "T3"
    assert data["turns"][1] == {
        "turn_id": "T2",
        "role": "assistant",
        "content": "只限本次还是以后都如此？",
    }
    assert data["pending_question"]["turn_id"] == "T2"
    assert data["own_memories"] == [{"category": "phone", "entity": "小米", "relation": "exclude"}]
    assert "codepoints" not in data["turns"][0]
    assert "expected" not in data and "schema_version" not in data["own_memories"][0]
    with pytest.raises(ExtractionError):
        build_interaction_messages(
            turns, (), pending_question=replace(turns[1], content="未发生的问题")
        )
