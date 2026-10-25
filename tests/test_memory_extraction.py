"""候选解析的格式和证据测试；不把假响应当模型语义准确率。"""

import json
from copy import deepcopy
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from memrecagent.memory_extraction import ExtractionError, Turn, decode_candidate

NOW = datetime(2026, 10, 20, 2, tzinfo=UTC)


def turns_for(*texts):
    return tuple(
        Turn(f"T{i}", "user", text, NOW + timedelta(minutes=i), NOW + timedelta(minutes=i))
        for i, text in enumerate(texts, 1)
    )


def span(turn, text=None):
    text = text or turn.content
    start = turn.content.index(text)
    return {"turn_id": turn.turn_id, "start": start, "end": start + len(text)}


def subject(kind="self", references=()):
    return {"kind": kind, "references": list(references)}


def output_for(turn):
    evidence = [span(turn)]
    return {
        "request": {
            "is_recommendation": False,
            "beneficiary": subject(),
            "category": "phone",
            "requested_entity": None,
            "change_scope": "request",
            "evidence": [],
        },
        "assertions": [
            {
                "subject": subject(),
                "category": "phone",
                "entity": "小米",
                "relation": "exclude",
                "scope": "long_term",
                "evidence": evidence,
            }
        ],
        "unresolved": [],
    }


def decode(data, turns):
    return decode_candidate(json.dumps(data, ensure_ascii=False), turns=turns)


def test_decodes_self_and_distinct_request_subject():
    turns = turns_for("我爸喜欢小米，给我推荐手机")
    data = output_for(turns[0])
    data["assertions"][0].update(
        subject=subject("other", [span(turns[0], "我爸")]), relation="prefer"
    )
    data["request"].update(is_recommendation=True, evidence=[span(turns[0], "给我推荐手机")])
    candidate = decode(data, turns)
    assert candidate.request.beneficiary.kind == "self"
    assert candidate.assertions[0].subject.kind == "other"
    assert candidate.assertions[0].subject.references[0].end == 2


def test_roundtrip_unicode_span():
    turns = turns_for("🙂以后买手机别推荐小米")
    data = output_for(turns[0])
    data["assertions"][0]["evidence"] = [span(turns[0], "以后买手机别推荐小米")]
    parsed = decode(data, turns)
    evidence = parsed.assertions[0].evidence[0]
    assert evidence.start == 1
    assert turns[0].content[evidence.start : evidence.end] == "以后买手机别推荐小米"


@pytest.mark.parametrize(
    "mutation",
    [
        lambda d: d.update(memory_id="private-forged"),
        lambda d: d.pop("unresolved"),
        lambda d: d["request"].update(is_recommendation=1),
        lambda d: d["request"].update(change_scope="forever"),
        lambda d: d["assertions"][0].update(confidence=0.99),
        lambda d: d["assertions"][0].update(relation="love"),
        lambda d: d["assertions"][0].update(scope="any"),
        lambda d: d["assertions"][0].update(evidence=[]),
        lambda d: d["assertions"][0].update(entity="未曾提及品牌"),
        lambda d: d["assertions"][0].update(subject=subject("other")),
        lambda d: d["assertions"][0]["evidence"][0].update(turn_id="T99"),
        lambda d: d["assertions"][0]["evidence"][0].update(start=True),
        lambda d: d["assertions"][0]["evidence"][0].update(start=-1),
        lambda d: d["assertions"][0]["evidence"][0].update(end=999),
        lambda d: d["assertions"][0]["evidence"][0].update(end=0),
        lambda d: d.update(assertions=d["assertions"] * 33),
    ],
)
def test_invalid_fields_or_evidence_reject_whole_output_without_echo(mutation):
    turns = turns_for("私有合成原话：以后买手机别推荐小米")
    data = deepcopy(output_for(turns[0]))
    mutation(data)
    with pytest.raises(ExtractionError) as error:
        decode(data, turns)
    assert error.value.code
    assert "私有合成原话" not in str(error.value)
    assert "private-forged" not in str(error.value)


@pytest.mark.parametrize("raw", ['{"request":1,"request":2}', "[]", "NaN", "{", "x" * 65537])
def test_invalid_json_duplicates_and_size_are_rejected(raw):
    with pytest.raises(ExtractionError):
        decode_candidate(raw, turns=turns_for("合成"))


def test_assistant_evidence_and_ambiguous_pronoun_as_other_identity_rejected():
    original = turns_for("他喜欢小米")[0]
    data = output_for(original)
    data["assertions"][0]["subject"] = subject("other", [span(original, "他")])
    with pytest.raises(ExtractionError):
        decode(data, (original,))
    assistant = Turn("T1", "assistant", original.content, NOW, NOW)
    with pytest.raises(ExtractionError):
        decode(output_for(assistant), (assistant,))


def test_turns_need_unique_ids_aware_times_and_order():
    turn = turns_for("合成")[0]
    with pytest.raises(ExtractionError):
        decode(output_for(turn), (turn, turn))
    with pytest.raises(ValueError):
        Turn("T1", "user", "合成", NOW.replace(tzinfo=None), NOW)
    with pytest.raises(ValueError):
        Turn("T1", "user", "合成", NOW, NOW - timedelta(seconds=1))


def test_frozen_synthetic_fixture_has_six_groups_of_four():
    cases = json.loads(
        (Path(__file__).parent / "fixtures/memory_extraction_cases.json").read_text()
    )
    assert len(cases) == len({case["case_id"] for case in cases}) == 24
    groups = {case["group"] for case in cases}
    assert len(groups) == 6
    assert all(sum(case["group"] == group for case in cases) == 4 for group in groups)
    assert all(case["synthetic"] is True and case["expected"] for case in cases)
