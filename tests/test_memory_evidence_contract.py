"""输入证据模板只提供原文；不会代替模型判断或修正非法响应。"""

import json
from dataclasses import replace

import pytest
from test_memory_analyst import task_for
from test_memory_extraction import subject
from test_memory_interaction import turn, value
from test_memory_interaction_extraction import decode, quote

from memrecagent.memory_analyst import build_analyst_messages
from memrecagent.memory_extraction import ExtractionError
from memrecagent.memory_interaction_extraction import build_interaction_messages


@pytest.mark.parametrize(
    "text,entity",
    [
        ("关于荣耀：这台我用着舒服，只说单台体验。", "荣耀"),
        ("索尼。索尼这台屏幕我喜欢，只评价设备。", "索尼"),
    ],
)
def test_full_user_quote_template_locates_complete_entity_even_if_short_quote_repeats(text, entity):
    t = turn(3, text)
    assistant = turn(2, "助手介绍苹果，不是用户态度。", "assistant")
    payload = json.loads(build_interaction_messages((assistant, t), ())[1]["content"])
    contract = payload["output_contract"]
    assert contract["latest_user_evidence_quote"] == {"turn_id": "T3", "quote": text}
    data = value(t, "share_experience", entity=entity, clue=True)
    data["topics"] = []
    data["clues"][0]["evidence"] = [contract["latest_user_evidence_quote"]]
    parsed = decode(data, (assistant, t))
    assert parsed.clues[0].entity == entity
    assert parsed.clues[0].evidence[0].turn_id == "T3"
    assert parsed.clues[0].evidence[0].start == 0
    assert parsed.clues[0].evidence[0].end == len(text)


def test_subject_templates_do_not_infer_identity_from_assistant_or_old_context():
    first = turn(1, "我姑姑想换机，我阿姨也想换，给她选但没指定哪位。")
    a = turn(2, "那就按姑姑算吧。", "assistant")
    latest = turn(3, "我还没指定她是谁。")
    contract = json.loads(build_interaction_messages((first, a, latest), ())[1]["content"])[
        "output_contract"
    ]
    assert contract["subject_templates"] == {
        "self": {"kind": "self", "references": []},
        "unknown": {"kind": "unknown", "references": []},
    }
    assert contract["latest_user_evidence_quote"] == {
        "turn_id": "T3",
        "quote": "我还没指定她是谁。",
    }
    assert "relation" not in contract and "entity" not in contract and "expected" not in contract


def test_same_evidence_contract_is_forwarded_to_independent_role_without_answers():
    t = turn(1, "朋友一直喜欢苹果，不代表我也喜欢。")
    direct = json.loads(build_interaction_messages((t,), ())[1]["content"])["output_contract"]
    independent = json.loads(build_analyst_messages(task_for(t))[1]["content"])["output_contract"]
    assert independent == direct
    assert direct["latest_user_evidence_quote"]["quote"] == t.content
    assert "assertions" not in direct


def test_person_positive_fact_and_self_projection_denial_can_remain_separate():
    t = turn(1, "我朋友一直喜欢苹果，不代表我也喜欢。")
    data = value(t, entity="苹果", relation="prefer")
    data["candidate"]["assertions"][0]["subject"] = subject("other", [quote(t, "我朋友")])
    parsed = decode(data, (t,))
    assert [(a.subject.kind, a.entity, a.relation) for a in parsed.candidate.assertions] == [
        ("other", "苹果", "prefer")
    ]
    assert not parsed.candidate.unresolved


def test_incomplete_evidence_is_still_rejected_instead_of_expanded_from_template():
    t = turn(1, "回到索尼：这台屏幕我用着舒服。")
    data = value(t, "share_experience", entity="索尼", clue=True)
    data["clues"][0]["evidence"] = [quote(t, "这台屏幕我用着舒服")]
    with pytest.raises(ExtractionError, match="entity_evidence"):
        decode(data, (t,))


def test_latest_template_never_copies_assistant_question_into_user_evidence():
    q, t = turn(1, "你是否喜欢苹果？", "assistant"), turn(2, "我没说喜欢。")
    task = replace(task_for(t), turns=(q, t), pending_question=q)
    payload = json.loads(build_analyst_messages(task)[1]["content"])
    assert payload["output_contract"]["latest_user_evidence_quote"] == {
        "turn_id": "T2",
        "quote": "我没说喜欢。",
    }
    assert payload["pending_question"]["content"] == q.content
