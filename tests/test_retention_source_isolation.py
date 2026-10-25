"""已识别他人/本次来源不能随本人长期宽引用落盘。"""

import json
from dataclasses import replace

import pytest
from test_attribute_language import durable
from test_memory_extraction import subject
from test_memory_interaction import after, turn, value
from test_memory_interaction_extraction import quote

from memrecagent.attribute_candidates import compile_attribute_changes
from memrecagent.memory_commit import MemoryCommitSession
from memrecagent.memory_extraction import EvidenceSpan, ExtractionError
from memrecagent.memory_interaction_extraction import decode_interaction
from memrecagent.memory_proposals import propose
from memrecagent.memory_store import JsonlMemoryStore


def mixed(destination="brand", source="other_brand"):
    prefix = {
        "other_brand": "我姐姐长期喜欢品牌A",
        "current_brand": "这次我喜欢品牌A",
        "other_attribute": "我姐姐长期预算三千",
    }[source]
    own = "我长期喜欢品牌C" if destination == "brand" else "我长期预算三千"
    t = turn(1, prefix + "；" + own + "。")
    raw = value(t, entity="品牌C", relation="prefer") if destination == "brand" else durable(t)
    raw["schema_version"] = 4
    raw.setdefault("attribute_request", None)
    raw.setdefault("attribute_changes", [])
    target = (
        raw["candidate"]["assertions"][0] if destination == "brand" else raw["attribute_changes"][0]
    )
    target["evidence"] = [quote(t, own)]
    if source == "other_attribute":
        extra = durable(t, who=subject("other", [quote(t, "我姐姐")]))["attribute_changes"][0]
        extra["evidence"] = [quote(t, prefix)]
        raw["attribute_changes"].append(extra)
    else:
        extra = value(t, entity="品牌A", relation="prefer")["candidate"]["assertions"][0]
        extra["evidence"] = [quote(t, prefix)]
        if source == "other_brand":
            extra["subject"] = subject("other", [quote(t, "我姐姐")])
        else:
            extra["scope"] = "current"
        raw["candidate"]["assertions"].append(extra)
    return t, raw, target


@pytest.mark.parametrize("destination", ["brand", "attribute"])
@pytest.mark.parametrize("source", ["other_brand", "current_brand", "other_attribute"])
def test_language_rejects_broad_retention_mix_but_accepts_separate_quotes(destination, source):
    t, raw, target = mixed(destination, source)
    decode_interaction(json.dumps(raw), turns=(t,))
    target["evidence"] = [quote(t)]
    with pytest.raises(ExtractionError, match="mixed_retention_evidence"):
        decode_interaction(json.dumps(raw), turns=(t,))


@pytest.mark.parametrize("destination", ["brand", "attribute"])
def test_direct_commit_cannot_bypass_cross_type_retention_guard(tmp_path, destination):
    t, raw, _ = mixed(destination, "other_brand")
    parsed = decode_interaction(json.dumps(raw), turns=(t,))
    broad = (EvidenceSpan(t.turn_id, 0, len(t.content)),)
    candidate, changes = parsed.candidate, parsed.attribute_changes
    if destination == "brand":
        candidate = replace(
            candidate,
            assertions=(
                replace(candidate.assertions[0], evidence=broad),
                *candidate.assertions[1:],
            ),
        )
    else:
        changes = (replace(changes[0], evidence=broad), *changes[1:])
    store = JsonlMemoryStore(tmp_path)
    with pytest.raises(ExtractionError, match="mixed_retention_evidence"):
        MemoryCommitSession(store, user_id="u", session_id="s").prepare(
            candidate, turns=(t,), attribute_changes=changes, as_of=after(t)
        )
    assert store.list_memories("u", as_of=after(t)) == ()
    assert store.list_attribute_memories("u", as_of=after(t)) == ()


def test_direct_proposal_and_attribute_compile_also_reject_known_other_sources(tmp_path):
    store = JsonlMemoryStore(tmp_path)
    t, raw, _ = mixed("brand", "other_brand")
    parsed = decode_interaction(json.dumps(raw), turns=(t,))
    candidate = replace(
        parsed.candidate,
        assertions=(
            replace(
                parsed.candidate.assertions[0],
                evidence=(EvidenceSpan(t.turn_id, 0, len(t.content)),),
            ),
            *parsed.candidate.assertions[1:],
        ),
    )
    with pytest.raises(ExtractionError, match="mixed_retention_evidence"):
        propose(
            candidate,
            store=store,
            user_id="u",
            session_id="s",
            turns=(t,),
            people=(),
            as_of=after(t),
        )
    t, raw, _ = mixed("attribute", "other_attribute")
    parsed = decode_interaction(json.dumps(raw), turns=(t,))
    changes = (
        replace(
            parsed.attribute_changes[0], evidence=(EvidenceSpan(t.turn_id, 0, len(t.content)),)
        ),
        *parsed.attribute_changes[1:],
    )
    with pytest.raises(ExtractionError, match="mixed_retention_evidence"):
        compile_attribute_changes(
            changes, store=store, user_id="u", session_id="s", turns=(t,), as_of=after(t)
        )
