"""真实提取/提交路径的长期属性、混合原子性及证据边界。"""

import json

import pytest
from test_conversation_products import process
from test_memory_extraction import subject
from test_memory_interaction import after, turn, value
from test_memory_interaction_extraction import quote
from test_request_attributes import host, ids, payload, plain

from memrecagent.memory_commit import CommitError
from memrecagent.memory_extraction import ExtractionError
from memrecagent.memory_interaction_extraction import decode_interaction
from memrecagent.memory_store import JsonlMemoryStore
from memrecagent.memory_use_policy import JsonMemoryUseStore, MemoryUsePolicy


def durable(t, *, amount=3000, who=None, scope="long_term", active=True, recommend=False):
    raw = plain(t, "recommend" if recommend else "express_preference")
    raw["schema_version"] = 4
    raw["attribute_changes"] = [
        dict(
            subject=who or subject(),
            category="phone",
            attribute="price_cny",
            operator="<=",
            value=amount,
            unit="CNY",
            is_hard=True,
            active=active,
            scope=scope,
            evidence=[quote(t)],
        )
    ]
    if recommend and who:
        raw["candidate"]["request"]["beneficiary"] = who
    if recommend:
        raw["intents"].append("express_preference")
    return raw


def test_known_temporary_budget_cannot_ride_in_long_term_brand_evidence_even_v2():
    t = turn(1, "我长期喜欢品牌C；这次预算三千。")
    raw = value(t, entity="品牌C", relation="prefer")
    with pytest.raises(ExtractionError) as caught:
        decode_interaction(json.dumps(raw), turns=(t,))
    assert caught.value.code == "mixed_retention_evidence"


def test_narrow_brand_and_separate_current_budget_commit_only_brand_text(tmp_path):
    h = host(tmp_path)
    t = turn(1, "我长期喜欢品牌C；这次预算三千。")
    raw = payload(t)
    raw["candidate"]["assertions"] = value(t, entity="品牌C", relation="prefer")["candidate"][
        "assertions"
    ]
    raw["candidate"]["assertions"][0]["evidence"] = [quote(t, "我长期喜欢品牌C")]
    event = raw["attribute_request"]
    event["evidence"] = event["conditions"][0]["evidence"] = [quote(t, "这次预算三千")]
    assert len(process(h, t, raw)["new_memory_ids"]) == 1
    memories = JsonlMemoryStore(tmp_path / "memory").list_memories("u", as_of=after(t))
    assert memories[0].evidence_text == "我长期喜欢品牌C"
    assert "三千" not in memories[0].evidence_text


def test_actual_language_candidate_saves_long_term_budget_and_new_session_ranks(tmp_path):
    h = host(tmp_path)
    t = turn(1, "以后我买手机预算上限三千，请长期保存。")
    result = process(h, t, durable(t))
    assert len(result["new_memory_ids"]) == 1
    store = JsonlMemoryStore(tmp_path / "memory")
    assert store.list_memories("u", as_of=after(t)) == ()
    assert store.active_attribute_memories("u", as_of=after(t))[0].value == 3000
    t = turn(3, "给我推荐手机。")
    assert ids(process(host(tmp_path, "new"), t, plain(t))) == ["B", "C"]


def test_language_revision_and_retraction_preserve_history_without_reviving_old_budget(tmp_path):
    h = host(tmp_path)
    t = turn(1, "以后我的预算上限三千。")
    process(h, t, durable(t))
    t = turn(3, "改为以后我的预算上限四千五。")
    process(h, t, durable(t, amount=4500))
    t = turn(5, "取消我的长期预算上限。")
    process(h, t, durable(t, active=False, amount=None))
    store = JsonlMemoryStore(tmp_path / "memory")
    history = store.list_attribute_memories("u", as_of=after(t))
    assert len(history) == 3 and history[1].supersedes == history[0].memory_id
    assert history[2].supersedes == history[1].memory_id
    assert store.active_attribute_memories("u", as_of=after(t)) == ()
    t = turn(7, "给我推荐手机。")
    assert ids(process(host(tmp_path, "new"), t, plain(t))) == ["A", "B", "C"]


def test_old_recorded_attribute_cannot_return_as_current_override_after_revision(tmp_path):
    from memrecagent.memory_commit import MemoryCommitSession

    h = host(tmp_path)
    old = turn(1, "我以后手机预算上限三千。")
    raw = durable(old)
    process(h, old, raw)
    new = turn(3, "以后我的手机预算上限改为四千五。")
    process(h, new, durable(new, amount=4500))
    parsed = decode_interaction(json.dumps(raw), turns=(old,))
    store = JsonlMemoryStore(tmp_path / "memory")
    ticket = MemoryCommitSession(store, user_id="u", session_id="s").prepare(
        parsed.candidate, attribute_changes=parsed.attribute_changes, turns=(old,), as_of=after(new)
    )
    assert ticket.proposal.proposed_attributes == ()
    assert ticket.proposal.session_attributes == ()


def test_failed_mixed_brand_attribute_publish_saves_neither(tmp_path, monkeypatch):
    h = host(tmp_path)
    t = turn(1, "我长期喜欢品牌C；以后预算三千。")
    raw = durable(t)
    raw["candidate"]["assertions"] = value(t, entity="品牌C", relation="prefer")["candidate"][
        "assertions"
    ]

    def fail(*args):
        raise OSError("synthetic publication failure")

    monkeypatch.setattr("memrecagent.memory_store.os.replace", fail)
    with pytest.raises(CommitError):
        process(h, t, raw)
    store = JsonlMemoryStore(tmp_path / "memory")
    assert store.list_memories("u", as_of=after(t)) == ()
    assert store.list_attribute_memories("u", as_of=after(t)) == ()


@pytest.mark.parametrize("mode", ["unknown", "unclear"])
def test_uncertain_attributes_block_whole_batch_until_complete_restatement(tmp_path, mode):
    h = host(tmp_path)
    t = turn(1, "买手机预算三千，但主体和范围还没确定。")
    raw = durable(
        t,
        who=subject("unknown") if mode == "unknown" else None,
        scope="unclear" if mode == "unclear" else "long_term",
    )
    result = process(h, t, raw)
    assert result["response_action"] == "clarify" and result["new_memory_ids"] == []
    t = turn(3, "是我本人，以后手机预算上限三千，长期保存。")
    assert len(process(h, t, durable(t))["new_memory_ids"]) == 1


def test_other_long_term_attribute_is_session_only_and_person_specific(tmp_path):
    h = host(tmp_path)
    t = turn(1, "我叔叔买手机一直预算上限三千。")
    who = subject("other", [quote(t, "我叔叔")])
    assert process(h, t, durable(t, who=who))["new_memory_ids"] == []
    t = turn(3, "给他推荐手机。")
    raw = plain(t)
    raw["candidate"]["request"]["beneficiary"] = who
    assert ids(process(h, t, raw)) == ["B", "C"]
    t = turn(5, "给我推荐手机。")
    assert ids(process(h, t, plain(t))) == ["A", "B", "C"]
    t = turn(7, "给我叔叔推荐手机。")
    fresh_who = subject("other", [quote(t, "我叔叔")])
    raw = plain(t)
    raw["candidate"]["request"]["beneficiary"] = fresh_who
    assert ids(process(host(tmp_path, "fresh"), t, raw)) == ["A", "B", "C"]
    assert not list((tmp_path / "memory").glob("*.jsonl"))


def test_updates_permission_disables_attribute_and_brand_persistence(tmp_path):
    JsonMemoryUseStore(tmp_path / "memory/use-policy").set(
        "u", MemoryUsePolicy(long_term_updates=False)
    )
    h = host(tmp_path)
    t = turn(1, "我以后预算三千。")
    assert process(h, t, durable(t))["new_memory_ids"] == []
    assert JsonlMemoryStore(tmp_path / "memory").list_attribute_memories("u", as_of=after(t)) == ()


def test_hand_constructed_attribute_candidate_cannot_bypass_numeric_grounding(tmp_path):
    from dataclasses import replace

    from memrecagent.memory_commit import MemoryCommitSession

    t = turn(1, "我以后手机预算三千。")
    decoded = decode_interaction(json.dumps(durable(t)), turns=(t,))
    forged = replace(decoded.attribute_changes[0], value=4500)
    memory = MemoryCommitSession(JsonlMemoryStore(tmp_path), user_id="u", session_id="s")
    with pytest.raises(ExtractionError) as caught:
        memory.prepare(decoded.candidate, attribute_changes=(forged,), turns=(t,), as_of=after(t))
    assert caught.value.code == "attribute_value_evidence"


def test_commit_of_attribute_ticket_is_idempotent_and_delete_invalidates_it(tmp_path):
    from test_memory_interaction import model

    from memrecagent.memory_interaction import ConversationMemorySession

    store = JsonlMemoryStore(tmp_path)
    h = ConversationMemorySession(store, user_id="u", session_id="s")
    t = turn(1, "我以后预算上限三千。")
    preview = h.prepare_user(t, complete=model(durable(t)), as_of=after(t))
    first = h.commit(preview.ticket_id, as_of=after(t))
    second = h.commit(preview.ticket_id, as_of=after(t))
    assert first.memory_ids == second.memory_ids and second.status == "already_applied"
    store.delete_attribute("u", category="phone", attribute="price_cny", deleted_at=after(t))
    with pytest.raises(CommitError):
        h.commit(preview.ticket_id, as_of=after(t))


@pytest.mark.parametrize(
    "mutation,code",
    [
        ("invented", "attribute_value_evidence"),
        ("assistant", "quote_role"),
        ("extra", "fields"),
        ("flag", "attribute_scope"),
        ("retract_value", "attribute_retraction"),
    ],
)
def test_schema4_rejects_fabricated_sources_flags_and_fields(mutation, code):
    t, a = turn(3, "我的手机预算长期三千。"), turn(2, "他的预算四千。", "assistant")
    raw = durable(t)
    row = raw["attribute_changes"][0]
    if mutation == "invented":
        row["value"] = 4000
    if mutation == "assistant":
        row["evidence"] = [quote(a)]
    if mutation == "extra":
        row["supersedes"] = "model-parent"
    if mutation == "flag":
        row["is_hard"] = 1
    if mutation == "retract_value":
        row["active"] = False
    with pytest.raises(ExtractionError) as caught:
        decode_interaction(json.dumps(raw), turns=(turn(1, "咨询手机"), a, t))
    assert caught.value.code == code


def test_both_analysis_roles_request_schema4_and_minimal_complete_evidence():
    from memrecagent.memory_analyst import ANALYST_PROMPT
    from memrecagent.memory_interaction_extraction import INTERACTION_PROMPT

    for prompt in (INTERACTION_PROMPT, ANALYST_PROMPT):
        assert "schema_version=4" in prompt and "attribute_changes" in prompt
        assert "混合整轮" in prompt


def test_language_explicit_temporary_removal_bypasses_permanent_budget_but_disk_stays(tmp_path):
    h = host(tmp_path)
    t = turn(1, "我以后手机预算上限三千。")
    process(h, t, durable(t))
    before = JsonlMemoryStore(tmp_path / "memory").snapshot_token("u")
    t = turn(3, "仅这次取消预算上限，给我推荐手机，长期预算不改。")
    raw = payload(t, amount=None, remove=("price_cny",), recommend=True)
    raw.update(schema_version=4, attribute_changes=[])
    assert ids(process(h, t, raw)) == ["A", "B", "C"]
    assert JsonlMemoryStore(tmp_path / "memory").snapshot_token("u") == before
    t = turn(5, "本次选购结束。")
    process(h, t, payload(t, op="end", amount=None))
    t = turn(7, "开始另一轮，给我推荐手机。")
    assert ids(process(h, t, plain(t))) == ["B", "C"]


def test_attribute_deletion_watermark_blocks_old_language_evidence_and_valid_brand_batch(tmp_path):
    from memrecagent.memory_commit import MemoryCommitSession

    t = turn(1, "我长期喜欢品牌C；以后我的手机预算上限三千。")
    store = JsonlMemoryStore(tmp_path)
    store.delete_attribute("u", category="phone", attribute="price_cny", deleted_at=after(t))
    raw = durable(t)
    raw["candidate"]["assertions"] = value(t, entity="品牌C", relation="prefer")["candidate"][
        "assertions"
    ]
    parsed = decode_interaction(json.dumps(raw), turns=(t,))
    h = MemoryCommitSession(store, user_id="u", session_id="s")
    preview = h.prepare(
        parsed.candidate,
        attribute_changes=parsed.attribute_changes,
        turns=(t,),
        as_of=after(turn(3, "later")),
    )
    with pytest.raises(CommitError):
        h.commit(preview.ticket_id, as_of=after(turn(3, "later")))
    assert store.list_memories("u", as_of=after(turn(5, "later"))) == ()


@pytest.mark.parametrize(
    "policy", [MemoryUsePolicy(long_term_updates=False), MemoryUsePolicy(personalisation=False)]
)
def test_disabling_future_update_or_profile_use_still_honors_explicit_current_requirement(
    tmp_path, policy
):
    JsonMemoryUseStore(tmp_path / "memory/use-policy").set("u", policy)
    h = host(tmp_path)
    t = turn(1, "以后我手机预算上限三千；现在请给我推荐手机。")
    assert ids(process(h, t, durable(t, recommend=True))) == ["B", "C"]
    t = turn(3, "重新给我推荐手机。")
    assert ids(process(host(tmp_path, "new"), t, plain(t))) == ["A", "B", "C"]
