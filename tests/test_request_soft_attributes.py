"""仅本次软偏好经过语言宿主，在选购任务内生效且不落盘。"""

import json

import pytest
from test_attribute_language import durable
from test_conversation_products import process
from test_memory_extraction import subject
from test_memory_interaction import after, turn
from test_memory_interaction_extraction import quote
from test_request_attributes import host, ids, payload, plain

from memrecagent.memory_extraction import ExtractionError
from memrecagent.memory_interaction_extraction import decode_interaction
from memrecagent.memory_store import JsonlMemoryStore


def temporary(t, *, amount=3000, who=None, active=True, recommend=False):
    raw = durable(t, amount=amount, who=who, scope="request", active=active, recommend=recommend)
    raw["attribute_changes"][0]["is_hard"] = False
    return raw


def test_request_soft_survives_followup_but_not_new_task_or_new_session(tmp_path):
    h = host(tmp_path)
    t = turn(1, "这次价格倾向三千以内，不是硬预算，别长期保存。")
    assert process(h, t, temporary(t))["new_memory_ids"] == []
    t = turn(3, "给我推荐手机。")
    result = process(h, t, plain(t))
    assert ids(result) == ["B", "C", "A"]
    assert ids({"ranking": h.rank_current(use_long_term_memory=False)}) == ["B", "C", "A"]
    assert result["ranking"]["n_filtered"] == 0
    assert JsonlMemoryStore(tmp_path / "memory").list_attribute_memories("u", as_of=after(t)) == ()
    t = turn(5, "现在重新开始选手机，给我推荐。")
    assert ids(process(h, t, payload(t, amount=None, recommend=True))) == ["A", "B", "C"]
    t = turn(7, "给我推荐手机。")
    assert ids(process(host(tmp_path, "new"), t, plain(t))) == ["A", "B", "C"]


def test_current_soft_does_not_cancel_current_hard_budget_and_end_clears_both(tmp_path):
    h = host(tmp_path)
    t = turn(1, "这次预算三千。")
    process(h, t, payload(t))
    t = turn(3, "这次价格更倾向四千五以内，给我推荐手机。")
    result = process(h, t, temporary(t, amount=4500, recommend=True))
    assert ids(result) == ["B", "C"]
    t = turn(5, "结束这次选手机。")
    process(h, t, payload(t, op="end", amount=None))
    t = turn(7, "给我推荐手机。")
    assert ids(process(h, t, plain(t))) == ["A", "B", "C"]


def test_other_request_soft_does_not_become_stable_session_preference(tmp_path):
    h = host(tmp_path)
    t = turn(1, "我叔叔这次更倾向三千以内，给他推荐手机。")
    who = subject("other", [quote(t, "我叔叔")])
    assert ids(process(h, t, temporary(t, who=who, recommend=True))) == ["B", "C", "A"]
    t = turn(3, "给我推荐手机。")
    assert ids(process(h, t, plain(t))) == ["A", "B", "C"]
    t = turn(5, "给我叔叔推荐手机。")
    raw = plain(t)
    raw["candidate"]["request"]["beneficiary"] = who
    assert ids(process(h, t, raw)) == ["A", "B", "C"]


def test_request_soft_retraction_ignores_soft_only_and_preserves_long_term(tmp_path):
    h = host(tmp_path)
    t = turn(1, "以后价格倾向三千以内。")
    raw = durable(t)
    raw["attribute_changes"][0]["is_hard"] = False
    process(h, t, raw)
    t = turn(3, "这次先不考虑价格软偏好，给我推荐手机。")
    result = process(h, t, temporary(t, amount=None, active=False, recommend=True))
    assert ids(result) == ["A", "B", "C"]
    assert (
        len(JsonlMemoryStore(tmp_path / "memory").list_attribute_memories("u", as_of=after(t))) == 1
    )
    t = turn(5, "结束这次选手机。")
    process(h, t, payload(t, op="end", amount=None))
    t = turn(7, "给我推荐手机。")
    assert ids(process(h, t, plain(t))) == ["B", "C", "A"]


def test_new_temporary_soft_evicts_old_receipt_even_without_disk_change(tmp_path):
    h = host(tmp_path)
    t = turn(1, "给我推荐手机。")
    old = process(h, t, plain(t))["state_token"]
    t = turn(3, "这次价格倾向三千以内。")
    process(h, t, temporary(t))
    assert h.state_token() != old


def test_request_scope_rejects_hard_attribute_duplicate_route():
    t = turn(1, "这次预算三千。")
    with pytest.raises(ExtractionError, match="attribute_scope"):
        decode_interaction(json.dumps(durable(t, scope="request")), turns=(t,))


def test_other_temporary_ignore_suppresses_session_soft_then_end_restores_it(tmp_path):
    h = host(tmp_path)
    t = turn(1, "我叔叔价格一直更倾向三千以内。")
    who = subject("other", [quote(t, "我叔叔")])
    raw = durable(t, who=who)
    raw["attribute_changes"][0]["is_hard"] = False
    process(h, t, raw)
    t = turn(3, "这次先不考虑我叔叔的价格偏好，给他推荐。")
    assert ids(process(h, t, temporary(t, who=who, amount=None, active=False, recommend=True))) == [
        "A",
        "B",
        "C",
    ]
    t = turn(5, "结束这次选购。")
    process(h, t, payload(t, op="end", amount=None, who=who))
    t = turn(7, "给我叔叔推荐手机。")
    raw = plain(t)
    raw["candidate"]["request"]["beneficiary"] = who
    assert ids(process(h, t, raw)) == ["B", "C", "A"]


def test_new_long_term_soft_replaces_earlier_temporary_same_attribute(tmp_path):
    h = host(tmp_path)
    t = turn(1, "这次价格更倾向三千以内。")
    process(h, t, temporary(t))
    t = turn(3, "以后价格更倾向四千五以内。")
    raw = durable(t, amount=4500)
    raw["attribute_changes"][0]["is_hard"] = False
    process(h, t, raw)
    t = turn(5, "给我推荐手机。")
    assert ids(process(h, t, plain(t))) == ["A", "B", "C"]


def test_failed_commit_does_not_publish_temporary_task(tmp_path, monkeypatch):
    from memrecagent.memory_commit import CommitError

    h = host(tmp_path)
    old = h.state_token()
    t = turn(1, "这次价格更倾向三千以内。")

    def fail(*args, **kwargs):
        raise CommitError("synthetic_commit_failure")

    with monkeypatch.context() as patch:
        patch.setattr(h._memory, "commit", fail)
        with pytest.raises(CommitError):
            process(h, t, temporary(t))
    assert h.state_token() == old
    t = turn(3, "给我推荐手机。")
    assert ids(process(h, t, plain(t))) == ["A", "B", "C"]


def test_deleted_temporary_soft_is_released_before_next_model_input(tmp_path):
    from memrecagent.memory_commit import CommitError

    h = host(tmp_path)
    t = turn(1, "这次价格更倾向三千以内。")
    process(h, t, temporary(t))
    store = JsonlMemoryStore(tmp_path / "memory")
    store.delete_attribute("u", category="phone", attribute="price_cny", deleted_at=after(t))
    with pytest.raises(CommitError):
        h.refresh_read_state()
    assert h._attribute_request is None and h._request_soft_preferences == ()
    t = turn(3, "给我推荐手机。")
    assert ids(process(h, t, plain(t))) == ["A", "B", "C"]


@pytest.mark.parametrize("active", [False, True])
def test_other_current_attribute_replaces_all_stable_predicates_of_same_attribute(tmp_path, active):
    h = host(tmp_path)
    t = turn(1, "我叔叔价格一直倾向三千以内。")
    who = subject("other", [quote(t, "我叔叔")])
    raw = durable(t, who=who)
    raw["attribute_changes"][0]["is_hard"] = False
    process(h, t, raw)
    t = turn(3, "我叔叔这次价格倾向三千五以上，或者不看价格偏好，请推荐。")
    raw = temporary(t, who=who, amount=3500 if active else None, active=active, recommend=True)
    raw["attribute_changes"][0]["operator"] = ">="
    result = process(h, t, raw)["ranking"]
    assert result["items"][0]["item_id"] == "A"
    assert result["items"][0].get("attribute_preference_signal", 0) == (1 if active else 0)
    assert len(result.get("soft_attribute_preferences", [])) == int(active)


def test_ignore_soft_never_removes_long_term_hard_budget(tmp_path):
    h = host(tmp_path)
    t = turn(1, "以后我的硬预算上限三千。")
    process(h, t, durable(t))
    t = turn(3, "这次忽略价格软偏好，给我推荐手机。")
    assert ids(process(h, t, temporary(t, active=False, amount=None, recommend=True))) == ["B", "C"]


@pytest.mark.parametrize("field", ["relaxations", "soft_relaxations"])
def test_deleted_attribute_old_relaxation_cannot_be_replayed(tmp_path, field):
    from datetime import timedelta

    from test_memory import T0
    from test_product_attributes import request
    from test_soft_attribute_ranking import priced

    from memrecagent.product_attributes import RequestRelaxation

    store = JsonlMemoryStore(tmp_path)
    store.delete_attribute(
        "demo-user", category="phone", attribute="price_cny", deleted_at=T0 + timedelta(hours=1)
    )
    ignore = RequestRelaxation("demo-user", "S2", "self", "R1", "price_cny", "S2:T1", T0)
    with pytest.raises(ValueError, match="deleted_attribute_evidence"):
        priced(tmp_path).recommend(**request(), **{field: (ignore,)})


@pytest.mark.parametrize("kind", ["soft", "hard"])
def test_bridge_expires_old_ranking_after_ram_task_update_without_reanalysis(tmp_path, kind):
    from test_conversation_bridge import request
    from test_conversation_receipt_privacy import assert_expired
    from test_memory_analyst_host import FixedStages

    from memrecagent.conversation_bridge import ConversationBridge

    t1, t2 = turn(1, "给我推荐手机。"), turn(3, "这次价格倾向三千以内，预算三千。")
    raw1 = plain(t1)
    raw2 = temporary(t2) if kind == "soft" else payload(t2)
    client = FixedStages(raw1, raw1, raw2, raw2)
    bridge = ConversationBridge(host(tmp_path), complete=client)
    old_request = request(t1, id="old")
    assert bridge.handle(old_request)["ok"]
    update = request(t2, id="update")
    accepted = bridge.handle(update)
    assert accepted["ok"]
    assert bridge.handle(update) == accepted
    assert_expired(bridge.handle(old_request))
    assert len(client.seen) == 4
