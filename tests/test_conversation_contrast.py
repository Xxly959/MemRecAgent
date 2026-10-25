"""对照只能读取宿主实际接受的请求，不由评分答案构造本次人物和预算。"""

import pytest
from test_conversation_products import process
from test_memory_analyst_host import FixedStages
from test_memory_interaction import after, turn, value
from test_request_attributes import host, ids, payload, plain

from memrecagent.memory_commit import CommitError
from memrecagent.memory_store import JsonlMemoryStore


def test_actual_accepted_request_can_be_contrasted_without_writing_or_changing_current_budget(
    tmp_path,
):
    h = host(tmp_path)
    t = turn(1, "我长期喜欢品牌C。")
    process(h, t, value(t, entity="品牌C", relation="prefer"))
    t = turn(3, "本次预算三千，给我推荐手机。")
    process(h, t, payload(t, recommend=True))
    store = JsonlMemoryStore(tmp_path / "memory")
    before = store.snapshot_token("u")
    assert [i["item_id"] for i in h.rank_current(use_long_term_memory=False)["items"]] == ["B", "C"]
    assert [i["item_id"] for i in h.rank_current(use_long_term_memory=True)["items"]] == ["C", "B"]
    assert store.snapshot_token("u") == before


def test_new_nonrecommendation_or_external_change_invalidates_old_contrast_request(tmp_path):
    h = host(tmp_path)
    t = turn(1, "给我推荐手机。")
    process(h, t, plain(t))
    JsonlMemoryStore(tmp_path / "memory").delete_entity(
        "u", category="phone", entity="品牌C", deleted_at=after(t)
    )
    with pytest.raises(CommitError):
        h.rank_current(use_long_term_memory=False)
    fresh = host(tmp_path, "new")
    t = turn(3, "给我推荐手机。")
    process(fresh, t, plain(t))
    t = turn(5, "今天天气不错。")
    process(fresh, t, value(t, "unrelated"))
    with pytest.raises(CommitError):
        fresh.rank_current(use_long_term_memory=False)


def test_direct_analysis_control_uses_one_real_client_call_with_same_product_host(tmp_path):
    h = host(tmp_path)
    t = turn(1, "这次预算三千，给我推荐手机。")
    raw = payload(t, recommend=True)
    fixed = FixedStages(raw, raw)
    result = h.process_user(
        t, complete=fixed, as_of=after(t), strategy="popularity", k=3, analysis_mode="direct"
    )
    assert len(fixed.seen) == 1 and ids(result) == ["B", "C"]
