"""冻结服务必须返回真实分数与来源，不在请求中训练或补造事实。"""

import pytest
import torch
from test_product_models import fitted

from memrecagent.bpr_evaluation import FittedBPR
from memrecagent.product_service import BundleVersionError, ProductService


def service():
    mf = fitted()
    bpr_base = fitted()
    bpr = FittedBPR(
        bpr_base.model,
        bpr_base.users,
        bpr_base.items,
        (),
        bpr_base.seen,
        active_users=frozenset({"u"}),
    )
    return ProductService(
        bundle_id="synthetic-v1",
        as_of_ms=1656633600000,
        counts={"A": 9, "B": 3, "C": 2},
        seen={"u": {"A"}, "outside": {"OUTSIDE"}, "full": {"A", "B", "C"}},
        products={
            "A": {"title": "商品A", "categories": []},
            "B": {
                "title": "商品B",
                "categories": ["配件"],
                "source": "meta.jsonl",
                "line_number": 3,
            },
        },
        models={"mf": mf, "bpr": bpr},
        dataset_revision="synthetic",
        metadata_url=None,
    )


def test_request_returns_actual_scores_keeps_missing_metadata_and_does_not_update_weights():
    engine = service()
    before = {
        name: model.model.item_vectors.weight.detach().clone()
        for name, model in engine.models.items()
    }
    result = engine.recommend("u", bundle_id="synthetic-v1", strategy="mf", k=10)
    assert [item["item_id"] for item in result["items"]] == ["C", "B"]
    assert [item["score"] for item in result["items"]] == [-1.0, -2.0]
    assert result["items"][0]["title"] is None and result["items"][0]["source"] is None
    assert result["items"][1]["source"]["line_number"] == 3
    assert "price" not in result["items"][1]
    assert result["trace"]["n_candidates"] == 2
    assert result["trace"]["n_seen_filtered"] == 1
    assert result["metadata_snapshot"]["available_at"] is None
    for name, model in engine.models.items():
        assert torch.equal(before[name], model.model.item_vectors.weight)


def test_cold_fallback_is_separate_from_no_candidate_history_and_empty_lists():
    engine = service()
    cold = engine.recommend("new", bundle_id="synthetic-v1", strategy="bpr_cold_popularity", k=2)
    assert [item["item_id"] for item in cold["items"]] == ["A", "B"]
    assert cold["used_strategy"] == "popularity" and cold["trace"]["cold_user"] is True
    outside = engine.recommend(
        "outside", bundle_id="synthetic-v1", strategy="bpr_cold_popularity", k=2
    )
    assert outside["used_strategy"] == "bpr" and outside["trace"]["cold_user"] is False
    assert [item["score"] for item in outside["items"]] == [0.0, 0.0]
    empty = engine.recommend("full", bundle_id="synthetic-v1", strategy="bpr_cold_popularity")
    assert empty["items"] == [] and empty["empty_reason"] == "no_unseen_candidates"


def test_responses_do_not_alias_snapshot_categories():
    engine = service()
    result = engine.recommend("u", bundle_id="synthetic-v1", strategy="mf")
    result["items"][1]["categories"].append("不能回写")
    assert engine.products["B"]["categories"] == ["配件"]


def test_wrong_version_is_distinct_from_invalid_k_and_no_silent_strategy_switch():
    engine = service()
    with pytest.raises(BundleVersionError):
        engine.recommend("u", bundle_id="other", strategy="mf")
    for k in [0, -1, True, 51]:
        with pytest.raises(ValueError):
            engine.recommend("u", bundle_id="synthetic-v1", strategy="mf", k=k)
    with pytest.raises(ValueError):
        engine.recommend("u", bundle_id="synthetic-v1", strategy="unknown")
