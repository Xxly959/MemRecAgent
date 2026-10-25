"""真实存储/策略/已有模型分与候选过滤融合，不伪造基础推荐结果。"""

from dataclasses import replace
from datetime import timedelta

import pytest
from test_memory import T0, sample_memory
from test_product_service import service

from memrecagent.memory_policy import ContextPreference
from memrecagent.memory_products import MemoryProductRecommender, ProductEntityFact
from memrecagent.memory_store import JsonlMemoryStore
from memrecagent.memory_use_policy import JsonMemoryUseStore, MemoryUsePolicy

AT = T0 + timedelta(days=1)


def engine(root, *, count=3):
    base = service()
    base.counts = {chr(65 + i): 10 - i for i in range(count)}
    base.seen = {}
    facts = tuple(
        ProductEntityFact(
            item_id=k,
            category="phone",
            entity=f"品牌{k}",
            evidence_ref=f"synthetic:{k}",
            available_at=T0 - timedelta(days=2),
        )
        for k in base.counts
    )
    return MemoryProductRecommender(
        base, JsonlMemoryStore(root), facts, facts_bundle_id=base.bundle_id
    )


def request(**changes):
    return dict(
        user_id="demo-user",
        session_id="S2",
        beneficiary_id="self",
        category="phone",
        as_of=AT,
        strategy="popularity",
        k=3,
        **changes,
    )


def test_memory_actually_filters_and_changes_ranking_after_reopening_store(tmp_path):
    store = JsonlMemoryStore(tmp_path)
    store.add(sample_memory(entity="品牌A", evidence_text="合成：以后别推荐品牌A"))
    store.add(
        sample_memory(
            memory_id="like-c",
            entity="品牌C",
            relation="prefer",
            evidence_ref="S1:T4",
            evidence_text="合成：我喜欢品牌C",
        )
    )
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir() if p.is_file()}
    ranker = engine(tmp_path)
    result = ranker.recommend(**request())
    assert [p["item_id"] for p in result["items"]] == ["C", "B"]
    assert [p["base_score"] for p in result["items"]] == [8.0, 9.0]
    assert [p["normalised_base_score"] for p in result["items"]] == [0.0, 0.5]
    assert [p["score"] for p in result["items"]] == [0.75, 0.5]
    assert result["items"][0]["signals"] == [
        {
            "entity": "品牌C",
            "relation": "prefer",
            "scope": "long_term",
            "reference": "like-c",
            "evidence_ref": "S1:T4",
            "value": 1,
        }
    ]
    assert result["filtered"][0]["item_id"] == "A"
    assert {p.name: p.read_bytes() for p in tmp_path.iterdir() if p.is_file()} == before
    # 空记忆对照保持相同算法/目录，绝非训练结果或LLM效果。
    control = engine(tmp_path / "control").recommend(**request())
    assert [p["item_id"] for p in control["items"]] == ["A", "B", "C"]


def test_current_explicit_brand_overrides_old_exclusion_without_modifying_it(tmp_path):
    JsonlMemoryStore(tmp_path).add(sample_memory(entity="品牌A"))
    ranker = engine(tmp_path)
    current = ranker.recommend(**request(requested_entity="品牌A"))
    assert [p["item_id"] for p in current["items"]] == ["A"]
    assert current["overridden_memory_ids"] == ["m1"]
    assert [p["item_id"] for p in ranker.recommend(**request())["items"]] == ["B", "C"]


def test_other_person_does_not_use_own_memory_and_unknown_conflict_clarifies(tmp_path):
    JsonlMemoryStore(tmp_path).add(sample_memory(entity="品牌A"))
    ranker = engine(tmp_path)
    other = ranker.recommend(**(request() | {"beneficiary_id": "relative:uncle"}))
    assert [p["item_id"] for p in other["items"]] == ["A", "B", "C"]
    assert all(not p["signals"] for p in other["items"])
    unclear = ranker.recommend(**(request(requested_entity="品牌A") | {"beneficiary_id": None}))
    assert unclear["status"] == "clarify" and unclear["items"] == []


def test_relative_context_is_session_bound_and_has_only_current_sources(tmp_path):
    context = ContextPreference(
        "demo-user",
        "S2",
        "relative:uncle",
        "phone",
        "品牌C",
        "prefer",
        "S2:T1",
        "合成：叔叔喜欢品牌C",
        T0 + timedelta(hours=2),
    )
    ranker = engine(tmp_path)
    result = ranker.recommend(
        **(request(context_preferences=(context,)) | {"beneficiary_id": "relative:uncle"})
    )
    assert result["items"][1]["item_id"] == "C"
    assert result["items"][1]["signals"][0]["scope"] == "current"
    with pytest.raises(ValueError):
        ranker.recommend(**(request(context_preferences=(context,)) | {"session_id": "S3"}))
    assert JsonlMemoryStore(tmp_path).list_memories("demo-user", as_of=AT) == ()


def test_filter_full_pool_before_topk_and_never_return_seen_items(tmp_path):
    ranker = engine(tmp_path)
    ranker._service.seen = {"demo-user": {"A"}}
    JsonlMemoryStore(tmp_path).add(sample_memory(entity="品牌B"))
    result = ranker.recommend(**(request() | {"k": 1}))
    assert [p["item_id"] for p in result["items"]] == ["C"]
    assert result["n_candidates"] == 2


def test_mf_uses_real_negative_scores_without_training_or_mutating_parameters(tmp_path):
    base = service()
    facts = tuple(
        ProductEntityFact(k, "phone", f"品牌{k}", f"synthetic:{k}", T0 - timedelta(days=1))
        for k in base.counts
    )
    ranker = MemoryProductRecommender(
        base, JsonlMemoryStore(tmp_path), facts, facts_bundle_id=base.bundle_id
    )
    baseline = base.recommend("u", bundle_id=base.bundle_id, strategy="mf")
    result = ranker.recommend(**(request() | {"user_id": "u", "strategy": "mf"}))
    assert [p["item_id"] for p in result["items"]] == ["C", "B"]
    assert [p["base_score"] for p in result["items"]] == [-1.0, -2.0]
    assert [p["score"] for p in result["items"]] == [1.0, 0.0]
    assert base.recommend("u", bundle_id=base.bundle_id, strategy="mf") == baseline


def test_future_facts_and_memories_do_not_apply_and_disable_blocks_personalisation(tmp_path):
    ranker = engine(tmp_path)
    JsonlMemoryStore(tmp_path).add(sample_memory(entity="品牌A", occurred_at=AT, available_at=AT))
    assert ranker.recommend(**request())["items"][0]["item_id"] == "A"
    store = JsonlMemoryStore(tmp_path)
    store.add(sample_memory(memory_id="old-b", entity="品牌B"))
    JsonMemoryUseStore(tmp_path / "use-policy").set(
        "demo-user", MemoryUsePolicy(personalisation=False)
    )
    result = ranker.recommend(**request())
    assert len(result["items"]) == 3 and all(not p["signals"] for p in result["items"])
    future = tuple(replace(f, available_at=AT) for f in ranker._facts)
    excluded = MemoryProductRecommender(
        ranker._service, store, future, facts_bundle_id=ranker._service.bundle_id
    ).recommend(**request())
    assert excluded["items"] == []


def test_state_token_changes_after_policy_aba_and_deletion(tmp_path):
    ranker = engine(tmp_path)
    store = JsonlMemoryStore(tmp_path)
    store.add(sample_memory(entity="品牌A"))
    old = ranker.recommend(**request())["state_token"]
    settings = JsonMemoryUseStore(tmp_path / "use-policy")
    settings.set("demo-user", MemoryUsePolicy(personalisation=False))
    settings.set("demo-user", MemoryUsePolicy())
    assert ranker.state_token("demo-user") != old
    old = ranker.state_token("demo-user")
    store.delete_entity("demo-user", category="phone", entity="品牌A", deleted_at=AT)
    assert ranker.state_token("demo-user") != old


@pytest.mark.parametrize("weight", [True, -1, 2, float("nan"), float("inf")])
def test_invalid_weight_is_rejected_before_ranking(tmp_path, weight):
    with pytest.raises(ValueError):
        engine(tmp_path).recommend(**request(soft_weight=weight))


def test_fact_identity_duplicates_missing_source_and_future_bundle_are_not_guessed(tmp_path):
    ranker = engine(tmp_path)
    for facts in [ranker._facts[:-1], (*ranker._facts, ranker._facts[0])]:
        with pytest.raises(ValueError):
            MemoryProductRecommender(
                ranker._service,
                JsonlMemoryStore(tmp_path),
                facts,
                facts_bundle_id=ranker._service.bundle_id,
            )
    with pytest.raises(ValueError):
        MemoryProductRecommender(
            ranker._service, JsonlMemoryStore(tmp_path), ranker._facts, facts_bundle_id="different"
        )
    with pytest.raises(ValueError):
        replace(ranker._facts[0], evidence_ref="")


def test_full_candidates_beyond_service_top50_are_still_eligible(tmp_path):
    ranker = engine(tmp_path, count=60)
    # 所有计数保持合法；最后商品没有进入基础Top50，但软偏好能把它提上来。
    ranker._service.counts = {k: 100 - i for i, k in enumerate(ranker._service.counts)}
    last = ranker._facts[-1]
    JsonlMemoryStore(tmp_path).add(sample_memory(entity=last.entity, relation="prefer"))
    result = ranker.recommend(**(request() | {"k": 50}))
    assert last.item_id in [p["item_id"] for p in result["items"]]
    assert result["n_candidates"] == 60


def test_model_score_user_normalisation_matches_existing_service(tmp_path):
    base = service()
    facts = tuple(
        ProductEntityFact(k, "phone", f"品牌{k}", f"synthetic:{k}", T0 - timedelta(days=1))
        for k in base.counts
    )
    ranker = MemoryProductRecommender(
        base, JsonlMemoryStore(tmp_path), facts, facts_bundle_id=base.bundle_id
    )
    result = ranker.recommend(**(request() | {"user_id": " u ", "strategy": "mf"}))
    assert [p["base_score"] for p in result["items"]] == [-1.0, -2.0]


def test_durable_state_change_during_real_scoring_blocks_result(tmp_path, monkeypatch):
    base = service()
    facts = tuple(
        ProductEntityFact(k, "phone", f"品牌{k}", f"synthetic:{k}", T0 - timedelta(days=1))
        for k in base.counts
    )
    ranker = MemoryProductRecommender(
        base, JsonlMemoryStore(tmp_path), facts, facts_bundle_id=base.bundle_id
    )
    from memrecagent import memory_products

    original = memory_products.score_model

    def changed(model, user_id):
        JsonMemoryUseStore(tmp_path / "use-policy").set(
            user_id, MemoryUsePolicy(personalisation=False)
        )
        return original(model, user_id)

    monkeypatch.setattr(memory_products, "score_model", changed)
    with pytest.raises(ValueError, match="request_state_changed"):
        ranker.recommend(**(request() | {"strategy": "mf"}))


@pytest.mark.parametrize("strategy", ["mf", "bpr", "bpr_cold_popularity"])
@pytest.mark.parametrize("beneficiary", ["relative:uncle", None])
def test_other_and_unknown_never_use_owner_implicit_model_or_seen(tmp_path, strategy, beneficiary):
    base = service()
    facts = tuple(
        ProductEntityFact(k, "phone", f"品牌{k}", f"synthetic:{k}", T0 - timedelta(days=1))
        for k in base.counts
    )
    ranker = MemoryProductRecommender(
        base, JsonlMemoryStore(tmp_path), facts, facts_bundle_id=base.bundle_id
    )
    result = ranker.recommend(
        **(request() | {"user_id": "u", "strategy": strategy, "beneficiary_id": beneficiary})
    )
    assert [p["item_id"] for p in result["items"]] == ["A", "B", "C"]
    assert [p["base_score"] for p in result["items"]] == [9.0, 3.0, 2.0]
    assert result["requested_strategy"] == strategy
    assert result["used_strategy"] == "popularity"
    assert result["routing_reason"] == "non_self_public_baseline"
