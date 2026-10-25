"""四组必须改变真实记忆和模型信号，不能把回退当算法收益。"""

from datetime import timedelta

import pytest
from test_memory import T0, sample_memory
from test_memory_products import engine, request
from test_product_service import service

from memrecagent.memory_products import MemoryProductRecommender, ProductEntityFact
from memrecagent.memory_store import JsonlMemoryStore


def test_rank_normalisation_preserves_ties_before_memory_filter(tmp_path):
    ranker = engine(tmp_path)
    ranker._service.counts = {"A": 100, "B": 1, "C": 1}
    JsonlMemoryStore(tmp_path).add(sample_memory(entity="品牌A"))
    result = ranker.recommend(**request(), normalisation="rank")
    assert [row["normalised_base_score"] for row in result["items"]] == [0.25, 0.25]
    assert [row["item_id"] for row in result["items"]] == ["B", "C"]
    assert result["normalisation"] == "rank"


def test_rank_constant_and_single_pools_do_not_invent_preference(tmp_path):
    ranker = engine(tmp_path)
    ranker._service.counts = {"A": 1, "B": 1, "C": 1}
    constant = ranker.recommend(**request(), normalisation="rank")
    assert [row["score"] for row in constant["items"]] == [0, 0, 0]
    ranker._service.seen = {"demo-user": {"A", "B"}}
    single = ranker.recommend(**request(), normalisation="rank")
    assert [row["score"] for row in single["items"]] == [0]


@pytest.mark.parametrize("normalisation", ["zscore", "", None, True])
def test_unknown_normalisation_fails_instead_of_silently_changing_method(tmp_path, normalisation):
    with pytest.raises(ValueError, match="归一化"):
        engine(tmp_path).recommend(**request(), normalisation=normalisation)


def fitted_ranker_inputs(root):
    base = service()
    # 历史是否已见四组一致；固定模型给u的真实分为A0、B-2、C-1。
    base.seen = {"u": set()}
    facts = tuple(
        ProductEntityFact(k, "phone", f"品牌{k}", f"synthetic:{k}", T0 - timedelta(days=1))
        for k in base.counts
    )
    store = JsonlMemoryStore(root)
    store.add(sample_memory(user_id="u", entity="品牌A"))
    store.add(sample_memory(memory_id="prefer-b", user_id="u", entity="品牌B", relation="prefer"))
    return base, store, facts


def test_four_arms_use_real_model_and_memory_without_mutating_source(tmp_path):
    from memrecagent.memory_ablation import run_ranking_ablation

    base, store, facts = fitted_ranker_inputs(tmp_path)
    before = store.snapshot_token("u")
    result = run_ranking_ablation(
        base, store, facts, request=request() | {"user_id": "u", "strategy": "bpr"}
    )
    assert {g: [i["item_id"] for i in r["ranking"]["items"]] for g, r in result.items()} == {
        "G00": ["A", "B", "C"],
        "G10": ["B", "C"],
        "G01": ["A", "C", "B"],
        "G11": ["B", "C"],
    }
    assert result["G11"]["algorithm_effective"] is True
    assert result["G01"]["algorithm_effective"] is True
    assert result["G00"]["algorithm_effective"] is False
    assert result["G11"]["ranking"]["base_score_kind"] == "dot_product"
    assert store.snapshot_token("u") == before
    assert len(store.list_memories("u", as_of=T0 + timedelta(days=1))) == 2


@pytest.mark.parametrize("user_id,beneficiary", [("new", "self"), ("u", "uncle"), ("u", None)])
def test_cold_or_nonself_is_explicitly_not_effective_algorithm(tmp_path, user_id, beneficiary):
    from memrecagent.memory_ablation import run_ranking_ablation

    base, store, facts = fitted_ranker_inputs(tmp_path)
    result = run_ranking_ablation(
        base,
        store,
        facts,
        request=request()
        | {"user_id": user_id, "beneficiary_id": beneficiary, "strategy": "bpr_cold_popularity"},
    )
    assert result["G01"]["algorithm_effective"] is False
    assert result["G11"]["algorithm_effective"] is False
    assert result["G01"]["ranking"]["used_strategy"] == "popularity"


def test_unknown_mf_vector_is_not_counted_even_if_route_says_mf(tmp_path):
    from memrecagent.memory_ablation import run_ranking_ablation

    base, store, facts = fitted_ranker_inputs(tmp_path)
    result = run_ranking_ablation(
        base, store, facts, request=request() | {"user_id": "missing", "strategy": "mf"}
    )
    assert result["G01"]["ranking"]["used_strategy"] == "mf"
    assert result["G01"]["algorithm_effective"] is False
    assert result["G01"]["algorithm_reason"] == "no_active_model_profile"


def test_memory_off_keeps_current_constraints_and_seen(tmp_path):
    from memrecagent.memory_ablation import run_ranking_ablation

    base, store, facts = fitted_ranker_inputs(tmp_path)
    base.seen = {"u": {"C"}}
    result = run_ranking_ablation(
        base,
        store,
        facts,
        request=request() | {"user_id": "u", "strategy": "bpr", "requested_entity": "品牌B"},
    )
    assert all([i["item_id"] for i in row["ranking"]["items"]] == ["B"] for row in result.values())


def test_ablation_rejects_non_algorithm_strategy_and_does_not_claim_two_toggles(tmp_path):
    from memrecagent.memory_ablation import run_ranking_ablation

    base, store, facts = fitted_ranker_inputs(tmp_path)
    with pytest.raises(ValueError, match="算法"):
        run_ranking_ablation(base, store, facts, request=request())


def test_memory_off_does_not_revive_deleted_current_evidence(tmp_path):
    from memrecagent.memory_ablation import run_ranking_ablation
    from memrecagent.memory_policy import ContextPreference

    base, store, facts = fitted_ranker_inputs(tmp_path)
    store.delete_entity("u", category="phone", entity="品牌B", deleted_at=T0 + timedelta(hours=2))
    context = ContextPreference(
        "u", "S2", "self", "phone", "品牌B", "exclude", "old", "合成旧证据", T0
    )
    result = run_ranking_ablation(
        base,
        store,
        facts,
        request=request() | {"user_id": "u", "strategy": "bpr", "context_preferences": (context,)},
    )
    assert [i["item_id"] for i in result["G00"]["ranking"]["items"]] == ["A", "B", "C"]


def test_all_groups_share_permission_and_detect_permission_aba(tmp_path, monkeypatch):
    from memrecagent import memory_ablation
    from memrecagent.memory_use_policy import JsonMemoryUseStore, MemoryUsePolicy

    base, store, facts = fitted_ranker_inputs(tmp_path)
    original = MemoryProductRecommender.recommend
    count = 0

    def changed(self, **kwargs):
        nonlocal count
        value = original(self, **kwargs)
        count += 1
        if count == 1:
            settings = JsonMemoryUseStore(tmp_path / "use-policy")
            settings.set("u", MemoryUsePolicy(personalisation=False))
            settings.set("u", MemoryUsePolicy())
        return value

    monkeypatch.setattr(MemoryProductRecommender, "recommend", changed)
    with pytest.raises(ValueError, match="source_state_changed"):
        memory_ablation.run_ranking_ablation(
            base, store, facts, request=request() | {"user_id": "u", "strategy": "bpr"}
        )
