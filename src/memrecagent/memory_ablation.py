"""长期记忆×已有模型四组重放；不接收答案标签、不训练、不改源记忆。"""

from memrecagent.bpr_evaluation import FittedBPR
from memrecagent.memory_products import MemoryProductRecommender, ProductEntityFact
from memrecagent.memory_store import JsonlMemoryStore
from memrecagent.product_service import ProductService


def run_ranking_ablation(
    service: ProductService,
    store: JsonlMemoryStore,
    facts: tuple[ProductEntityFact, ...],
    *,
    request: dict,
    normalisation: str = "minmax",
) -> dict[str, dict]:
    """相同当前请求/事实/seen；off仅关闭长期偏好，保留删除和许可。"""
    if request.get("strategy") not in ("mf", "bpr", "bpr_cold_popularity"):
        raise ValueError("四组需要指定已有算法策略，热门只能作为公共对照")
    if "normalisation" in request:
        raise ValueError("归一化应通过独立参数指定")
    if "use_long_term_memory" in request:
        raise ValueError("四组长期记忆开关由组配置指定")
    user_id = request["user_id"]
    source_ranker = MemoryProductRecommender(
        service, store, facts, facts_bundle_id=service.bundle_id
    )
    before = source_ranker.state_token(user_id)
    result = {}
    for group, memory_on, algorithm_on in (
        ("G00", False, False),
        ("G10", True, False),
        ("G01", False, True),
        ("G11", True, True),
    ):
        ranker = MemoryProductRecommender(service, store, facts, facts_bundle_id=service.bundle_id)
        ranking = ranker.recommend(
            **(request | {"strategy": request["strategy"] if algorithm_on else "popularity"}),
            normalisation=normalisation,
            use_long_term_memory=memory_on,
        )
        used = ranking.get("used_strategy")
        fitted = service.models.get(used)
        profile = fitted is not None and user_id.strip() in fitted.users
        if isinstance(fitted, FittedBPR):
            profile = profile and user_id.strip() in fitted.active_users
        effective = bool(algorithm_on and profile and used in ("mf", "bpr"))
        if not algorithm_on:
            reason = "algorithm_off"
        elif ranking["status"] == "clarify":
            reason = "clarification_required"
        elif used == "popularity":
            reason = "public_baseline_route"
        elif not profile:
            reason = "no_active_model_profile"
        else:
            reason = "active_model_profile"
        result[group] = {
            "memory_enabled": memory_on,
            "algorithm_enabled": algorithm_on,
            "algorithm_effective": effective,
            "algorithm_reason": reason,
            "ranking": ranking,
        }
    if source_ranker.state_token(user_id) != before:
        raise ValueError("ablation_source_state_changed")
    return result
