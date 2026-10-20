"""冻结商品推理服务；不访问未来标签，不在请求中拟合或更新参数。"""

from dataclasses import dataclass

from memrecagent.bpr_evaluation import FittedBPR
from memrecagent.fallback import recommend_with_fallback
from memrecagent.mf_evaluation import FittedMF
from memrecagent.product_evaluation import predict_popularity
from memrecagent.product_models import predict_model, score_model


class BundleVersionError(ValueError):
    """请求版本与当前进程冻结版本不匹配。"""


@dataclass
class ProductService:
    """调用方只能选择已冻结策略；展示字段保留来源和未知时间状态。"""

    bundle_id: str
    as_of_ms: int
    counts: dict[str, int]
    seen: dict[str, set[str]]
    products: dict[str, dict]
    models: dict[str, FittedMF]
    dataset_revision: str | None
    metadata_url: str | None

    def health(self) -> dict:
        """只返回版本和容量，不返回原始用户列表。"""
        return {
            "status": "ready",
            "bundle_id": self.bundle_id,
            "as_of_ms": self.as_of_ms,
            "n_catalog_items": len(self.counts),
            "strategies": ["popularity", "mf", "bpr", "bpr_cold_popularity"],
        }

    def recommend(self, user_id: str, *, bundle_id: str, strategy: str, k: int = 10) -> dict:
        """返回稳定、合法的商品列表，原始得分与实际路由可核验。"""
        if bundle_id != self.bundle_id:
            raise BundleVersionError("版本不匹配，请先读取/health取得冻结版本")
        if not isinstance(user_id, str) or not user_id.strip() or len(user_id.strip()) > 128:
            raise ValueError("user_id必须是1至128字符的非空标识")
        if type(k) is not int or not 1 <= k <= 50:
            raise ValueError("k必须是1至50的整数")
        if strategy not in self.health()["strategies"]:
            raise ValueError("不支持的推荐策略，禁止静默切换")
        user_id = user_id.strip()
        cold = user_id not in self.seen
        used = (
            "popularity"
            if strategy == "bpr_cold_popularity" and cold
            else ("bpr" if strategy == "bpr_cold_popularity" else strategy)
        )
        if used == "popularity":
            ranked = predict_popularity(self.counts, self.seen, [user_id], k)[user_id]
            scores = self.counts
            model_known, active = None, None
        else:
            if used not in self.models or self.models[used].items.keys() != self.counts.keys():
                raise ValueError("模型缺失或模型词表与冻结候选不符")
            fitted = self.models[used]
            ranked = predict_model(fitted, [user_id], self.seen, k)[user_id]
            scores = score_model(fitted, user_id)
            model_known = user_id in fitted.users
            active = (
                user_id in fitted.active_users if isinstance(fitted, FittedBPR) else model_known
            )
        if strategy == "bpr_cold_popularity":
            # 复用Phase1的窄路由，不将无正向/无模型向量/空列表改称冷用户。
            ranked = recommend_with_fallback(
                user_id, self.seen.keys(), ranked if not cold else [], ranked if cold else [], k
            )
        items = []
        for item in ranked:
            metadata = self.products.get(item)
            items.append(
                {
                    "item_id": item,
                    "title": metadata.get("title") if metadata else None,
                    "categories": list(metadata.get("categories", [])) if metadata else [],
                    "score": float(scores[item]),
                    "score_kind": "positive_reviewer_count"
                    if used == "popularity"
                    else "dot_product",
                    "source": {
                        "revision": self.dataset_revision,
                        "url": self.metadata_url,
                        "file": metadata.get("source"),
                        "line_number": metadata.get("line_number"),
                    }
                    if metadata
                    else None,
                }
            )
        candidates = self.counts.keys() - self.seen.get(user_id, set())
        return {
            "bundle_id": self.bundle_id,
            "as_of_ms": self.as_of_ms,
            "requested_strategy": strategy,
            "used_strategy": used,
            "items": items,
            "metadata_snapshot": {
                "revision": self.dataset_revision,
                "available_at": None,
                "usage": "display_only",
            },
            "empty_reason": "no_unseen_candidates" if not candidates else None,
            "trace": {
                "cold_user": cold,
                "model_known_user": model_known,
                "active_model_user": active,
                "n_catalog_items": len(self.counts),
                "n_candidates": len(candidates),
                "n_seen_filtered": len(self.counts) - len(candidates),
                "requested_k": k,
                "returned": len(items),
                "fallback_reason": "no_history"
                if strategy == "bpr_cold_popularity" and cold
                else None,
            },
        }
