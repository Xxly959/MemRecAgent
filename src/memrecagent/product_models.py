"""将已有向量模型用于显式商品请求，不消费未来标签。"""

from collections.abc import Iterable, Mapping, Set
from pathlib import Path

import torch

from memrecagent.bpr_evaluation import FittedBPR
from memrecagent.mf import MatrixFactorization
from memrecagent.mf_evaluation import FittedMF


def load_fitted_model(path: Path) -> FittedMF:
    """恢复本地生成的推理检查点；包层另行校验文件哈希，不加载执行对象。"""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    kind = payload["model_kind"]
    users, items = payload["users"], payload["items"]
    for vocabulary, limit in [(users, 150000), (items, 5000)]:
        if (
            not isinstance(vocabulary, dict)
            or not 0 < len(vocabulary) <= limit
            or any(
                not isinstance(key, str) or type(index) is not int
                for key, index in vocabulary.items()
            )
            or set(vocabulary.values()) != set(range(len(vocabulary)))
        ):
            raise ValueError("检查点词表不是受控的连续编号")
    factors = payload["config"]["factors"]
    if type(factors) is not int or not 1 <= factors <= 16 or kind not in {"mf", "bpr"}:
        raise ValueError("检查点模型类型或维度超出阶段范围")
    state = payload["state_dict"]
    if any(
        not isinstance(t, torch.Tensor) or t.dtype != torch.float32 or not torch.isfinite(t).all()
        for t in state.values()
    ):
        raise ValueError("检查点必须是有限float32张量")
    model = MatrixFactorization(len(users), len(items), factors, seed=42)
    model.load_state_dict(state, strict=True)
    model.eval()
    seen = {user: set(values) for user, values in payload["seen"].items()}
    if not seen.keys() <= users.keys() or any(
        not values <= items.keys() for values in seen.values()
    ):
        raise ValueError("检查点已见集合与词表不符")
    if kind == "mf":
        return FittedMF(model, users, items, (), seen)
    active = frozenset(payload["active_users"])
    if not active <= users.keys():
        raise ValueError("检查点活跃用户与词表不符")
    return FittedBPR(model, users, items, (), seen, active_users=active)


@torch.no_grad()
def score_model(fitted: FittedMF, user_id: str) -> dict[str, float]:
    """点积原始分数不是概率；未拟合/无有效三元组用户统一零分。"""
    catalog = list(fitted.items)
    if user_id not in fitted.users or (
        isinstance(fitted, FittedBPR) and user_id not in fitted.active_users
    ):
        return {item: 0.0 for item in catalog}
    fitted.model.eval()
    item_ids = torch.tensor([fitted.items[item] for item in catalog], dtype=torch.long)
    user_ids = torch.full_like(item_ids, fitted.users[user_id])
    scores = fitted.model(user_ids, item_ids)
    if not torch.isfinite(scores).all():
        raise FloatingPointError("商品模型产生非有限分数")
    return dict(zip(catalog, scores.tolist(), strict=True))


def predict_model(
    fitted: FittedMF,
    user_ids: Iterable[str],
    seen: Mapping[str, Set[str]],
    k: int = 10,
) -> dict[str, list[str]]:
    """模型只打分历史词表；过滤由完整合法历史确定，保留负/零分。"""
    if type(k) is not int or k <= 0:
        raise ValueError("k必须是正整数")
    predictions = {}
    for user in sorted(set(user_ids)):
        scores = score_model(fitted, user)
        candidates = [item for item in scores if item not in seen.get(user, set())]
        predictions[user] = sorted(candidates, key=lambda item: (-scores[item], item))[:k]
    return predictions
