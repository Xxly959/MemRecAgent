"""商品模型适配只消费历史模型与显式用户，不要求未来标签。"""

import pytest
import torch

from memrecagent.bpr_evaluation import FittedBPR
from memrecagent.mf import MatrixFactorization
from memrecagent.mf_evaluation import FittedMF
from memrecagent.product_models import load_fitted_model, predict_model, score_model


def fitted():
    model = MatrixFactorization(1, 3, factors=2, seed=42)
    with torch.no_grad():
        model.user_vectors.weight.copy_(torch.tensor([[1.0, 0.0]]))
        model.item_vectors.weight.copy_(torch.tensor([[0.0, 0.0], [-2.0, 0.0], [-1.0, 0.0]]))
    return FittedMF(model, {"u": 0}, {"A": 0, "B": 1, "C": 2}, (), {"u": {"A"}})


def test_negative_scores_rank_and_cold_users_are_zero_without_random_preference():
    model = fitted()
    assert score_model(model, "u") == {"A": 0.0, "B": -2.0, "C": -1.0}
    assert score_model(model, "cold") == {"A": 0.0, "B": 0.0, "C": 0.0}
    result = predict_model(model, ["u", "cold"], {"u": {"A"}}, k=2)
    assert result == {"cold": ["A", "B"], "u": ["C", "B"]}


def test_explicit_seen_history_controls_filtering_even_when_model_view_differs():
    model = fitted()
    before = model.model.item_vectors.weight.detach().clone()
    assert predict_model(model, ["u"], {"u": {"A", "C"}}, 10) == {"u": ["B"]}
    assert torch.equal(before, model.model.item_vectors.weight)


def test_bpr_without_training_triples_is_not_scored_by_random_or_untrained_vectors():
    base = fitted()
    model = FittedBPR(
        base.model, base.users, base.items, base.records, base.seen, active_users=frozenset()
    )
    assert score_model(model, "u") == {"A": 0.0, "B": 0.0, "C": 0.0}
    assert predict_model(model, ["u"], {"u": {"A"}}, 2) == {"u": ["B", "C"]}


def test_checkpoint_restores_exact_scores_and_bpr_activity_mask(tmp_path):
    base = fitted()
    payload = {
        "model_kind": "mf",
        "state_dict": base.model.state_dict(),
        "users": base.users,
        "items": base.items,
        "seen": {"u": ["A"]},
        "active_users": None,
        "config": {"factors": 2, "seed": 42},
    }
    path = tmp_path / "mf.pt"
    torch.save(payload, path)
    restored = load_fitted_model(path)
    assert score_model(restored, "u") == {"A": 0.0, "B": -2.0, "C": -1.0}
    payload.update(model_kind="bpr", active_users=[])
    torch.save(payload, tmp_path / "bpr.pt")
    assert score_model(load_fitted_model(tmp_path / "bpr.pt"), "u") == {
        "A": 0.0,
        "B": 0.0,
        "C": 0.0,
    }


def test_checkpoint_rejects_noncontiguous_item_mapping_instead_of_mislabeling_scores(tmp_path):
    base = fitted()
    path = tmp_path / "bad.pt"
    torch.save(
        {
            "model_kind": "mf",
            "state_dict": base.model.state_dict(),
            "users": base.users,
            "items": {"A": 0, "B": 0, "C": 2},
            "seen": {},
            "active_users": None,
            "config": {"factors": 2},
        },
        path,
    )
    with pytest.raises(ValueError, match="词表"):
        load_fitted_model(path)
