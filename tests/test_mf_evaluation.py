"""Guard catalog causality, checkpoint ownership, and rating-to-ranking conversion."""

import pytest
import torch

from memrecagent import mf_evaluation as m
from memrecagent.movielens import Rating


def history():
    return [Rating("u", "A", 5, 1), Rating("v", "B", 2, 2), Rating("v", "C", 4, 3)]


def config():
    return m.MFConfig(factors=2, max_epochs=3, checkpoint_every=1, batch_size=2, lr=0.05)


def test_raw_scores_rank_negative_candidates_and_filter_low_ratings():
    fitted = m.fit_fixed(history(), config(), epochs=1)
    with torch.no_grad():
        fitted.model.user_vectors.weight[fitted.users["v"]].copy_(torch.tensor([1.0, 0.0]))
        fitted.model.item_vectors.weight[fitted.items["A"]].copy_(torch.tensor([-2.0, 0.0]))
    result = m.rank_mf(fitted, [Rating("v", "A", 5, 10)], k=2)
    assert result["predictions"]["v"] == ["A"]
    assert result["metrics"]["mean"]["recall"] == 1.0


def test_cold_users_do_not_use_random_vectors_or_future_item_vocabulary():
    fitted = m.fit_fixed(history(), config(), epochs=1)
    result = m.rank_mf(fitted, [Rating("new", "Z", 5, 10)], k=2)
    assert "new" not in fitted.users
    assert "Z" not in fitted.items
    assert result["predictions"]["new"] == ["A", "B"]
    assert result["relevant"]["new"] == ["Z"]
    assert result["metrics"]["n_evaluated"] == 1
    assert result["metrics"]["mean"]["recall"] == 0.0


def test_future_labels_change_metrics_not_fitted_scores_or_ranking():
    fitted = m.fit_fixed(history(), config(), epochs=1)
    first = m.rank_mf(fitted, [Rating("u", "C", 5, 10)], k=2)
    second = m.rank_mf(fitted, [Rating("u", "C", 1, 10)], k=2)
    assert first["predictions"] == second["predictions"]
    assert first["metrics"] != second["metrics"]


def test_validation_ties_select_earliest_and_restore_a_copied_checkpoint():
    # 冷用户排序始终相同，三轮验证Recall都为1，必须保留第1轮参数。
    selected = m.select_mf(history(), [Rating("new", "A", 5, 10)], config(), k=2)
    assert selected.best_epoch == 1
    first = m.fit_fixed(history(), config(), epochs=1)
    last = m.fit_fixed(history(), config(), epochs=3)
    torch.testing.assert_close(
        selected.fitted.model.user_vectors.weight, first.model.user_vectors.weight
    )
    assert not torch.equal(
        selected.fitted.model.user_vectors.weight, last.model.user_vectors.weight
    )


def test_no_validation_positive_is_not_a_fake_checkpoint_score():
    with pytest.raises(ValueError, match="positive"):
        m.select_mf(history(), [Rating("new", "A", 1, 10)], config())


def test_history_order_does_not_change_training_or_predictions():
    a = m.fit_fixed(history(), config(), epochs=2)
    b = m.fit_fixed(list(reversed(history())), config(), epochs=2)
    assert a.users == b.users
    assert a.items == b.items
    torch.testing.assert_close(a.model.user_vectors.weight, b.model.user_vectors.weight)


@pytest.mark.parametrize(
    "future",
    [
        [Rating("u", "C", 5, 2)],
        [Rating("u", "A", 5, 10)],
    ],
)
def test_interleaved_or_repeated_history_target_is_rejected(future):
    fitted = m.fit_fixed(history(), config(), epochs=1)
    with pytest.raises(ValueError):
        m.rank_mf(fitted, future)


def test_refit_builds_new_tables_only_from_its_expanded_history():
    first = m.fit_fixed(history(), config(), epochs=1)
    refit = m.fit_fixed(history() + [Rating("new", "D", 5, 10)], config(), epochs=1)
    assert "new" not in first.users and "D" not in first.items
    assert "new" in refit.users and "D" in refit.items
    assert refit.model.user_vectors.weight.shape[0] == 3


@pytest.mark.parametrize(
    "kwargs", [{"lr": float("nan")}, {"l2": -1}, {"max_epochs": 0}, {"checkpoint_every": 0}]
)
def test_invalid_config_is_rejected(kwargs):
    with pytest.raises(ValueError):
        m.MFConfig(**kwargs)
