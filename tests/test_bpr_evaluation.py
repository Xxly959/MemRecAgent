"""Protect BPR sampling causality, inactive users, and checkpoint ownership."""

import random

import pytest
import torch

from memrecagent import bpr_evaluation as b
from memrecagent.movielens import Rating


def history():
    rows = [
        ("u", "A", 5),
        ("u", "B", 1),
        ("v", "C", 4),
        ("v", "D", 2),
        ("v", "E", 5),
        ("full", "A", 5),
        ("full", "B", 1),
        ("full", "C", 4),
        ("full", "D", 2),
        ("full", "E", 5),
        ("low", "B", 1),
        ("low", "E", 2),
    ]
    return [Rating(u, i, r, t) for t, (u, i, r) in enumerate(rows, 1)]


def config(**kwargs):
    return b.BPRConfig(factors=2, max_epochs=3, checkpoint_every=1, batch_size=2, **kwargs)


def test_training_counts_skips_and_retains_partial_batches():
    fitted = b.fit_bpr_fixed(history(), config(), epochs=2)
    assert fitted.active_users == frozenset({"u", "v"})
    for row in fitted.trace:
        assert row["n_positive_pairs"] == 6
        assert row["n_triples"] == 3
        assert row["n_skipped_pairs_no_candidates"] == 3
        assert row["n_positive_users_without_candidates"] == 1
        assert row["n_users_without_positives"] == 1
        assert row["n_batches"] == 2
        assert row["last_batch_size"] == 1
        assert row["mean_training_loss"] > 0
    assert fitted.trace[0]["triples_sha256"] != fitted.trace[1]["triples_sha256"]


def test_ranking_filters_all_observed_and_keeps_negative_scores():
    fitted = b.fit_bpr_fixed(history(), config(), epochs=1)
    with torch.no_grad():
        fitted.model.user_vectors.weight[fitted.users["u"]].copy_(torch.tensor([1.0, 0.0]))
        for item, score in [("A", 99), ("B", 98), ("C", -2), ("D", -1), ("E", -3)]:
            fitted.model.item_vectors.weight[fitted.items[item]].copy_(torch.tensor([score, 0.0]))
    result = b.rank_bpr(fitted, [Rating("u", "D", 5, 20)], k=3)
    assert result["predictions"]["u"] == ["D", "C", "E"]
    assert result["metrics"]["mean"]["recall"] == 1


def test_inactive_and_cold_users_have_zero_scores_without_future_vocab():
    fitted = b.fit_bpr_fixed(history(), config(), epochs=1)
    with torch.no_grad():
        fitted.model.user_vectors.weight.fill_(100)
        fitted.model.item_vectors.weight[fitted.items["D"]].fill_(100)
    result = b.rank_bpr(fitted, [Rating("low", "Z", 5, 20), Rating("new", "Z", 5, 21)], k=3)
    assert result["predictions"] == {"low": ["A", "C", "D"], "new": ["A", "B", "C"]}
    assert "new" not in fitted.users and "Z" not in fitted.items
    assert result["metrics"]["n_evaluated"] == 2
    assert result["metrics"]["mean"]["recall"] == 0


def test_future_labels_affect_metrics_but_not_ranking_or_parameters():
    fitted = b.fit_bpr_fixed(history(), config(), epochs=1)
    before = {k: v.clone() for k, v in fitted.model.state_dict().items()}
    yes = b.rank_bpr(fitted, [Rating("u", "C", 5, 20)], k=3)
    no = b.rank_bpr(fitted, [Rating("u", "C", 1, 20)], k=3)
    assert yes["predictions"] == no["predictions"]
    assert yes["metrics"] != no["metrics"]
    for key, value in before.items():
        assert torch.equal(value, fitted.model.state_dict()[key])


def test_ties_restore_earliest_checkpoint_not_last_mutated_weights():
    selected = b.select_bpr(history(), [Rating("new", "A", 5, 20)], config(), k=2)
    assert selected.best_epoch == 1
    first = b.fit_bpr_fixed(history(), config(), epochs=1)
    last = b.fit_bpr_fixed(history(), config(), epochs=3)
    for key, value in first.model.state_dict().items():
        assert torch.equal(selected.fitted.model.state_dict()[key], value)
    assert not torch.equal(first.model.item_vectors.weight, last.model.item_vectors.weight)
    assert selected.validation == b.rank_bpr(selected.fitted, [Rating("new", "A", 5, 20)], k=2)


def test_final_noninterval_epoch_is_a_checkpoint():
    cfg = b.BPRConfig(factors=2, max_epochs=3, checkpoint_every=2, batch_size=2)
    selected = b.select_bpr(history(), [Rating("new", "A", 5, 20)], cfg)
    assert [r["epoch"] for r in selected.checkpoints] == [2, 3]


def test_no_validation_positive_cannot_choose_a_fake_best_epoch():
    with pytest.raises(ValueError, match="positive"):
        b.select_bpr(history(), [Rating("new", "A", 1, 20)], config())


@pytest.mark.parametrize(
    "holdout",
    [
        [Rating("u", "C", 5, 10)],
        [Rating("u", "A", 5, 20)],
        [Rating("new", "A", 5, 20), Rating("new", "A", 4, 21)],
    ],
)
def test_invalid_time_or_duplicate_target_is_rejected(holdout):
    with pytest.raises(ValueError):
        b.select_bpr(history(), holdout, config())


def test_order_and_global_random_state_do_not_affect_reproducibility():
    python_state, torch_state = random.getstate(), torch.random.get_rng_state().clone()
    first = b.fit_bpr_fixed(history(), config(), epochs=3)
    second = b.fit_bpr_fixed(list(reversed(history())), config(), epochs=3)
    assert random.getstate() == python_state
    assert torch.equal(torch_state, torch.random.get_rng_state())
    assert first.trace == second.trace
    for key, value in first.model.state_dict().items():
        assert torch.equal(value, second.model.state_dict()[key])


def test_refit_can_expand_history_without_using_final_test_vocab():
    first = b.fit_bpr_fixed(history(), config(), epochs=1)
    refit = b.fit_bpr_fixed(history() + [Rating("new", "F", 5, 20)], config(), epochs=1)
    assert "new" not in first.users and "F" not in first.items
    assert "new" in refit.active_users and "F" in refit.items
    assert "Z" not in refit.items


def test_no_legal_triples_retains_parameters_and_reports_none_loss():
    records = [Rating("u", "A", 5, 1), Rating("u", "B", 1, 2)]
    first = b.fit_bpr_fixed(records, config(), epochs=1)
    last = b.fit_bpr_fixed(records, config(), epochs=3)
    assert not first.active_users
    for row in last.trace:
        assert row["mean_training_loss"] is None
        assert row["n_batches"] == 0
        assert row["n_skipped_pairs_no_candidates"] == 1
    assert torch.equal(first.model.user_vectors.weight, last.model.user_vectors.weight)


@pytest.mark.parametrize(
    "kwargs", [{"lr": float("nan")}, {"l2": -1}, {"max_epochs": 0}, {"threshold": 6}]
)
def test_invalid_config_is_rejected(kwargs):
    with pytest.raises(ValueError):
        b.BPRConfig(**kwargs)
