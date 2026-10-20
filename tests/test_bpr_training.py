"""Check three-vector regularization, resampling streams and skipped epochs."""

import random

import pytest
import torch

from memrecagent import bpr_training as training
from memrecagent.mf import MatrixFactorization
from memrecagent.sampling import BPRSampler


def records():
    return [
        ("u", "A", 5.0),
        ("u", "B", 4.0),
        ("v", "C", 5.0),
        ("full", "A", 5.0),
        ("full", "B", 1.0),
        ("full", "C", 4.0),
        ("low", "B", 1.0),
    ]


def test_bpr_objective_regularizes_user_once_and_both_item_vectors():
    model = MatrixFactorization(1, 2, factors=2, seed=42)
    with torch.no_grad():
        model.user_vectors.weight.copy_(torch.tensor([[1.0, 2.0]]))
        model.item_vectors.weight.copy_(torch.tensor([[2.0, 1.0], [1.0, 1.0]]))
    loss = training.bpr_objective(
        model, torch.tensor([0]), torch.tensor([0]), torch.tensor([1]), l2=0.01
    )
    assert loss.item() == pytest.approx(0.433261688)
    loss.backward()
    torch.testing.assert_close(
        model.user_vectors.weight.grad[0], torch.tensor([-0.248941421, 0.04])
    )
    torch.testing.assert_close(
        model.item_vectors.weight.grad[0], torch.tensor([-0.228941421, -0.517882843])
    )
    torch.testing.assert_close(
        model.item_vectors.weight.grad[1], torch.tensor([0.288941421, 0.557882843])
    )


def test_training_keeps_partial_batches_counts_skips_and_improves_fixed_probe():
    result = training.fit_bpr(records(), factors=4, epochs=80, batch_size=2, lr=0.1, seed=42)
    assert len(result.history) == 81
    assert result.history[0].epoch == 0
    assert result.history[0].sampling is None
    for row in result.history[1:]:
        assert row.batch_sizes == (2, 1)
        assert row.sampling.n_positive_pairs == 5
        assert len(row.sampling.triples) == 3
        assert row.sampling.n_skipped_pairs_no_candidates == 2
        assert row.sampling.n_positive_users_without_candidates == 1
        assert row.sampling.n_users_without_positives == 1
    assert result.history[-1].check_ranking_loss < result.history[0].check_ranking_loss * 0.5
    assert result.history[-1].check_mean_gap > result.history[0].check_mean_gap


def test_fixed_probe_does_not_consume_the_training_sampling_rng():
    result = training.fit_bpr(records(), epochs=2, seed=42)
    sampler = BPRSampler(records())
    rng = random.Random(42)
    assert result.history[1].sampling == sampler.sample_epoch(rng)
    assert result.history[2].sampling == sampler.sample_epoch(rng)


def test_no_training_triples_preserves_parameters_and_reports_unavailable_loss():
    data = [("u", "A", 5.0), ("u", "B", 1.0)]
    result = training.fit_bpr(data, factors=2, epochs=2, seed=42)
    initial = MatrixFactorization(1, 2, factors=2, seed=42)
    assert result.reference_triples == []
    for row in result.history[1:]:
        assert row.sampling.n_skipped_pairs_no_candidates == 1
        assert row.batch_sizes == ()
        assert row.mean_training_loss is None
        assert row.check_ranking_loss is None
        assert row.check_mean_gap is None
    torch.testing.assert_close(result.model.user_vectors.weight, initial.user_vectors.weight)
    torch.testing.assert_close(result.model.item_vectors.weight, initial.item_vectors.weight)


def test_users_without_usable_positive_pairs_do_not_receive_fake_updates():
    result = training.fit_bpr(records(), factors=4, epochs=3, seed=42)
    initial = MatrixFactorization(4, 3, factors=4, seed=42)
    for user in ["full", "low"]:
        row = result.users[user]
        torch.testing.assert_close(
            result.model.user_vectors.weight[row], initial.user_vectors.weight[row]
        )


def test_no_positive_history_is_distinct_from_no_negative_candidates():
    result = training.fit_bpr([("u", "A", 1.0), ("v", "B", 2.0)], epochs=1)
    row = result.history[1]
    assert row.sampling.n_positive_pairs == 0
    assert row.sampling.n_skipped_pairs_no_candidates == 0
    assert row.sampling.n_users_without_positives == 2
    assert row.mean_training_loss is None


def test_training_is_reproducible_and_does_not_change_global_rngs():
    py_state, torch_state = random.getstate(), torch.random.get_rng_state().clone()
    first = training.fit_bpr(records(), epochs=3, seed=42)
    second = training.fit_bpr(list(reversed(records())), epochs=3, seed=42)
    assert first.users == second.users and first.items == second.items
    assert first.history == second.history
    assert first.reference_triples == second.reference_triples
    torch.testing.assert_close(first.model.item_vectors.weight, second.model.item_vectors.weight)
    assert random.getstate() == py_state
    torch.testing.assert_close(torch.random.get_rng_state(), torch_state)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"epochs": 0},
        {"batch_size": 0},
        {"lr": 0},
        {"l2": -1},
        {"lr": float("nan")},
        {"l2": float("nan")},
        {"threshold": 6},
    ],
)
def test_invalid_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError):
        training.fit_bpr(records(), **kwargs)


def test_empty_training_history_is_rejected():
    with pytest.raises(ValueError):
        training.fit_bpr([])
