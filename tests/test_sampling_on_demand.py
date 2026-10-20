"""按需采样不能物化用户×目录，且必须保持合法、均匀与可复现。"""

import random
import tracemalloc
from collections import Counter

import torch

from memrecagent.bpr_evaluation import BPRConfig, fit_bpr_fixed
from memrecagent.mf import MatrixFactorization
from memrecagent.movielens import Rating
from memrecagent.sampling import BPRSampler


def test_on_demand_sampling_replays_and_covers_legal_items_without_using_future_labels():
    records = [("u", "A", 5), ("u", "B", 1), ("v", "C", 1), ("v", "D", 1), ("v", "E", 1)]
    sampler = BPRSampler(records, materialize_pools=False)
    other = BPRSampler(list(reversed(records)), materialize_pools=False)
    first, second = random.Random(17), random.Random(17)
    frequencies = Counter()
    for _ in range(6000):
        a, b = sampler.sample_epoch(first), other.sample_epoch(second)
        assert a == b
        assert len(a.triples) == 1
        user, positive, negative = a.triples[0]
        assert (user, positive) == ("u", "A")
        assert negative in {"C", "D", "E"}
        frequencies[negative] += 1
    assert all(1800 < frequencies[item] < 2200 for item in ["C", "D", "E"])


def test_dense_history_has_a_bounded_fallback_instead_of_infinite_rejection():
    class RepeatedChoice:
        def __init__(self):
            self.calls = 0

        def choice(self, values):
            self.calls += 1
            assert self.calls < 32, "密集历史必须在有限尝试后使用合法池"
            return values[0]

    sampler = BPRSampler([("u", "A", 5), ("u", "B", 1), ("v", "C", 1)], materialize_pools=False)
    assert sampler.sample_epoch(RepeatedChoice()).triples == [("u", "A", "C")]


def test_on_demand_sampler_memory_does_not_scale_as_every_user_times_every_item():
    records = [(f"u{i}", "shared", 5) for i in range(2000)] + [
        (f"v{i}", f"item{i}", 1) for i in range(2000)
    ]
    tracemalloc.start()
    try:
        sampler = BPRSampler(records, materialize_pools=False)
        _, peak = tracemalloc.get_traced_memory()
        assert peak < 16 * 1024 * 1024
        result = sampler.sample_epoch(random.Random(42))
        assert len(result.triples) == 2000
    finally:
        tracemalloc.stop()


def test_bpr_training_uses_on_demand_mode_and_preserves_inactive_users():
    history = [
        Rating("u", "A", 5, 1),
        Rating("low", "B", 1, 2),
        Rating("full", "A", 5, 3),
        Rating("full", "B", 1, 4),
    ]
    config = BPRConfig(
        factors=2, max_epochs=2, checkpoint_every=1, batch_size=2, materialize_pools=False
    )
    first = fit_bpr_fixed(history, config, epochs=2)
    second = fit_bpr_fixed(history, config, epochs=2)
    assert first.active_users == {"u"}
    assert first.trace == second.trace
    assert first.trace[0]["n_skipped_pairs_no_candidates"] == 1
    initial = MatrixFactorization(3, 2, factors=2, seed=42)
    assert torch.equal(
        first.model.user_vectors.weight[first.users["full"]],
        initial.user_vectors.weight[first.users["full"]],
    )
