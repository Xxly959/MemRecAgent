"""History-only BPR training, compact sampling logs, and validation selection."""

import json
import random
from collections.abc import Callable
from dataclasses import dataclass, replace
from hashlib import sha256

import torch

from memrecagent.bpr_training import bpr_objective
from memrecagent.mf import MatrixFactorization
from memrecagent.mf_evaluation import FittedMF, MFConfig, rank_mf
from memrecagent.movielens import Rating
from memrecagent.sampling import BPRSampler


@dataclass(frozen=True)
class BPRConfig(MFConfig):
    """Same dimension/SGD defaults as rating MF, but a pairwise objective."""

    threshold: float = 4.0
    materialize_pools: bool = True

    def __post_init__(self) -> None:
        super().__post_init__()
        if not 1 <= self.threshold <= 5:
            raise ValueError("Positive threshold must be in [1,5]")
        if type(self.materialize_pools) is not bool:
            raise ValueError("materialize_pools必须是布尔值")


@dataclass
class FittedBPR(FittedMF):
    active_users: frozenset[str] = frozenset()


@dataclass
class BPRSelection:
    fitted: FittedBPR
    best_epoch: int
    validation: dict
    checkpoints: list[dict]


def _prepare(history: list[Rating], config: BPRConfig) -> tuple[FittedBPR, BPRSampler]:
    if not history:
        raise ValueError("Fitting history must be nonempty")
    records = tuple(sorted(history, key=lambda r: (r.user_id, r.item_id)))
    sampler = BPRSampler(
        [(r.user_id, r.item_id, r.rating) for r in records],
        config.threshold,
        materialize_pools=config.materialize_pools,
    )
    users = {u: index for index, u in enumerate(sampler.observed)}
    items = {i: index for index, i in enumerate(sorted(sampler.catalog))}
    active = frozenset(u for u, _ in sampler.positive_pairs if sampler.has_candidates(u))
    return FittedBPR(
        model=MatrixFactorization(len(users), len(items), config.factors, config.seed),
        users=users,
        items=items,
        records=records,
        seen={u: set(seen) for u, seen in sampler.observed.items()},
        active_users=active,
    ), sampler


def rank_bpr(fitted: FittedBPR, holdout: list[Rating], k: int = 10, threshold: float = 4) -> dict:
    """Reuse the full dot-product ranking protocol, masking users without triples.

    The temporary scoring view retains original embedding row numbers and
    all observed history. It neither changes saved vocabulary nor filters
    future positive items out of evaluation denominators.
    """
    scoring_view = replace(
        fitted, users={u: row for u, row in fitted.users.items() if u in fitted.active_users}
    )
    return rank_mf(scoring_view, holdout, k, threshold)


def _train(
    fitted: FittedBPR,
    sampler: BPRSampler,
    config: BPRConfig,
    epochs: int,
    on_checkpoint: Callable[[int, FittedBPR, dict], None] | None = None,
) -> None:
    if epochs <= 0:
        raise ValueError("Training epochs must be positive")
    sampling_rng = random.Random(config.seed)
    shuffle_rng = torch.Generator().manual_seed(config.seed)
    optimizer = torch.optim.SGD(fitted.model.parameters(), lr=config.lr)
    for epoch in range(1, epochs + 1):
        fitted.model.train()
        sampled = sampler.sample_epoch(sampling_rng)
        triples = sampled.triples
        tensors = (
            torch.tensor([fitted.users[u] for u, _, _ in triples], dtype=torch.long),
            torch.tensor([fitted.items[i] for _, i, _ in triples], dtype=torch.long),
            torch.tensor([fitted.items[j] for _, _, j in triples], dtype=torch.long),
        )
        order = torch.randperm(len(triples), generator=shuffle_rng)
        total, n_batches, last_size = 0.0, 0, 0
        for start in range(0, len(order), config.batch_size):
            indices = order[start : start + config.batch_size]
            u, i, j = (values[indices] for values in tensors)
            optimizer.zero_grad()
            loss = bpr_objective(fitted.model, u, i, j, config.l2)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite BPR training loss")
            loss.backward()
            optimizer.step()
            total += loss.detach().item() * len(indices)
            n_batches += 1
            last_size = len(indices)
        # Keep counts and a replayable sample fingerprint, not every epoch's triples.
        row = {
            "epoch": epoch,
            "mean_training_loss": total / len(triples) if triples else None,
            "n_positive_pairs": sampled.n_positive_pairs,
            "n_triples": len(triples),
            "n_skipped_pairs_no_candidates": sampled.n_skipped_pairs_no_candidates,
            "n_positive_users_without_candidates": sampled.n_positive_users_without_candidates,
            "n_users_without_positives": sampled.n_users_without_positives,
            "n_batches": n_batches,
            "last_batch_size": last_size,
            "triples_sha256": sha256(json.dumps(triples).encode()).hexdigest(),
        }
        fitted.trace.append(row)
        if on_checkpoint is not None and (epoch % config.checkpoint_every == 0 or epoch == epochs):
            on_checkpoint(epoch, fitted, row)


def fit_bpr_fixed(history: list[Rating], config: BPRConfig, epochs: int) -> FittedBPR:
    """Fresh history-only fit for an epoch count already fixed on validation."""
    fitted, sampler = _prepare(history, config)
    _train(fitted, sampler, config, epochs)
    return fitted


def select_bpr(
    history: list[Rating],
    validation: list[Rating],
    config: BPRConfig,
    k: int = 10,
) -> BPRSelection:
    """Highest validation macro Recall, ties earliest; epoch0 is not a candidate."""
    fitted, sampler = _prepare(history, config)
    initial = rank_bpr(fitted, validation, k, config.threshold)
    if initial["metrics"]["mean"] is None:
        raise ValueError("Validation needs at least one user with a positive target")
    best_epoch, best_score = 0, float("-inf")
    best_state, best_result = None, None
    checkpoints = []

    def observe(epoch: int, current: FittedBPR, row: dict) -> None:
        nonlocal best_epoch, best_score, best_state, best_result
        result = rank_bpr(current, validation, k, config.threshold)
        score = result["metrics"]["mean"]["recall"]
        checkpoints.append({**row, "validation_metrics": result["metrics"]})
        if score > best_score:
            best_epoch, best_score, best_result = epoch, score, result
            best_state = {
                name: value.detach().clone() for name, value in current.model.state_dict().items()
            }

    _train(fitted, sampler, config, config.max_epochs, observe)
    fitted.model.load_state_dict(best_state)
    fitted.model.eval()
    return BPRSelection(fitted, best_epoch, best_result, checkpoints)
