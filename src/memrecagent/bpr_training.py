"""Small CPU BPR training with resampling and a fixed training diagnostic probe."""

import random
from dataclasses import dataclass
from math import isfinite

import torch

from memrecagent.bpr import bpr_loss
from memrecagent.mf import MatrixFactorization
from memrecagent.sampling import BPRSampleBatch, BPRSampler


@dataclass(frozen=True)
class BPREpoch:
    epoch: int
    sampling: BPRSampleBatch | None
    batch_sizes: tuple[int, ...]
    mean_training_loss: float | None
    check_ranking_loss: float | None
    check_mean_gap: float | None


@dataclass
class BPRTrainingResult:
    model: MatrixFactorization
    users: dict[str, int]
    items: dict[str, int]
    reference_triples: list[tuple[str, str, str]]
    history: list[BPREpoch]


def bpr_objective(
    model: MatrixFactorization,
    users: torch.Tensor,
    positives: torch.Tensor,
    negatives: torch.Tensor,
    l2: float = 0.001,
) -> torch.Tensor:
    """Mean pair loss plus per-triple norms of one user and two item vectors."""
    ranking = bpr_loss(model(users, positives), model(users, negatives))
    norms = model.user_vectors(users).square().sum(dim=-1)
    norms = norms + model.item_vectors(positives).square().sum(dim=-1)
    norms = norms + model.item_vectors(negatives).square().sum(dim=-1)
    return ranking + l2 * norms.mean()


def _tensors(
    triples: list[tuple[str, str, str]], users: dict[str, int], items: dict[str, int]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.tensor([users[u] for u, _, _ in triples], dtype=torch.long),
        torch.tensor([items[i] for _, i, _ in triples], dtype=torch.long),
        torch.tensor([items[j] for _, _, j in triples], dtype=torch.long),
    )


@torch.no_grad()
def _check(
    model: MatrixFactorization, reference: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
) -> tuple[float | None, float | None]:
    """Compare fixed training-derived pairs at fixed weights, without L2."""
    model.eval()
    users, positives, negatives = reference
    if users.numel() == 0:
        return None, None
    positive_scores, negative_scores = model(users, positives), model(users, negatives)
    loss = bpr_loss(positive_scores, negative_scores).item()
    gap = (positive_scores - negative_scores).mean().item()
    if not isfinite(loss) or not isfinite(gap):
        raise FloatingPointError("Nonfinite BPR diagnostic scores")
    return loss, gap


def fit_bpr(
    train_records: list[tuple[str, str, float]],
    *,
    factors: int = 4,
    epochs: int = 100,
    batch_size: int = 2,
    lr: float = 0.1,
    l2: float = 0.001,
    threshold: float = 4.0,
    seed: int = 42,
) -> BPRTrainingResult:
    """Train only legal history triples, retaining partial batches and skip counts.

    Each epoch samples one j per usable positive pair, then shuffles triples.
    A separate RNG builds fixed training diagnostics; it is not a validation
    set. No future observations, checkpoint selection, or ranking claims here.
    """
    if not train_records or epochs <= 0 or batch_size <= 0:
        raise ValueError("Nonempty history and positive epochs/batch size required")
    if not isfinite(lr) or lr <= 0 or not isfinite(l2) or l2 < 0:
        raise ValueError("Finite positive learning rate and nonnegative L2 required")
    sampler = BPRSampler(train_records, threshold)
    users = {u: row for row, u in enumerate(sampler.observed)}
    items = {i: column for column, i in enumerate(sorted(sampler.catalog))}
    model = MatrixFactorization(len(users), len(items), factors=factors, seed=seed)
    reference_triples = sampler.sample_epoch(random.Random(seed + 1)).triples
    reference = _tensors(reference_triples, users, items)
    initial_loss, initial_gap = _check(model, reference)
    history = [BPREpoch(0, None, (), None, initial_loss, initial_gap)]
    sampling_rng = random.Random(seed)
    shuffle_rng = torch.Generator().manual_seed(seed)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        model.train()
        sampled = sampler.sample_epoch(sampling_rng)
        tensors = _tensors(sampled.triples, users, items)
        order = torch.randperm(len(sampled.triples), generator=shuffle_rng)
        weighted_loss, sizes = 0.0, []
        for start in range(0, len(order), batch_size):
            indices = order[start : start + batch_size]
            u, i, j = (values[indices] for values in tensors)
            optimizer.zero_grad()
            loss = bpr_objective(model, u, i, j, l2=l2)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite BPR training loss")
            loss.backward()
            optimizer.step()
            weighted_loss += loss.detach().item() * len(indices)
            sizes.append(len(indices))
        check_loss, check_gap = _check(model, reference)
        history.append(
            BPREpoch(
                epoch,
                sampled,
                tuple(sizes),
                weighted_loss / len(sampled.triples) if sampled.triples else None,
                check_loss,
                check_gap,
            )
        )
    return BPRTrainingResult(model, users, items, reference_triples, history)
