"""Explicit-rating matrix factorization for small CPU teaching experiments."""

from dataclasses import dataclass
from math import isfinite

import torch
from torch import nn

RatingRecord = tuple[int, int, float]


class MatrixFactorization(nn.Module):
    """Learn user/item vector tables and predict one score for each input pair."""

    def __init__(self, n_users: int, n_items: int, factors: int = 8, seed: int = 42):
        super().__init__()
        if min(n_users, n_items, factors) <= 0:
            raise ValueError("User count, item count, and factors must be positive")
        # 局部生成器固定初始化，不改动进程其他代码使用的随机数状态。
        generator = torch.Generator().manual_seed(seed)
        self.user_vectors = nn.Embedding.from_pretrained(
            torch.randn(n_users, factors, generator=generator) * 0.1, freeze=False
        )
        self.item_vectors = nn.Embedding.from_pretrained(
            torch.randn(n_items, factors, generator=generator) * 0.1, freeze=False
        )

    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        """Return shape (batch,), rather than all user-item cross combinations."""
        if (
            user_ids.ndim != 1
            or item_ids.ndim != 1
            or user_ids.shape != item_ids.shape
            or user_ids.numel() == 0
        ):
            raise ValueError("User/item IDs must be matching nonempty one-dimensional batches")
        p = self.user_vectors(user_ids)
        q = self.item_vectors(item_ids)
        return (p * q).sum(dim=-1)

    def rating_loss(
        self,
        user_ids: torch.Tensor,
        item_ids: torch.Tensor,
        ratings: torch.Tensor,
        l2: float = 0.001,
    ) -> torch.Tensor:
        """MSE plus mean per-interaction squared vector norms.

        Repeated IDs contribute once per rating, including to regularization;
        this is frequency-weighted regularization, not a whole-table penalty.
        No optimizer weight_decay is added on top.
        """
        predictions = self(user_ids, item_ids)
        if ratings.shape != predictions.shape:
            raise ValueError("ratings must contain exactly one value per prediction")
        mse = (predictions - ratings).square().mean()
        norms = self.user_vectors(user_ids).square().sum(dim=-1)
        norms = norms + self.item_vectors(item_ids).square().sum(dim=-1)
        return mse + l2 * norms.mean()


@dataclass(frozen=True)
class EpochMetrics:
    """Post-epoch plain MSE at fixed weights; neither metric includes L2."""

    epoch: int
    train_mse: float
    validation_mse: float
    batch_sizes: tuple[int, ...]


def _rating_tensors(records: list[RatingRecord]) -> tuple[torch.Tensor, ...]:
    return (
        torch.tensor([u for u, _, _ in records], dtype=torch.long),
        torch.tensor([i for _, i, _ in records], dtype=torch.long),
        torch.tensor([r for _, _, r in records], dtype=torch.float32),
    )


@torch.no_grad()
def _mse(model: MatrixFactorization, tensors: tuple[torch.Tensor, ...]) -> float:
    model.eval()
    users, items, ratings = tensors
    return (model(users, items) - ratings).square().mean().item()


def fit_mf(
    model: MatrixFactorization,
    train_records: list[RatingRecord],
    validation_records: list[RatingRecord],
    *,
    epochs: int = 100,
    batch_size: int = 5,
    lr: float = 0.05,
    l2: float = 0.001,
    seed: int = 42,
) -> list[EpochMetrics]:
    """Run a fixed number of shuffled SGD epochs on observed CPU ratings only.

    Validation records never take part in backward, checkpoint selection, or
    parameter updates. Input IDs are pre-mapped and all lie inside the tables;
    temporal splitting and any cold-start protocol belong to the caller.
    """
    if epochs <= 0 or batch_size <= 0 or not isfinite(lr) or lr <= 0:
        raise ValueError("epochs, batch_size, and finite learning rate must be positive")
    if not isfinite(l2) or l2 < 0 or not train_records or not validation_records:
        raise ValueError("Nonempty train/validation and finite nonnegative l2 required")
    if model.user_vectors.weight.device.type != "cpu":
        raise ValueError("This teaching trainer supports CPU only")
    pairs = set()
    for u, i, rating in [*train_records, *validation_records]:
        if not (
            0 <= u < model.user_vectors.num_embeddings
            and 0 <= i < model.item_vectors.num_embeddings
            and 1 <= rating <= 5
        ):
            raise ValueError("IDs must be in table range and ratings in [1,5]")
        if (u, i) in pairs:
            raise ValueError("Duplicate or overlapping train/validation user-item pair")
        pairs.add((u, i))

    train = _rating_tensors(train_records)
    validation = _rating_tensors(validation_records)
    generator = torch.Generator().manual_seed(seed)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    history = [EpochMetrics(0, _mse(model, train), _mse(model, validation), ())]
    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(len(train_records), generator=generator)
        sizes = []
        for start in range(0, len(order), batch_size):
            indices = order[start : start + batch_size]
            users, items, ratings = (values[indices] for values in train)
            optimizer.zero_grad()
            loss = model.rating_loss(users, items, ratings, l2=l2)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite training loss; inspect training configuration")
            loss.backward()
            optimizer.step()
            sizes.append(len(indices))
        history.append(
            EpochMetrics(epoch, _mse(model, train), _mse(model, validation), tuple(sizes))
        )
    return history
