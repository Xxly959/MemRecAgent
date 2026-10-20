"""History-only MF fitting, validation epoch selection, and catalog ranking."""

from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from math import isfinite

import torch

from memrecagent.metrics import evaluate_users
from memrecagent.mf import MatrixFactorization
from memrecagent.movielens import Rating


@dataclass(frozen=True)
class MFConfig:
    factors: int = 16
    max_epochs: int = 80
    checkpoint_every: int = 10
    batch_size: int = 256
    lr: float = 0.5
    l2: float = 0.001
    seed: int = 42

    def __post_init__(self) -> None:
        if min(self.factors, self.max_epochs, self.checkpoint_every, self.batch_size) <= 0:
            raise ValueError(
                "Dimensions, epochs, checkpoint interval, and batch size must be positive"
            )
        if not isfinite(self.lr) or self.lr <= 0 or not isfinite(self.l2) or self.l2 < 0:
            raise ValueError("Finite positive learning rate and nonnegative L2 required")


@dataclass
class FittedMF:
    model: MatrixFactorization
    users: dict[str, int]
    items: dict[str, int]
    records: tuple[Rating, ...]
    seen: dict[str, set[str]]
    trace: list[dict] = field(default_factory=list)


@dataclass
class MFSelection:
    fitted: FittedMF
    best_epoch: int
    validation: dict
    checkpoints: list[dict]


def _prepare(history: list[Rating], config: MFConfig) -> FittedMF:
    if not history:
        raise ValueError("Fitting history must be nonempty")
    records = tuple(sorted(history, key=lambda r: (r.user_id, r.item_id)))
    users = {user: index for index, user in enumerate(sorted({r.user_id for r in records}))}
    items = {item: index for index, item in enumerate(sorted({r.item_id for r in records}))}
    seen: dict[str, set[str]] = {user: set() for user in users}
    for r in records:
        if not 1 <= r.rating <= 5:
            raise ValueError("Observed ratings must be in [1,5]")
        if r.item_id in seen[r.user_id]:
            raise ValueError("Duplicate fitting user-item pair")
        seen[r.user_id].add(r.item_id)
    return FittedMF(
        MatrixFactorization(len(users), len(items), config.factors, config.seed),
        users,
        items,
        records,
        seen,
    )


def _check_holdout(fitted: FittedMF, holdout: list[Rating]) -> None:
    if holdout and max(r.timestamp for r in fitted.records) >= min(r.timestamp for r in holdout):
        raise ValueError("All fitting history must precede the holdout")
    pairs = {(r.user_id, r.item_id) for r in fitted.records}
    for r in holdout:
        if (r.user_id, r.item_id) in pairs:
            raise ValueError("Duplicate history/holdout user-item pair")
        pairs.add((r.user_id, r.item_id))


@torch.no_grad()
def rank_mf(fitted: FittedMF, holdout: list[Rating], k: int = 10, threshold: float = 4) -> dict:
    """Rank the full unseen history catalog; cold users get uniform zero scores."""
    if k <= 0 or not 1 <= threshold <= 5:
        raise ValueError("Positive k and threshold in [1,5] required")
    _check_holdout(fitted, holdout)
    fitted.model.eval()
    cohort = sorted({r.user_id for r in holdout})
    relevant: dict[str, set[str]] = {user: set() for user in cohort}
    for r in holdout:
        if r.rating >= threshold:
            relevant[r.user_id].add(r.item_id)
    catalog = list(fitted.items)
    item_ids = torch.arange(len(catalog))
    predictions = {}
    for user in cohort:
        if user not in fitted.users:
            scores = [0.0] * len(catalog)
        else:
            user_ids = torch.full((len(catalog),), fitted.users[user], dtype=torch.long)
            values = fitted.model(user_ids, item_ids)
            if not torch.isfinite(values).all():
                raise FloatingPointError("Nonfinite MF ranking scores")
            scores = values.tolist()
        candidates = [
            (item, score)
            for item, score in zip(catalog, scores, strict=True)
            if item not in fitted.seen.get(user, set())
        ]
        predictions[user] = [
            item for item, _ in sorted(candidates, key=lambda x: (-x[1], x[0]))[:k]
        ]
    return {
        "predictions": predictions,
        "relevant": {user: sorted(answers) for user, answers in relevant.items()},
        "metrics": asdict(evaluate_users(predictions, relevant, k)),
    }


def _train(
    fitted: FittedMF,
    config: MFConfig,
    epochs: int,
    on_checkpoint: Callable[[int, FittedMF, float], None] | None = None,
) -> None:
    if epochs <= 0:
        raise ValueError("Training epochs must be positive")
    users = torch.tensor([fitted.users[r.user_id] for r in fitted.records])
    items = torch.tensor([fitted.items[r.item_id] for r in fitted.records])
    ratings = torch.tensor([r.rating for r in fitted.records], dtype=torch.float32)
    generator = torch.Generator().manual_seed(config.seed)
    optimizer = torch.optim.SGD(fitted.model.parameters(), lr=config.lr)
    for epoch in range(1, epochs + 1):
        fitted.model.train()
        order = torch.randperm(len(ratings), generator=generator)
        for start in range(0, len(order), config.batch_size):
            indices = order[start : start + config.batch_size]
            optimizer.zero_grad()
            loss = fitted.model.rating_loss(
                users[indices], items[indices], ratings[indices], config.l2
            )
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite MF training loss")
            loss.backward()
            optimizer.step()
        if epoch % config.checkpoint_every == 0 or epoch == epochs:
            fitted.model.eval()
            with torch.no_grad():
                mse = (fitted.model(users, items) - ratings).square().mean().item()
            if not isfinite(mse):
                raise FloatingPointError("Nonfinite MF training MSE")
            fitted.trace.append({"epoch": epoch, "train_mse": mse})
            if on_checkpoint is not None:
                on_checkpoint(epoch, fitted, mse)


def fit_fixed(history: list[Rating], config: MFConfig, epochs: int) -> FittedMF:
    """Refit from scratch for a previously chosen epoch count; no future labels."""
    fitted = _prepare(history, config)
    _train(fitted, config, epochs)
    return fitted


def select_mf(
    history: list[Rating],
    validation: list[Rating],
    config: MFConfig,
    k: int = 10,
    threshold: float = 4,
) -> MFSelection:
    """Select highest validation Recall@K, earliest checkpoint breaks ties."""
    fitted = _prepare(history, config)
    initial = rank_mf(fitted, validation, k, threshold)
    if initial["metrics"]["mean"] is None:
        raise ValueError("Validation needs at least one user with a positive target")
    best_epoch, best_score = 0, float("-inf")
    best_state, best_result = None, None
    checkpoints = []

    def observe(epoch: int, current: FittedMF, mse: float) -> None:
        nonlocal best_epoch, best_score, best_state, best_result
        result = rank_mf(current, validation, k, threshold)
        score = result["metrics"]["mean"]["recall"]
        checkpoints.append(
            {"epoch": epoch, "train_mse": mse, "validation_metrics": result["metrics"]}
        )
        if score > best_score:
            best_epoch, best_score, best_result = epoch, score, result
            # state_dict本身引用可变参数；必须克隆，否则后续训练会改掉最佳权重。
            best_state = {
                name: value.detach().clone() for name, value in current.model.state_dict().items()
            }

    _train(fitted, config, config.max_epochs, observe)
    fitted.model.load_state_dict(best_state)
    fitted.model.eval()
    return MFSelection(fitted, best_epoch, best_result, checkpoints)
