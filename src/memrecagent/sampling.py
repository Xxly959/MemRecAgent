"""Prepare BPR comparison triples from already-cut training ratings only."""

import random
from dataclasses import dataclass


@dataclass(frozen=True)
class BPRSampleBatch:
    """One attempted triple per positive pair, with explicit exclusion counts."""

    triples: list[tuple[str, str, str]]
    n_positive_pairs: int
    n_skipped_pairs_no_candidates: int
    n_positive_users_without_candidates: int
    n_users_without_positives: int


class BPRSampler:
    """Uniform sampling from each user's sorted, unobserved history catalog.

    Caller supplies only legal training ratings, with one row per user-item.
    There are no timestamps or future-label inputs here, so this class does
    not perform or certify a time split. Catalog includes low-rated items;
    each user's own low ratings are excluded from that user's negative pool.
    """

    def __init__(
        self,
        train_records: list[tuple[str, str, float]],
        threshold: float = 4.0,
        *,
        materialize_pools: bool = True,
    ):
        if not 1 <= threshold <= 5:
            raise ValueError("threshold must be in [1,5]")
        if type(materialize_pools) is not bool:
            raise ValueError("materialize_pools必须是布尔值")
        observed: dict[str, set[str]] = {}
        positive_pairs = []
        for user, item, rating in train_records:
            if not 1 <= rating <= 5:
                raise ValueError("rating must be in [1,5]")
            seen = observed.setdefault(user, set())
            if item in seen:
                raise ValueError("Duplicate user-item interaction")
            seen.add(item)
            if rating >= threshold:
                positive_pairs.append((user, item))

        self.catalog = frozenset(item for seen in observed.values() for item in seen)
        self.observed = {user: frozenset(observed[user]) for user in sorted(observed)}
        self.positive_pairs = tuple(sorted(positive_pairs))
        self.catalog_items = tuple(sorted(self.catalog))
        self.materialize_pools = materialize_pools
        # 默认保持旧MovieLens的池和随机序列；商品任务显式关闭U×I预存。
        self.negative_pools = (
            {user: tuple(sorted(self.catalog - seen)) for user, seen in self.observed.items()}
            if materialize_pools
            else None
        )

    def has_candidates(self, user: str) -> bool:
        """已见均来自本目录，数量比较即可判断空池，无需展开差集。"""
        return len(self.observed[user]) < len(self.catalog)

    def _sample_unobserved(self, user: str, rng: random.Random) -> str:
        if self.negative_pools is not None:
            return rng.choice(self.negative_pools[user])
        seen = self.observed[user]
        # 拒绝已见项后的接受分布仍在合法集合上均匀；回退也使用均匀合法池。
        for _ in range(16):
            item = rng.choice(self.catalog_items)
            if item not in seen:
                return item
        legal = tuple(item for item in self.catalog_items if item not in seen)
        return rng.choice(legal)

    def sample_epoch(self, rng: random.Random) -> BPRSampleBatch:
        """Use the caller's continuing RNG stream; duplicates across pairs allowed."""
        triples = []
        skipped_pairs = 0
        skipped_users = set()
        for user, positive in self.positive_pairs:
            if not self.has_candidates(user):
                skipped_pairs += 1
                skipped_users.add(user)
                continue
            triples.append((user, positive, self._sample_unobserved(user, rng)))
        positive_users = {user for user, _ in self.positive_pairs}
        return BPRSampleBatch(
            triples=triples,
            n_positive_pairs=len(self.positive_pairs),
            n_skipped_pairs_no_candidates=skipped_pairs,
            n_positive_users_without_candidates=len(skipped_users),
            n_users_without_positives=len(self.observed) - len(positive_users),
        )
