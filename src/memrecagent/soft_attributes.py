"""有限谓词的软属性信号；未知中性，同一属性不重复投票。"""

from datetime import datetime

from memrecagent.product_attributes import (
    AttributePredicate,
    ProductAttributeValue,
    constraint_failure,
)


def soft_attribute_signal(
    attributes: tuple[ProductAttributeValue, ...],
    preferences: tuple[AttributePredicate, ...],
    *,
    as_of: datetime,
) -> tuple[float, list[dict]]:
    by_attribute: dict[str, list[int]] = {}
    for index, preference in enumerate(preferences):
        by_attribute.setdefault(preference.attribute, []).append(index)
    signals = []
    for attribute, indices in sorted(by_attribute.items()):
        failure = constraint_failure(
            attributes, tuple(preferences[i] for i in indices), as_of=as_of
        )
        status = (
            "match"
            if failure is None
            else "unknown"
            if failure[0] == "attribute_unknown"
            else "mismatch"
        )
        fact = next(
            (a for a in attributes if a.attribute == attribute and a.available_at < as_of), None
        )
        signals.append(
            dict(
                attribute=attribute,
                status=status,
                value={"match": 1, "mismatch": -1, "unknown": 0}[status],
                preference_indices=indices,
                fact_source=fact.evidence_ref if fact is not None else None,
            )
        )
    return (sum(s["value"] for s in signals) / len(signals) if signals else 0.0), signals
