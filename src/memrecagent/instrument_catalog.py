"""复用已核验乐器推理包；精确ID是商品实体，不从标题推测品牌。"""

import json
from collections import Counter
from datetime import datetime
from pathlib import Path

from memrecagent.entity_aliases import EntityAliases, _key
from memrecagent.memory_products import ProductEntityFact
from memrecagent.product_bundle import load_bundle

INSTRUMENT_CATEGORY = "musical_instruments"


def _category(metadata: dict) -> str:
    categories = metadata.get("categories") or []
    root = categories[0] if categories else None
    if isinstance(root, list):
        root = root[0] if root else None
    return (
        INSTRUMENT_CATEGORY
        if root in ("Musical Instruments", "Instrument Accessories")
        else "catalog_unknown"
    )


def build_instrument_mapping(bundle: Path) -> dict:
    """按快照根类目映射；目录可用时间取包发布时刻，不回填到训练时点。"""
    service = load_bundle(bundle)
    manifest = json.loads((bundle / "bundle.json").read_text(encoding="utf-8"))
    available = datetime.fromisoformat(manifest["created_at_utc"])
    facts = []
    for item in sorted(service.counts):
        category = _category(service.products.get(item, {}))
        fact = ProductEntityFact(
            item_id=item,
            category=category,
            entity=item,
            evidence_ref=f"bundle:{service.bundle_id}:catalog:{item}",
            available_at=available,
        )
        facts.append(
            dict(
                item_id=fact.item_id,
                category=fact.category,
                entity=fact.entity,
                evidence_ref=fact.evidence_ref,
                available_at=fact.available_at.isoformat(),
                attributes=[],
            )
        )
    return {"bundle_id": service.bundle_id, "facts": facts}


def build_instrument_title_aliases(bundle: Path) -> tuple[dict, dict]:
    """只登记全目录中唯一的完整标题，不截短、猜系列或吞掉ID冲突。"""
    service = load_bundle(bundle)
    manifest = json.loads((bundle / "bundle.json").read_text(encoding="utf-8"))
    titles = {item: service.products.get(item, {}).get("title") for item in service.counts}
    title_counts = Counter(_key(t) for t in titles.values() if isinstance(t, str) and t.strip())
    id_owners: dict[str, set[str]] = {}
    for item in service.counts:
        id_owners.setdefault(_key(item), set()).add(item)
    skipped: Counter[str] = Counter()
    entries = []
    for item in sorted(titles):
        title = titles[item]
        if _category(service.products.get(item, {})) != INSTRUMENT_CATEGORY:
            skipped["outside_domain"] += 1
            continue
        if not isinstance(title, str) or not title.strip():
            skipped["missing_title"] += 1
            continue
        if len(title) > 256:
            skipped["title_length"] += 1
            continue
        try:
            title.encode("utf-8")
        except UnicodeError:
            skipped["invalid_encoding"] += 1
            continue
        if title_counts[_key(title)] != 1:
            skipped["ambiguous_title"] += 1
            continue
        if id_owners.get(_key(title), set()) - {item}:
            skipped["canonical_collision"] += 1
            continue
        entries.append(
            dict(
                category=INSTRUMENT_CATEGORY,
                canonical=item,
                aliases=[title],
                source=f"bundle:{service.bundle_id}:catalog:{item}:title",
            )
        )
    table = EntityAliases.from_document(dict(schema_version=1, entries=entries))
    report = dict(
        bundle_id=service.bundle_id,
        usable_after=manifest["created_at_utc"],
        catalog_items=len(titles),
        registered_titles=len(entries),
        skipped=dict(sorted(skipped.items())),
        registry_digest=table.digest,
        matching="whole_title_nfkc_casefold_strip",
        api_calls=0,
    )
    return table.to_document(), report
