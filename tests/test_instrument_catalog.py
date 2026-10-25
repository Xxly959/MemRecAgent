"""从已核验目录构造精确ID实体，未知类目不冒充乐器。"""

import pytest
from test_product_service import service

from memrecagent.entity_aliases import EntityAliases
from memrecagent.instrument_catalog import build_instrument_mapping, build_instrument_title_aliases
from memrecagent.memory_products import decode_product_entity_fact
from memrecagent.product_bundle import save_bundle


def test_mapping_uses_native_ids_and_bundle_availability_not_brand_guesses(tmp_path):
    base = service()
    roots = ["Musical Instruments", "Instrument Accessories", "Unknown"]
    for item, root in zip(sorted(base.counts), roots, strict=True):
        base.products.setdefault(item, {})["categories"] = [root]
        base.products[item]["title"] = "Samsung 看起来像品牌但不是可信品牌字段"
    bundle = tmp_path / "bundle"
    manifest = save_bundle(base, bundle, provenance={"kind": "synthetic"})
    mapping = build_instrument_mapping(bundle)
    assert set(mapping) == {"bundle_id", "facts"}
    facts = [decode_product_entity_fact(row) for row in mapping["facts"]]
    assert [f.entity for f in facts] == sorted(base.counts)
    assert [f.category for f in facts] == [
        "musical_instruments",
        "musical_instruments",
        "catalog_unknown",
    ]
    assert all(f.available_at.isoformat() == manifest["created_at_utc"] for f in facts)
    assert all(f.attributes == () and manifest["bundle_id"] in f.evidence_ref for f in facts)


def test_corrupt_bundle_cannot_generate_trusted_mapping(tmp_path):
    bundle = tmp_path / "bundle"
    save_bundle(service(), bundle, provenance={"kind": "synthetic"})
    with (bundle / "service.json").open("a") as stream:
        stream.write("corrupt")
    with pytest.raises(ValueError, match="SHA256"):
        build_instrument_mapping(bundle)


def catalog(tmp_path, titles):
    base = service()
    for item, title in titles.items():
        base.products[item] = {"title": title, "categories": ["Musical Instruments"]}
    path = tmp_path / "bundle"
    save_bundle(base, path, provenance={"kind": "synthetic"})
    return path


def test_only_unique_complete_titles_become_native_id_aliases(tmp_path):
    path = catalog(
        tmp_path, {"A": "Studio Guitar", "B": "ＳＴＵＤＩＯ ＧＵＩＴＡＲ", "C": "Concert Harp"}
    )
    document, report = build_instrument_title_aliases(path)
    table = EntityAliases.from_document(document)
    assert table.canonical("musical_instruments", "concert harp") == "C"
    assert table.canonical("musical_instruments", "Concert") == "Concert"
    assert table.canonical("musical_instruments", "Studio Guitar") == "Studio Guitar"
    assert table.canonical("phone", "Concert Harp") == "Concert Harp"
    assert report["registered_titles"] == 1
    assert report["skipped"]["ambiguous_title"] == 2
    assert report["registered_titles"] + sum(report["skipped"].values()) == 3
    assert table.source_for("musical_instruments", "Concert Harp").startswith("bundle:")


def test_title_cannot_steal_another_native_product_id(tmp_path):
    path = catalog(tmp_path, {"A": "b", "B": "Basic Drums", "C": "Unique Harp"})
    document, report = build_instrument_title_aliases(path)
    table = EntityAliases.from_document(document)
    assert table.canonical("musical_instruments", "B") == "B"
    assert table.canonical("musical_instruments", "Basic Drums") == "B"
    assert report["skipped"]["canonical_collision"] == 1


def test_long_missing_and_unknown_domain_titles_are_not_truncated_or_registered(tmp_path):
    base = service()
    base.products = {
        "A": {"title": "x" * 257, "categories": ["Musical Instruments"]},
        "B": {"title": "Harp " + "x" * 130, "categories": ["Musical Instruments"]},
        "C": {"title": "Unknown instrument", "categories": []},
    }
    path = tmp_path / "bundle"
    save_bundle(base, path, provenance={"kind": "synthetic"})
    document, report = build_instrument_title_aliases(path)
    assert [row["canonical"] for row in document["entries"]] == ["B"]
    assert document["entries"][0]["aliases"] == [base.products["B"]["title"]]
    assert report["skipped"]["title_length"] == 1
    assert report["skipped"]["outside_domain"] == 1


def test_registry_supports_catalog_size_and_rejects_unbounded_growth():
    document = dict(
        schema_version=1,
        entries=[
            dict(
                category="musical_instruments",
                canonical=f"ID{i}",
                aliases=[f"Instrument {i}"],
                source="synthetic:catalog",
            )
            for i in range(5000)
        ],
    )
    table = EntityAliases.from_document(document)
    assert table.canonical("musical_instruments", "Instrument 4999") == "ID4999"
    document["entries"].append(
        dict(
            category="musical_instruments",
            canonical="overflow",
            aliases=[],
            source="synthetic:catalog",
        )
    )
    with pytest.raises(ValueError):
        EntityAliases.from_document(document)


def test_title_export_cli_creates_reviewable_files_without_installing_or_overwriting(tmp_path):
    import json
    import subprocess
    import sys
    from pathlib import Path

    path = catalog(tmp_path, {"A": "Studio Guitar", "B": "Basic Drums", "C": "Concert Harp"})
    output = tmp_path / "export"
    command = [
        sys.executable,
        str(Path(__file__).parents[1] / "scripts/create_instrument_aliases.py"),
        "--bundle",
        str(path),
        "--output-dir",
        str(output),
    ]
    first = subprocess.run(command, capture_output=True, text=True)
    assert first.returncode == 0, first.stderr
    document = json.loads((output / "entity-aliases.json").read_text())
    assert (
        EntityAliases.from_document(document).canonical("musical_instruments", "Studio Guitar")
        == "A"
    )
    assert json.loads((output / "report.json").read_text())["registered_titles"] == 3
    old = (output / "entity-aliases.json").read_bytes()
    second = subprocess.run(command, capture_output=True, text=True)
    assert second.returncode != 0
    assert (output / "entity-aliases.json").read_bytes() == old
    assert not (tmp_path / "memory").exists()
