"""可信别名只匹配声明实体，冲突/换表安装/已有数据不得静默合并。"""

import json

import pytest

from memrecagent.entity_aliases import EntityAliases, install_entity_aliases, load_entity_aliases


def document():
    return dict(
        schema_version=1,
        entries=[
            dict(
                category="phone",
                canonical="品牌A",
                aliases=["BrandA", "A牌"],
                source="synthetic:catalog:v1",
            )
        ],
    )


def test_known_whole_entity_matches_normalised_spelling_without_fuzzy_guess():
    table = EntityAliases.from_document(document())
    assert table.canonical("phone", "  ＢＲＡＮＤＡ  ") == "品牌A"
    assert table.canonical("phone", "A牌") == "品牌A"
    assert table.canonical("phone", "品牌A") == "品牌A"
    assert table.canonical("laptop", "BrandA") == "BrandA"
    assert table.canonical("phone", "BrandA Series") == "BrandA Series"
    assert table.canonical("phone", "未知对象") == "未知对象"
    assert table.source_for("phone", "BrandA") == "synthetic:catalog:v1"
    assert table.source_for("phone", "unknown") is None


def test_input_and_export_mutations_cannot_change_trusted_registry():
    raw = document()
    table = EntityAliases.from_document(raw)
    digest = table.digest
    raw["entries"][0]["aliases"].append("attacker")
    exported = table.to_document()
    exported["entries"][0]["canonical"] = "evil"
    assert table.canonical("phone", "attacker") == "attacker"
    assert table.canonical("phone", "BrandA") == "品牌A" and table.digest == digest


def test_normalised_collision_between_different_entities_is_rejected():
    raw = document()
    raw["entries"].append(
        dict(category="phone", canonical="品牌B", aliases=["branda"], source="synthetic:catalog:v2")
    )
    with pytest.raises(ValueError):
        EntityAliases.from_document(raw)


@pytest.mark.parametrize(
    "change",
    [
        "missing_source",
        "unknown_field",
        "bool_version",
        "oversized_aliases",
        "blank",
        "bad_category",
    ],
)
def test_untrusted_or_incomplete_table_is_not_accepted(change):
    raw = document()
    entry = raw["entries"][0]
    if change == "missing_source":
        entry.pop("source")
    elif change == "unknown_field":
        entry["write_memory"] = True
    elif change == "bool_version":
        raw["schema_version"] = True
    elif change == "oversized_aliases":
        entry["aliases"] = [str(i) for i in range(17)]
    elif change == "blank":
        entry["aliases"] = ["  "]
    else:
        entry["category"] = None
    with pytest.raises(ValueError):
        EntityAliases.from_document(raw)


def test_empty_namespace_install_is_atomic_private_and_cannot_be_overwritten(tmp_path):
    root = tmp_path / "new"
    assert load_entity_aliases(root) is None and not root.exists()
    expected = install_entity_aliases(root, document())
    assert load_entity_aliases(root).digest == expected.digest
    file = root / "entity-aliases.json"
    assert file.stat().st_mode & 0o777 == 0o600
    old = file.read_bytes()
    with pytest.raises(FileExistsError):
        install_entity_aliases(root, document())
    assert file.read_bytes() == old


@pytest.mark.parametrize("kind", ["memory", "feedback", "pending"])
def test_existing_namespace_needs_explicit_migration_not_alias_install(tmp_path, kind):
    if kind == "memory":
        (tmp_path / "existing.jsonl").write_text("")
    else:
        p = tmp_path / ("feedback" if kind == "feedback" else ".management")
        p.mkdir()
        (p / "existing.json").write_text("{}")
    with pytest.raises(ValueError):
        install_entity_aliases(tmp_path, document())
    assert not (tmp_path / "entity-aliases.json").exists()


def test_duplicate_json_key_and_symlink_registry_are_rejected(tmp_path):
    source = tmp_path / "entity-aliases.json"
    source.write_text('{"schema_version":1,"schema_version":1,"entries":[]}')
    with pytest.raises(ValueError):
        load_entity_aliases(tmp_path)
    source.unlink()
    target = tmp_path / "target.json"
    target.write_text(json.dumps(document()))
    source.symlink_to(target)
    with pytest.raises(ValueError):
        load_entity_aliases(tmp_path)


def test_config_cli_defaults_to_plan_and_only_initializes_explicit_empty_namespace(tmp_path):
    import subprocess
    import sys
    from pathlib import Path

    source = tmp_path / "input.json"
    source.write_text(json.dumps(document()))
    root = tmp_path / "memory"
    script = Path(__file__).parents[1] / "scripts/configure_entity_aliases.py"
    command = [sys.executable, str(script), "--root", str(root), "--input", str(source)]
    plan = subprocess.run(command, capture_output=True, text=True)
    assert plan.returncode == 0, plan.stderr
    assert json.loads(plan.stdout)["mode"] == "plan" and not root.exists()
    applied = subprocess.run(command + ["--initialize"], capture_output=True, text=True)
    assert applied.returncode == 0 and json.loads(applied.stdout)["mode"] == "initialized"
    assert load_entity_aliases(root).canonical("phone", "BrandA") == "品牌A"
