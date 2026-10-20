"""冻结包应独立于原始对象，版本和每个文件损坏均拒绝加载。"""

import json

import pytest
import torch
from test_product_service import service

from memrecagent.product_bundle import load_bundle, save_bundle


def test_saved_bundle_replays_and_does_not_follow_later_source_changes(tmp_path):
    original = service()
    path = tmp_path / "bundle"
    manifest = save_bundle(original, path, provenance={"kind": "synthetic"})
    restored = load_bundle(path)
    result = restored.recommend("u", bundle_id=manifest["bundle_id"], strategy="mf", k=2)
    assert [item["item_id"] for item in result["items"]] == ["C", "B"]
    assert [item["score"] for item in result["items"]] == [-1.0, -2.0]
    with torch.no_grad():
        original.models["mf"].model.item_vectors.weight.add_(100)
    original.products["B"]["title"] = "后来的源快照变化"
    again = load_bundle(path).recommend("u", bundle_id=manifest["bundle_id"], strategy="mf", k=2)
    assert again == result


@pytest.mark.parametrize("filename", ["service.json", "mf.pt", "bpr.pt"])
def test_corrupted_bundle_file_is_rejected_by_digest_before_serving(tmp_path, filename):
    path = tmp_path / "bundle"
    save_bundle(service(), path, provenance={"kind": "synthetic"})
    with (path / filename).open("ab") as stream:
        stream.write(b"corrupt")
    with pytest.raises(ValueError, match="SHA256"):
        load_bundle(path)


def test_changed_historical_cutoff_cannot_keep_the_original_bundle_identity(tmp_path):
    path = tmp_path / "bundle"
    save_bundle(service(), path, provenance={"kind": "synthetic"})
    manifest_path = path / "bundle.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["as_of_ms"] += 1000
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="版本"):
        load_bundle(path)


def test_export_never_overwrites_an_existing_directory(tmp_path):
    path = tmp_path / "bundle"
    path.mkdir()
    (path / "keep").write_text("existing")
    with pytest.raises(FileExistsError):
        save_bundle(service(), path, provenance={"kind": "synthetic"})
    assert (path / "keep").read_text() == "existing"


def test_missing_categories_survive_freeze_as_unknown_instead_of_breaking_requests(tmp_path):
    original = service()
    del original.products["B"]["categories"]
    manifest = save_bundle(original, tmp_path / "bundle", provenance={"kind": "synthetic"})
    result = load_bundle(tmp_path / "bundle").recommend(
        "u", bundle_id=manifest["bundle_id"], strategy="mf"
    )
    assert result["items"][1]["categories"] == []
