"""用真实JSONL文件验证时间、子集、候选和展示目录互不越界。"""

import json

import pytest

from memrecagent.amazon_preparation import prepare_amazon, selected_user

TRAIN = 1640995200000
VALIDATION = 1656633600000
TEST_END = 1672531200000


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def row(user, parent, rating, timestamp):
    return {
        "user_id": user,
        "parent_asin": parent,
        "asin": parent + "-variant",
        "rating": rating,
        "timestamp": timestamp,
        "text": "不应进入处理文件的合成文本",
    }


def test_user_bucket_is_stable_and_does_not_use_activity_or_python_hash():
    assert selected_user("alice", modulus=16, bucket=0)
    assert not selected_user("alice", modulus=16, bucket=1)
    assert selected_user("anyone", modulus=1, bucket=0)
    with pytest.raises(ValueError):
        selected_user("alice", modulus=0, bucket=0)


def test_preparation_uses_only_training_popularity_and_does_not_drop_missing_metadata(tmp_path):
    reviews, meta = tmp_path / "reviews", tmp_path / "meta"
    write_jsonl(
        reviews,
        [
            row("u", "P", 5, TRAIN - 2),
            row("v", "P", 1, TRAIN - 1),
            row("u", "Q", 2, TRAIN - 1),
            row("u", "P", 1, TRAIN),
            row("cold", "FUTURE", 5, VALIDATION),
            row("later", "Z", 5, TEST_END),
        ],
    )
    write_jsonl(meta, [{"parent_asin": "Q", "title": "Q商品"}])
    output = tmp_path / "prepared"
    report = prepare_amazon(reviews, meta, output, modulus=1, catalog_limit=1)
    assert json.loads((output / "candidates.json").read_text()) == ["P"]
    assert report["selected_window_counts"] == {"train": 3, "validation": 1, "test": 1}
    assert report["n_selected_events"] == 5
    assert report["n_source_reviews"] == 6
    assert report["n_candidate_items_without_metadata"] == 1
    assert report["n_selected_items_without_metadata"] == 2
    normalized = (output / "events.jsonl").read_text()
    assert "合成文本" not in normalized
    assert len(normalized.splitlines()) == 5
    assert len(json.loads((output / "products.json").read_text())) == 1


def test_future_only_item_activity_cannot_select_a_training_candidate(tmp_path):
    reviews, meta = tmp_path / "reviews", tmp_path / "meta"
    write_jsonl(meta, [])
    old = [row("u", "B", 5, TRAIN - 1), row("v", "A", 5, TRAIN - 1)]
    write_jsonl(reviews, old)
    prepare_amazon(reviews, meta, tmp_path / "first", modulus=1, catalog_limit=1)
    write_jsonl(reviews, [*old, *[row(str(i), "Z", 5, TRAIN) for i in range(8)]])
    prepare_amazon(reviews, meta, tmp_path / "second", modulus=1, catalog_limit=1)
    assert json.loads((tmp_path / "first/candidates.json").read_text()) == ["A"]
    assert (tmp_path / "first/candidates.json").read_bytes() == (
        tmp_path / "second/candidates.json"
    ).read_bytes()


def test_resource_limit_rejects_instead_of_silently_truncating_events(tmp_path):
    reviews, meta = tmp_path / "reviews", tmp_path / "meta"
    write_jsonl(reviews, [row("u", "P", 5, TRAIN - 1), row("v", "P", 5, TRAIN - 1)])
    write_jsonl(meta, [])
    output = tmp_path / "prepared"
    with pytest.raises(ValueError, match="预算"):
        prepare_amazon(reviews, meta, output, modulus=1, max_events=1)
    assert not output.exists()


def test_existing_output_and_malformed_input_are_never_overwritten_or_hidden(tmp_path):
    reviews, meta = tmp_path / "reviews", tmp_path / "meta"
    reviews.write_text('{"broken":\n')
    write_jsonl(meta, [])
    with pytest.raises(ValueError, match="reviews:1"):
        prepare_amazon(reviews, meta, tmp_path / "bad", modulus=1)
    output = tmp_path / "existing"
    output.mkdir()
    marker = output / "keep"
    marker.write_text("existing")
    with pytest.raises(FileExistsError):
        prepare_amazon(reviews, meta, output, modulus=1)
    assert marker.read_text() == "existing"


def test_source_hash_mismatch_prevents_publishing_processed_data(tmp_path):
    reviews, meta = tmp_path / "reviews", tmp_path / "meta"
    write_jsonl(reviews, [row("u", "P", 5, TRAIN - 1)])
    write_jsonl(meta, [])
    output = tmp_path / "prepared"
    with pytest.raises(ValueError, match="SHA256"):
        prepare_amazon(
            reviews,
            meta,
            output,
            modulus=1,
            expected_hashes={"reviews": "0" * 64, "meta": "0" * 64},
        )
    assert not output.exists()
