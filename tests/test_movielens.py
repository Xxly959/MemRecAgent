"""Guard parsing and causal boundaries before fitting any model."""

import pytest

from memrecagent import movielens


def test_loader_preserves_fields_and_accepts_shuffled_rows(tmp_path):
    path = tmp_path / "u.data"
    path.write_text("2\t8\t4\t30\n1\t3\t2\t10\n")
    rows = movielens.load_ratings(path)
    assert [(r.user_id, r.item_id, r.rating, r.timestamp) for r in rows] == [
        ("2", "8", 4.0, 30),
        ("1", "3", 2.0, 10),
    ]


@pytest.mark.parametrize(
    "row",
    [
        "1\t2\t5\n",
        "x\t2\t5\t10\n",
        "1\t0\t5\t10\n",
        "1\t2\t6\t10\n",
        "1\t2\t4.5\t10\n",
        "1\t2\t5\t-1\n",
        "1\t2\t5\tbad\n",
        "\n",
        "",
    ],
)
def test_loader_rejects_invalid_records(tmp_path, row):
    path = tmp_path / "u.data"
    path.write_text(row)
    with pytest.raises(ValueError):
        movielens.load_ratings(path)


@pytest.mark.parametrize("second_rating", ["5", "1"])
def test_loader_rejects_duplicate_pairs_even_with_different_time(tmp_path, second_rating):
    path = tmp_path / "u.data"
    path.write_text(f"1\t2\t5\t10\n1\t2\t{second_rating}\t20\n")
    with pytest.raises(ValueError, match="Duplicate"):
        movielens.load_ratings(path)


def test_split_uses_global_timestamps_and_exact_inclusive_boundaries():
    r = movielens.Rating
    rows = [r("u2", "D", 5, 30), r("u1", "B", 2, 20), r("u1", "A", 5, 19), r("u2", "C", 4, 20)]
    train, validation, test = movielens.temporal_split(rows, 20, 30)
    assert [x.item_id for x in train] == ["A"]
    assert {x.item_id for x in validation} == {"B", "C"}
    assert [x.item_id for x in test] == ["D"]
    assert len(train) + len(validation) + len(test) == len(rows)


@pytest.mark.parametrize("train_end,validation_end", [(20, 20), (30, 20)])
def test_split_rejects_non_increasing_boundaries(train_end, validation_end):
    with pytest.raises(ValueError):
        movielens.temporal_split([], train_end, validation_end)


def test_empty_windows_are_explicit():
    assert movielens.temporal_split([], 20, 30) == ([], [], [])
