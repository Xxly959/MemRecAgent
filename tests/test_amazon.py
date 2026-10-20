"""商品父ID、毫秒字段与时点视图的行为约束。"""

import math

import pytest

from memrecagent.amazon import history_before, parse_product, parse_review

CUTOFF = 1640995200000  # 2022-01-01 UTC，所有输入均为合成说明。


def review(**changes):
    """生成合成输入，不含真实用户或评论。"""
    return {
        "user_id": "u",
        "parent_asin": "P",
        "asin": "P-red",
        "rating": 5.0,
        "timestamp": CUTOFF - 1000,
        **changes,
    }


def event(**changes):
    return parse_review(review(**changes), source="reviews.jsonl", line_number=7)


def test_review_uses_parent_identity_and_preserves_milliseconds_and_location():
    value = event(timestamp=CUTOFF - 123)
    assert (value.user_id, value.item_id, value.variant_id) == ("u", "P", "P-red")
    assert (value.rating, value.timestamp_ms) == (5.0, 1640995199877)
    assert (value.source, value.line_number) == ("reviews.jsonl", 7)


@pytest.mark.parametrize("field", ["user_id", "parent_asin", "asin", "rating", "timestamp"])
def test_missing_required_review_field_is_rejected_with_location(field):
    row = review()
    del row[field]
    with pytest.raises(ValueError, match=r"reviews.jsonl:7"):
        parse_review(row, source="reviews.jsonl", line_number=7)


@pytest.mark.parametrize("field,value", [("user_id", " "), ("parent_asin", 5), ("asin", None)])
def test_invalid_ids_are_not_coerced_or_guessed(field, value):
    with pytest.raises(ValueError):
        event(**{field: value})


@pytest.mark.parametrize("rating", [True, "5", math.nan, math.inf, 0, 6])
def test_invalid_rating_is_not_an_observed_preference(rating):
    with pytest.raises(ValueError):
        event(rating=rating)


@pytest.mark.parametrize("timestamp", [True, 1640995200, 1640995200000000000, "1640995200000"])
def test_seconds_nanoseconds_and_untyped_time_are_rejected(timestamp):
    with pytest.raises(ValueError):
        event(timestamp=timestamp)


def test_error_does_not_echo_user_identifier_or_review_text():
    with pytest.raises(ValueError) as captured:
        event(user_id="PRIVATE-USER", text="PRIVATE-TEXT", rating="invalid")
    assert "PRIVATE-USER" not in str(captured.value)
    assert "PRIVATE-TEXT" not in str(captured.value)


def test_latest_parent_rating_uses_only_the_strict_time_prefix():
    rows = [
        event(timestamp=CUTOFF - 2000),
        event(asin="P-blue", timestamp=CUTOFF - 1, rating=2),
        event(asin="P-green", timestamp=CUTOFF, rating=4),
        event(parent_asin="FUTURE", timestamp=CUTOFF + 1),
        event(user_id="v", parent_asin="Q", timestamp=CUTOFF - 500),
    ]
    result = history_before(rows, CUTOFF)
    assert [(r.user_id, r.item_id, r.variant_id, r.rating) for r in result] == [
        ("u", "P", "P-blue", 2.0),
        ("v", "Q", "P-red", 5.0),
    ]
    assert history_before(list(reversed(rows)), CUTOFF) == result
    assert len(rows) == 5


def test_future_conflict_cannot_invalidate_or_rewrite_past_history():
    earlier = event()
    later = [event(timestamp=CUTOFF + 1, rating=1), event(timestamp=CUTOFF + 1, rating=5)]
    assert history_before([earlier, *later], CUTOFF) == (earlier,)


def test_equal_timestamp_conflict_in_history_is_explicit_even_if_later_rating_exists():
    with pytest.raises(ValueError, match="reviews.jsonl:7"):
        history_before([event(rating=1), event(rating=5), event(timestamp=CUTOFF - 1)], CUTOFF)


def test_equal_parent_rating_is_merged_deterministically_across_variants():
    red, blue = event(), event(asin="P-blue")
    assert history_before([red, blue], CUTOFF) == (blue,)
    assert history_before([blue, red, blue], CUTOFF) == (blue,)


def test_empty_history_and_all_future_history_are_explicit():
    assert history_before([], CUTOFF) == ()
    assert history_before([event(timestamp=CUTOFF)], CUTOFF) == ()


def test_product_snapshot_keeps_missing_facts_unknown_and_store_distinct_from_brand():
    value = parse_product(
        {"parent_asin": "P", "title": "", "store": "Store", "price": None},
        source="meta.jsonl",
        line_number=3,
    )
    assert value.item_id == "P"
    assert value.title is None
    assert value.categories == ()
    assert value.details == {}
    assert value.store == "Store"
    assert value.source == "meta.jsonl" and value.line_number == 3


def test_product_parser_does_not_alias_input_attributes():
    raw = {"parent_asin": "P", "details": {"Brand": "B", "Nested": ["x"]}}
    value = parse_product(raw, source="meta.jsonl", line_number=1)
    value.details["Nested"].append("y")
    assert raw["details"]["Nested"] == ["x"]


@pytest.mark.parametrize(
    "changes", [{"parent_asin": ""}, {"categories": "Guitars"}, {"details": "{}"}]
)
def test_wrong_raw_product_schema_is_not_silently_converted(changes):
    with pytest.raises(ValueError, match=r"meta.jsonl:2"):
        parse_product({"parent_asin": "P", **changes}, source="meta.jsonl", line_number=2)
