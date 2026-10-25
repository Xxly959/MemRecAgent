"""真实画像机制审计：不借用别人的向量给新用户，不按得分挑目标。"""

from datetime import timedelta

import pytest
from test_memory import T0
from test_product_service import service

from memrecagent.product_bundle import save_bundle
from memrecagent.warm_profile_audit import run_profile_audit, select_profiles, target_item


def test_only_actual_active_model_owner_with_candidate_is_selected():
    s = service()
    assert select_profiles(s, count=1) == ("u",)
    s.seen = {"outside": {"OUTSIDE"}}
    with pytest.raises(ValueError):
        select_profiles(s, count=1)


def test_selection_and_target_do_not_depend_on_model_scores_or_counts():
    s = service()
    assert target_item(s, "u") in ("B", "C")
    before = target_item(s, "u")
    s.counts = {"C": 1000000, "B": 0, "A": 0}
    assert select_profiles(s, count=1) == ("u",)
    assert target_item(s, "u") == before
    assert s.seen["u"] == {"A"}


@pytest.mark.parametrize("count", [0, True, 65])
def test_profile_count_is_bounded_and_not_silently_replaced(count):
    with pytest.raises(ValueError):
        select_profiles(service(), count=count)


def test_runner_preserves_bundle_and_marks_counterfactual_not_language_gain(tmp_path):
    source = tmp_path / "bundle"
    save_bundle(service(), source, provenance={"scope": "synthetic engineering"})
    before = {p.name: p.read_bytes() for p in source.iterdir()}
    result = run_profile_audit(
        bundle=source,
        output=tmp_path / "audit",
        count=1,
        cold_count=1,
        as_of=T0 + timedelta(days=10),
        category="catalog_items",
        run=True,
    )
    assert result["evidence_kind"] == "real_profile_structured_counterfactual"
    assert result["language_memory_evaluated"] is False and result["utility_evaluated"] is False
    assert result["n_execution_errors"] == 0 and result["n_cases"] == 4
    assert result["by_stratum"]["active"]["algorithm_effective_arms"] == 4
    assert result["by_stratum"]["cold"]["algorithm_effective_arms"] == 0
    assert {p.name: p.read_bytes() for p in source.iterdir()} == before
    with pytest.raises(FileExistsError):
        run_profile_audit(
            bundle=source,
            output=tmp_path / "audit",
            count=1,
            cold_count=1,
            as_of=T0 + timedelta(days=10),
            category="catalog_items",
            run=True,
        )


def test_default_plan_does_not_read_bundle_or_create_output(tmp_path):
    result = run_profile_audit(
        bundle=tmp_path / "absent",
        output=tmp_path / "none",
        count=16,
        cold_count=4,
        as_of=T0,
        category="musical_instruments",
        run=False,
    )
    assert result["mode"] == "plan" and not (tmp_path / "none").exists()
