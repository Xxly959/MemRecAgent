"""属性来自可信目录；硬约束在完整候选过滤，未知不冒充满足。"""

import json
import os
import subprocess
import sys
from dataclasses import replace
from datetime import timedelta

import pytest
from test_memory import T0
from test_memory_ablation import fitted_ranker_inputs
from test_memory_ablation_replay import command, replay_fixture
from test_memory_product_tool import fixture, run_tool
from test_memory_products import AT, engine
from test_memory_products import request as base_request

from memrecagent.memory_ablation import run_ranking_ablation
from memrecagent.product_attributes import ProductAttributeValue, RequestConstraint


def request():
    return base_request() | {"request_id": "R1"}


def value(attribute="price_cny", value=2500, unit="CNY", **extra):
    return ProductAttributeValue(attribute, value, unit, "catalog:fixed", extra.get("at", T0))


def constraint(attribute="price_cny", operator="<=", value=3000, unit="CNY", **extra):
    return RequestConstraint(
        "demo-user",
        "S2",
        extra.get("person", "self"),
        attribute,
        operator,
        value,
        unit,
        "S2:T1",
        extra.get("at", T0),
        "R1",
    )


def with_values(tmp_path, rows):
    ranker = engine(tmp_path)
    ranker._facts = tuple(
        replace(f, attributes=tuple(rows.get(f.item_id, ()))) for f in ranker._facts
    )
    return ranker


def test_hard_budget_filters_before_top_k_and_memory_off_does_not_disable_it(tmp_path):
    ranker = with_values(
        tmp_path, {"A": [value(value=4000)], "B": [value(value=3000)], "C": [value(value=2000)]}
    )
    result = ranker.recommend(
        **(request() | {"k": 1}), constraints=(constraint(),), use_long_term_memory=False
    )
    assert [r["item_id"] for r in result["items"]] == ["B"]
    assert result["filter_counts"] == {"attribute_constraint": 1}
    # 基础分归一化仍用同一个完整候选池，不因预算改变特征。
    assert result["items"][0]["normalised_base_score"] == 0.5


def test_missing_future_and_exact_query_time_attributes_remain_unknown(tmp_path):
    ranker = with_values(
        tmp_path,
        {
            "A": [],
            "B": [value(value=1000, at=AT)],
            "C": [value(value=1000, at=AT + timedelta(seconds=1))],
        },
    )
    result = ranker.recommend(**request(), constraints=(constraint(),))
    assert result["items"] == []
    assert result["filter_counts"] == {"attribute_unknown": 3}
    assert result["empty_reason"] == "no_matching_candidates"


def test_enum_boolean_and_multiple_numeric_conditions_are_conjunctive(tmp_path):
    rows = {
        "A": [value(), value("screen_type", "OLED", "type"), value("eye_care", False, "bool")],
        "B": [value(), value("screen_type", "OLED", "type"), value("eye_care", True, "bool")],
        "C": [value(), value("screen_type", "LCD", "type"), value("eye_care", True, "bool")],
    }
    result = with_values(tmp_path, rows).recommend(
        **request(),
        constraints=(
            constraint(),
            constraint("screen_type", "==", "OLED", "type"),
            constraint("eye_care", "==", True, "bool"),
        ),
    )
    assert [r["item_id"] for r in result["items"]] == ["B"]


@pytest.mark.parametrize(
    "field,bad",
    [
        ("value", True),
        ("value", float("nan")),
        ("value", -1),
        ("unit", "USD"),
        ("operator", "approximately"),
        ("attribute", "unknown"),
    ],
)
def test_invalid_predicates_cannot_enter_ranking(field, bad):
    original = constraint()
    with pytest.raises(ValueError):
        replace(original, **{field: bad})


@pytest.mark.parametrize(
    "field,bad",
    [
        ("user_id", "other"),
        ("session_id", "S1"),
        ("beneficiary_id", "relative:uncle"),
        ("available_at", AT),
    ],
)
def test_current_constraint_binding_or_future_evidence_cannot_be_reused(tmp_path, field, bad):
    with pytest.raises(ValueError):
        engine(tmp_path).recommend(
            **request(), constraints=(replace(constraint(), **{field: bad}),)
        )


def test_attribute_value_unit_and_duplicate_catalog_fields_are_rejected(tmp_path):
    with pytest.raises(ValueError):
        value(unit="USD")
    with pytest.raises(ValueError):
        with_values(tmp_path, {"A": [value(), value(value=2000)]})


def test_other_beneficiary_uses_their_bound_constraint_and_public_scores(tmp_path):
    ranker = with_values(tmp_path, {"A": [value(value=4000)], "B": [value()], "C": [value()]})
    result = ranker.recommend(
        **(request() | {"beneficiary_id": "relative:uncle"}),
        constraints=(constraint(person="relative:uncle"),),
    )
    assert [r["item_id"] for r in result["items"]] == ["B", "C"]
    assert result["routing_reason"] == "non_self_public_baseline"


def test_same_session_cannot_replay_budget_from_another_request(tmp_path):
    old = replace(constraint(), request_id="old-request")
    with pytest.raises(ValueError):
        engine(tmp_path).recommend(
            **(request() | {"request_id": "new-request"}), constraints=(old,)
        )


def test_conversation_loader_accepts_enriched_catalog_without_model_calls(tmp_path):
    fixture(tmp_path)
    path = tmp_path / "facts.json"
    facts = json.loads(path.read_text())
    facts["facts"][0]["attributes"] = [
        dict(
            attribute="price_cny",
            value=2500,
            unit="CNY",
            evidence_ref="synthetic",
            available_at=T0.isoformat(),
        )
    ]
    path.write_text(json.dumps(facts))
    payload = dict(version=1, id="check", method="check_state", params=dict(state_token="0" * 64))
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "memrecagent.conversation_bridge",
            "--live",
            "--bundle",
            str(tmp_path / "bundle"),
            "--facts",
            str(path),
            "--memory-root",
            str(tmp_path / "memory"),
            "--user-id",
            "demo-user",
            "--session-id",
            "S2",
        ],
        input=json.dumps(payload).encode() + b"\n",
        capture_output=True,
        timeout=15,
        env=os.environ | {"DEEPSEEK_API_KEY": "not-a-real-key"},
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["error"]["code"] == "conversation_state"


@pytest.mark.parametrize("bad", [None, 1, "bad-row", []])
def test_malformed_fact_row_gets_fixed_error_without_traceback(tmp_path, bad):
    fixture(tmp_path)
    path = tmp_path / "facts.json"
    facts = json.loads(path.read_text())
    facts["facts"][0] = bad
    path.write_text(json.dumps(facts))
    result = run_tool(tmp_path)
    assert result.returncode == 2 and result.stdout == b""
    assert result.stderr == b"recommendation_unavailable\n"


def test_actual_tool_decodes_attributes_and_budget_change_invalidates_receipt(tmp_path):
    fixture(tmp_path)
    path = tmp_path / "facts.json"
    facts = json.loads(path.read_text())
    for row, price in zip(facts["facts"], (4000, 2500, 2000), strict=True):
        row["attributes"] = [
            dict(
                attribute="price_cny",
                value=price,
                unit="CNY",
                evidence_ref="catalog:fixed",
                available_at=T0.isoformat(),
            )
        ]
    path.write_text(json.dumps(facts))
    path = tmp_path / "request.json"
    query = json.loads(path.read_text())
    query["request_id"] = "R1"
    query["constraints"] = [
        dict(
            user_id="demo-user",
            session_id="S2",
            beneficiary_id="self",
            attribute="price_cny",
            operator="<=",
            value=3000,
            unit="CNY",
            evidence_ref="S2:T1",
            available_at=T0.isoformat(),
            request_id="R1",
        )
    ]
    path.write_text(json.dumps(query))
    result = run_tool(tmp_path)
    assert result.returncode == 0, result.stderr
    receipt = json.loads(result.stdout)
    assert [r["item_id"] for r in receipt["items"]] == ["B", "C"]
    query["constraints"][0]["value"] = 4500
    path.write_text(json.dumps(query))
    assert run_tool(tmp_path, "--check-token", receipt["delivery_token"]).returncode == 2
    assert [r["item_id"] for r in json.loads(run_tool(tmp_path).stdout)["items"]] == ["A", "B", "C"]


def test_four_arms_keep_same_budget_and_full_normalisation_pool(tmp_path):
    service, store, facts = fitted_ranker_inputs(tmp_path)
    enriched = tuple(
        replace(f, attributes=(value(value=4000 if f.item_id == "A" else 2000),)) for f in facts
    )
    budget = replace(constraint(), user_id="u")
    before = store.snapshot_token("u")
    result = run_ranking_ablation(
        service,
        store,
        enriched,
        request=request() | dict(user_id="u", strategy="bpr", constraints=(budget,)),
    )
    assert {g: [i["item_id"] for i in r["ranking"]["items"]] for g, r in result.items()} == {
        "G00": ["B", "C"],
        "G10": ["B", "C"],
        "G01": ["C", "B"],
        "G11": ["B", "C"],
    }
    assert all(
        r["ranking"]["n_candidates"] == 3 and r["ranking"]["n_constraints"] == 1
        for r in result.values()
    )
    assert store.snapshot_token("u") == before


def test_offline_ablation_cli_accepts_same_typed_budget_in_all_groups(tmp_path):
    suite = replay_fixture(tmp_path)
    path = tmp_path / "facts.json"
    facts = json.loads(path.read_text())
    for row in facts["facts"]:
        row["attributes"] = [
            dict(
                attribute="price_cny",
                value=2000 if row["item_id"] == "C" else 4000,
                unit="CNY",
                evidence_ref="synthetic",
                available_at=T0.isoformat(),
            )
        ]
    path.write_text(json.dumps(facts))
    query = suite["cases"][0]["request"]
    query["request_id"] = "R1"
    query["constraints"] = [
        dict(
            user_id="u",
            session_id="S2",
            beneficiary_id="self",
            request_id="R1",
            attribute="price_cny",
            operator="<=",
            value=3000,
            unit="CNY",
            evidence_ref="S2:T1",
            available_at=T0.isoformat(),
        )
    ]
    (tmp_path / "suite.json").write_text(json.dumps(suite))
    result = subprocess.run(command(tmp_path, "--run-offline"), capture_output=True, timeout=15)
    assert result.returncode == 0, result.stderr
    case = json.loads((tmp_path / "output/cases.jsonl").read_text())
    assert all(group["item_ids"] == ["C"] for group in case["groups"].values())
