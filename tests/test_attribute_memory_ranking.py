"""长期属性的实际过滤、当前覆盖与四组边界；不以预置偏好冒称语言收益。"""

from dataclasses import replace

import pytest
from test_attribute_memory_store import attribute
from test_memory_ablation import fitted_ranker_inputs
from test_product_attributes import constraint, request, value, with_values

from memrecagent.memory_ablation import run_ranking_ablation
from memrecagent.memory_store import JsonlMemoryStore
from memrecagent.memory_use_policy import JsonMemoryUseStore, MemoryUsePolicy


def priced(tmp_path):
    return with_values(tmp_path, {"A": [value(value=4000)], "B": [value(value=2500)], "C": []})


def test_long_term_hard_budget_filters_and_current_same_direction_overrides(tmp_path):
    store = JsonlMemoryStore(tmp_path)
    store.add(attribute())
    before = store.snapshot_token("demo-user")
    ranker = priced(tmp_path)
    assert [i["item_id"] for i in ranker.recommend(**request())["items"]] == ["B"]
    override = ranker.recommend(**request(), constraints=(constraint(value=4500),))
    assert [i["item_id"] for i in override["items"]] == ["A", "B"]
    assert override["overridden_attribute_memory_ids"] == ["a1"]
    assert store.snapshot_token("demo-user") == before


@pytest.mark.parametrize("mode", ["off", "other", "denied"])
def test_disabled_other_or_denied_path_does_not_load_personal_attribute_view(
    tmp_path, monkeypatch, mode
):
    store = JsonlMemoryStore(tmp_path)
    store.add(attribute())
    if mode == "denied":
        JsonMemoryUseStore(tmp_path / "use-policy").set(
            "demo-user", MemoryUsePolicy(personalisation=False)
        )

    def forbidden(*args, **kwargs):
        pytest.fail("disabled path read attribute view")

    monkeypatch.setattr(JsonlMemoryStore, "active_attribute_memories", forbidden)
    result = priced(tmp_path).recommend(
        **(
            request()
            | dict(
                use_long_term_memory=mode != "off",
                beneficiary_id="uncle" if mode == "other" else "self",
            )
        )
    )
    assert [i["item_id"] for i in result["items"]] == ["A", "B", "C"]


def test_soft_attribute_is_preserved_but_does_not_become_a_hard_filter(tmp_path):
    JsonlMemoryStore(tmp_path).add(attribute(is_hard=False))
    assert {i["item_id"] for i in priced(tmp_path).recommend(**request())["items"]} == {
        "A",
        "B",
        "C",
    }


def test_current_opposite_direction_does_not_silently_erase_permanent_other_bound(tmp_path):
    JsonlMemoryStore(tmp_path).add(attribute(operator=">=", value=3000))
    result = priced(tmp_path).recommend(**request(), constraints=(constraint(value=4500),))
    assert [i["item_id"] for i in result["items"]] == ["A"]


def test_four_arms_differ_only_on_long_term_attribute_use(tmp_path):
    service, store, facts = fitted_ranker_inputs(tmp_path)
    store = JsonlMemoryStore(tmp_path / "attribute-only")
    store.add(attribute(user_id="u"))
    enriched = tuple(
        replace(f, attributes=(value(value=4000 if f.item_id == "A" else 2000),)) for f in facts
    )
    result = run_ranking_ablation(
        service, store, enriched, request=request() | dict(user_id="u", strategy="bpr")
    )
    assert {i["item_id"] for i in result["G00"]["ranking"]["items"]} == {"A", "B", "C"}
    assert {i["item_id"] for i in result["G01"]["ranking"]["items"]} == {"A", "B", "C"}
    assert {i["item_id"] for i in result["G10"]["ranking"]["items"]} == {"B", "C"}
    assert {i["item_id"] for i in result["G11"]["ranking"]["items"]} == {"B", "C"}


def relaxation(**changes):
    from test_memory import T0

    from memrecagent.product_attributes import RequestRelaxation

    return RequestRelaxation(
        **(
            dict(
                user_id="demo-user",
                session_id="S2",
                beneficiary_id="self",
                request_id="R1",
                attribute="price_cny",
                evidence_ref="S2:T2",
                available_at=T0,
            )
            | changes
        )
    )


def test_explicit_relaxation_removes_all_long_term_bounds_without_mutating_disk(tmp_path):
    store = JsonlMemoryStore(tmp_path)
    store.add_many((attribute(), attribute(memory_id="lower", operator=">=", value=2000)))
    before = store.snapshot_token("demo-user")
    result = priced(tmp_path).recommend(**request(), relaxations=(relaxation(),))
    assert [i["item_id"] for i in result["items"]] == ["A", "B", "C"]
    assert set(result["overridden_attribute_memory_ids"]) == {"a1", "lower"}
    assert store.snapshot_token("demo-user") == before


@pytest.mark.parametrize(
    "changes",
    [
        dict(user_id="other"),
        dict(session_id="S1"),
        dict(beneficiary_id="uncle"),
        dict(request_id="old"),
        dict(attribute="unknown"),
    ],
)
def test_relaxation_requires_current_request_person_and_supported_attribute(tmp_path, changes):
    with pytest.raises(ValueError):
        priced(tmp_path).recommend(**request(), relaxations=(relaxation(**changes),))


def test_relaxation_cannot_be_future_duplicate_or_conflict_with_current_constraint(tmp_path):
    from test_memory_products import AT

    for relaxations, constraints in [
        ((relaxation(available_at=AT),), ()),
        ((relaxation(), relaxation()), ()),
        ((relaxation(),), (constraint(),)),
    ]:
        with pytest.raises(ValueError):
            priced(tmp_path).recommend(
                **request(), relaxations=relaxations, constraints=constraints
            )


def test_real_tool_reads_long_term_budget_and_decodes_bound_relaxation(tmp_path):
    import json
    from dataclasses import asdict

    from test_memory_product_tool import fixture, run_tool

    fixture(tmp_path)
    path = tmp_path / "facts.json"
    facts = json.loads(path.read_text())
    for row, price in zip(facts["facts"], (4000, 2500, 2000), strict=True):
        row["attributes"] = [
            asdict(value(value=price)) | {"available_at": value().available_at.isoformat()}
        ]
    path.write_text(json.dumps(facts))
    JsonlMemoryStore(tmp_path / "memory").add(attribute())
    first = run_tool(tmp_path)
    assert first.returncode == 0, first.stderr
    receipt = json.loads(first.stdout)
    assert [i["item_id"] for i in receipt["items"]] == ["B", "C"]
    path = tmp_path / "request.json"
    query = json.loads(path.read_text()) | {
        "request_id": "R1",
        "relaxations": [
            asdict(relaxation()) | {"available_at": relaxation().available_at.isoformat()}
        ],
    }
    path.write_text(json.dumps(query))
    relaxed = run_tool(tmp_path)
    assert relaxed.returncode == 0, relaxed.stderr
    assert [i["item_id"] for i in json.loads(relaxed.stdout)["items"]] == ["A", "B", "C"]
    assert run_tool(tmp_path, "--check-token", receipt["delivery_token"]).returncode == 2


def test_ablation_cli_decodes_relaxations_identically_in_all_arms(tmp_path):
    import json
    import subprocess
    from dataclasses import asdict

    from test_memory_ablation_replay import command, replay_fixture

    suite = replay_fixture(tmp_path)
    # 无本次条件；明确移除长期预算，不能因目录无价格过滤全部候选。
    JsonlMemoryStore(tmp_path / "memory").add(attribute(user_id="u"))
    query = suite["cases"][0]["request"]
    query.update(
        request_id="R1",
        relaxations=[
            asdict(relaxation(user_id="u"))
            | {"available_at": relaxation().available_at.isoformat()}
        ],
    )
    (tmp_path / "suite.json").write_text(json.dumps(suite))
    process = subprocess.run(command(tmp_path, "--run-offline"), capture_output=True, timeout=15)
    assert process.returncode == 0, process.stderr
    case = json.loads((tmp_path / "output/cases.jsonl").read_text())
    assert all(
        group["status"] == "ready" and group["item_ids"] for group in case["groups"].values()
    )
    assert all(
        case["arms"][group]["ranking"]["overridden_attribute_memory_ids"] == ["a1"]
        for group in ("G10", "G11")
    )
