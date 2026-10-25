"""真实ID环境只绑定本机画像，不进入模型消息，不借用手机语义标签。"""

import json
from datetime import datetime, timedelta

import pytest
from test_conversation_domain import raw
from test_memory_interaction import turn
from test_product_service import service

from memrecagent.deepseek_client import ModelReply, ProviderError
from memrecagent.external_replay import evaluate_external_replay
from memrecagent.instrument_catalog import build_instrument_mapping, build_instrument_title_aliases
from memrecagent.native_replay import load_native_environment
from memrecagent.product_bundle import save_bundle


def fixture(tmp_path, user="native-user-private-123456"):
    base = service()
    base.counts = dict(A=10, B=9, C=8)
    base.seen = {"u": set()} if user == "u" else {}
    base.products = {
        i: dict(title=t, categories=["Musical Instruments"])
        for i, t in zip("ABC", ["Studio Guitar", "Travel Drums", "Concert Harp"], strict=True)
    }
    bundle = tmp_path / "bundle"
    manifest = save_bundle(base, bundle, provenance=dict(kind="synthetic"))
    (tmp_path / "facts.json").write_text(json.dumps(build_instrument_mapping(bundle)))
    aliases, _ = build_instrument_title_aliases(bundle)
    (tmp_path / "aliases.json").write_text(json.dumps(aliases))
    (tmp_path / "bindings.json").write_text(json.dumps({"E1": user}))
    config = dict(
        bundle="bundle",
        facts="facts.json",
        aliases="aliases.json",
        user_bindings="bindings.json",
        base_at=(
            datetime.fromisoformat(manifest["created_at_utc"]) + timedelta(seconds=1)
        ).isoformat(),
    )
    (tmp_path / "environment.json").write_text(json.dumps(config))
    fact = dict(
        subject="self",
        category="musical_instruments",
        target_type="product",
        entity="A",
        relation="exclude",
    )
    steps = [
        dict(
            turn_id="T1",
            session_id="S1",
            offset_seconds=10,
            reply_to=None,
            text="以后请长期排除Studio Guitar。",
            reference=dict(long_term_state_after=[fact], action="answer"),
            reference_reviews=[],
        ),
        dict(
            turn_id="T2",
            session_id="S2",
            offset_seconds=20,
            reply_to=None,
            text="给我推荐三款乐器",
            reference=dict(long_term_state_after=[fact], action="recommend"),
            reference_reviews=[],
        ),
    ]
    suite = dict(
        schema_version=1,
        synthetic=True,
        episodes=[dict(episode_id="E1", template_family="exclude", steps=steps)],
        tasks=[
            dict(
                episode_id="E1",
                task_id="Q1",
                turn_id="T2",
                split="test",
                runtime=dict(request_text=steps[1]["text"]),
                reference=dict(
                    status="provisional",
                    review_codes=[],
                    item_scores=[dict(item_id=i, relevance=0, eligible=i != "A") for i in "ABC"],
                ),
            )
        ],
    )
    return suite, tmp_path / "environment.json"


def complete(messages):
    p = json.loads(messages[-1]["content"])
    assert "native-user-private-123456" not in json.dumps(messages)
    assert "reference" not in p and "item_scores" not in p
    latest = p["turns"][-1]
    t = turn(int(latest["turn_id"][1:]), latest["content"])
    candidate = (
        raw(t, entity="Studio Guitar", relation="exclude")
        if t.turn_id == "T1"
        else raw(t, "recommend")
    )
    return ModelReply(json.dumps(candidate), "fixed", None, {"prompt_tokens": 1}, 0.0)


@pytest.mark.parametrize("mode,calls", [("direct", 2), ("analyst", 4)])
def test_native_replay_consumes_catalog_titles_without_inventing_utility(tmp_path, mode, calls):
    suite, path = fixture(tmp_path)
    report = evaluate_external_replay(
        suite,
        complete=complete,
        output_dir=tmp_path / "run",
        analysis_mode=mode,
        split="test",
        scoring_mode="request",
        native_environment=load_native_environment(path),
    )
    assert report["state_matches"] == 2 and report["strict_steps"] == 2
    assert report["analysis_calls"] == calls and report["api_calls"] == 0
    assert report["algorithm_effect"] is None
    assert report["groups"]["G11"]["ndcg"] is None
    case = json.loads((tmp_path / "run/cases.jsonl").read_text())
    assert case["groups"]["G00"]["item_ids"] == ["A", "B", "C"]
    assert case["groups"]["G10"]["item_ids"] == ["B", "C"]
    assert case["groups"]["G10"]["ranking"]["items"][0]["category"] == "musical_instruments"
    assert case["scoring_cutoff"]["k"] == 3


def test_native_transport_stops_story_but_can_continue_wholly_unattempted_story(tmp_path):
    suite, path = fixture(tmp_path)
    import copy

    second = copy.deepcopy(suite["episodes"][0])
    second["episode_id"] = "E2"
    suite["episodes"].append(second)
    task = copy.deepcopy(suite["tasks"][0])
    task.update(episode_id="E2", task_id="Q2")
    suite["tasks"].append(task)
    (tmp_path / "bindings.json").write_text(
        json.dumps(dict(E1="native-user-private-123456", E2="cold-second-user"))
    )
    n = 0

    def fail_once(messages):
        nonlocal n
        n += 1
        if n == 1:
            raise ProviderError("transport")
        return complete(messages)

    report = evaluate_external_replay(
        suite,
        complete=fail_once,
        output_dir=tmp_path / "run",
        split="test",
        analysis_mode="direct",
        native_environment=load_native_environment(path),
    )
    assert n == 3 and report["provider_errors"] == 1 and report["not_run"] == 1
    assert report["successful_steps"] == 2


def test_changed_native_binding_is_stopped_before_a_second_request(tmp_path):
    suite, path = fixture(tmp_path)
    environment = load_native_environment(path)
    n = 0

    def changed(messages):
        nonlocal n
        n += 1
        response = complete(messages)
        (tmp_path / "bindings.json").write_text("{}")
        return response

    report = evaluate_external_replay(
        suite,
        complete=changed,
        output_dir=tmp_path / "run",
        split="test",
        analysis_mode="direct",
        native_environment=environment,
    )
    assert n == 1 and report["inputs_source_unchanged"] is False
    assert report["provider_errors"] == 1


def test_actual_active_bpr_arm_is_distinguished_from_cold_fallback(tmp_path):
    suite, path = fixture(tmp_path, user="u")
    report = evaluate_external_replay(
        suite,
        complete=complete,
        output_dir=tmp_path / "run",
        split="test",
        analysis_mode="direct",
        native_environment=load_native_environment(path),
    )
    case = json.loads((tmp_path / "run/cases.jsonl").read_text())
    assert report["active_algorithm_case_arms"] == 2
    assert case["groups"]["G01"]["used_strategy"] == "bpr"
    assert case["groups"]["G01"]["algorithm_effective"] is True
    assert report["algorithm_estimand_identified"] is False


def test_cli_native_plan_does_not_read_key_or_create_run(tmp_path):
    import hashlib
    import subprocess
    import sys
    from pathlib import Path

    suite, path = fixture(tmp_path)
    raw_suite = json.dumps(suite).encode()
    (tmp_path / "suite.json").write_bytes(raw_suite)
    (tmp_path / "manifest.json").write_text(
        json.dumps(dict(derived_sha256=hashlib.sha256(raw_suite).hexdigest()))
    )
    command = [
        sys.executable,
        str(Path("scripts/evaluate_external_replay.py").resolve()),
        "--suite",
        str(tmp_path / "suite.json"),
        "--native-environment",
        str(path),
        "--output-dir",
        str(tmp_path / "run"),
        "--split",
        "test",
        "--analysis-mode",
        "analyst",
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    plan = json.loads(result.stdout)
    assert plan["api_calls"] == 0 and plan["planned_max_calls"] == 4
    assert plan["native_environment"] is True and not (tmp_path / "run").exists()


@pytest.mark.parametrize("correct_subject", [False, True])
def test_every_recommendation_records_actual_subject_not_only_final_state(
    tmp_path, correct_subject
):
    from test_memory_extraction import subject
    from test_memory_interaction_extraction import quote

    suite, path = fixture(tmp_path)
    step = suite["episodes"][0]["steps"][1]
    step["text"] = "给我叔叔推荐三款乐器。"
    step["reference"]["beneficiary"] = "other"
    suite["tasks"][0]["runtime"]["request_text"] = step["text"]

    def client(messages):
        p = json.loads(messages[-1]["content"])
        latest = p["turns"][-1]
        if latest["turn_id"] == "T1":
            return complete(messages)
        t = turn(2, latest["content"])
        candidate = raw(t, "recommend")
        if correct_subject:
            candidate["candidate"]["request"]["beneficiary"] = subject(
                "other", [quote(t, "我叔叔")]
            )
        return ModelReply(json.dumps(candidate), "fixed", None, None, 0.0)

    report = evaluate_external_replay(
        suite,
        complete=client,
        output_dir=tmp_path / "run",
        split="test",
        analysis_mode="direct",
        native_environment=load_native_environment(path),
    )
    rows = [json.loads(s) for s in (tmp_path / "run/steps.jsonl").read_text().splitlines()]
    assert report["state_matches"] == 2
    assert report["strict_steps"] == (2 if correct_subject else 1)
    assert rows[1]["subject_match"] is correct_subject
    assert rows[1]["result"]["ranking"]["routing_reason"] == (
        "non_self_public_baseline" if correct_subject else "owner_profile"
    )
