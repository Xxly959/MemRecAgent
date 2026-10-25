"""写入质量评测必须消费实际磁盘状态，不能把参考答案当执行结果。"""

import copy
import json

import pytest
from test_memory_interaction_extraction import envelope

from memrecagent.deepseek_client import ModelReply, ProviderError


def expected(*, writes=(), heads=(), activated=(), actions=("answer",), **changes):
    return dict(
        writes=list(writes),
        heads=list(heads),
        activated=list(activated),
        actions=list(actions),
        other_context=[],
        other_only=[],
        current_only=[],
        quoted_only=[],
        revisions=[],
        **changes,
    )


def suite():
    pref = ["phone", "小米", "prefer"]
    return {
        "schema_version": 1,
        "synthetic": True,
        "label_source": "author-written; not independent or real users",
        "episodes": [
            {
                "episode_id": "dev_self",
                "template_family": "explicit_self_sequence",
                "split": "dev",
                "scenario": "self_preference",
                "people": [],
                "max_memory_age_seconds": None,
                "steps": [
                    dict(
                        turn_id="T1",
                        session_id="S1",
                        offset_seconds=60,
                        text="我以后一直优先选小米手机。",
                        tags=["explicit_self"],
                        reply_to=None,
                        trusted_delete=None,
                        assistant=None,
                        expected=expected(writes=[pref], heads=[pref]),
                    ),
                    dict(
                        turn_id="T2",
                        session_id="S2",
                        offset_seconds=120,
                        text="给我推荐手机。",
                        tags=["activation", "restart"],
                        reply_to=None,
                        trusted_delete=None,
                        assistant=None,
                        expected=expected(heads=[pref], activated=[pref], actions=["recommend"]),
                    ),
                ],
            }
        ],
    }


def client(*, bad_first=False, seen=None):
    def complete(messages):
        payload = json.loads(messages[-1]["content"])
        if seen is not None:
            seen.append(payload)
        turn = payload["turns"][-1]
        q = {"turn_id": turn["turn_id"], "quote": turn["content"]}
        raw = envelope()
        if turn["turn_id"] == "T1":
            if bad_first:
                return ModelReply("bad-json", "fixed", None, {"total_tokens": 7}, 0.01)
            raw["intents"] = ["express_preference"]
            raw["candidate"]["assertions"] = [
                dict(
                    subject={"kind": "self", "references": []},
                    category="phone",
                    entity="小米",
                    relation="prefer",
                    scope="long_term",
                    evidence=[q],
                )
            ]
        else:
            raw["intents"] = ["recommend"]
            raw["candidate"]["request"].update(
                is_recommendation=True,
                beneficiary={"kind": "self", "references": []},
                category="phone",
                evidence=[q],
            )
        return ModelReply(
            json.dumps(raw, ensure_ascii=False), "fixed", None, {"total_tokens": 7}, 0.01
        )

    return complete


def test_real_commit_restart_and_activation_with_no_labels_sent_to_model(tmp_path):
    from memrecagent.memory_quality_evaluation import evaluate_memory_quality

    seen = []
    report = evaluate_memory_quality(
        suite(), complete=client(seen=seen), output_dir=tmp_path / "run"
    )
    assert report["n_steps"] == 2 and report["strict_steps"] == 2
    assert report["writes"]["precision"] == 1 and report["writes"]["recall"] == 1
    assert report["writes"]["actual"] == 1
    assert report["n_episodes"] == report["strict_episodes"] == 1
    assert report["provider_attempts"] == 2 and report["reported_total_tokens"] == 14
    assert report["inputs_unchanged"] is True
    assert all(
        "expected" not in json.dumps(payload) and "label_source" not in payload for payload in seen
    )
    assert [t["turn_id"] for t in seen[-1]["turns"]] == ["T2"]
    assert (tmp_path / "run/source/memory_interaction.py").is_file()
    assert len((tmp_path / "run/calls.jsonl").read_text().splitlines()) == 2


def test_failed_first_write_is_not_replaced_with_oracle_before_next_session(tmp_path):
    from memrecagent.memory_quality_evaluation import evaluate_memory_quality

    report = evaluate_memory_quality(
        suite(), complete=client(bad_first=True), output_dir=tmp_path / "run"
    )
    assert report["strict_steps"] == 0
    assert report["n_processing_errors"] == 1
    assert report["writes"]["recall"] == 0
    rows = [json.loads(x) for x in (tmp_path / "run/steps.jsonl").read_text().splitlines()]
    assert rows[1]["actual"]["heads"] == [] and rows[1]["actual"]["activated"] == []
    assert report["provider_attempts"] == 2


def test_provider_failure_stops_and_retains_unexecuted_denominator(tmp_path):
    from memrecagent.memory_quality_evaluation import evaluate_memory_quality

    def failed(messages):
        raise ProviderError("http", 429)

    report = evaluate_memory_quality(suite(), complete=failed, output_dir=tmp_path / "run")
    assert report["n_steps"] == 2 and report["strict_steps"] == 0
    assert report["provider_attempts"] == 1
    assert report["n_provider_errors"] == 1 and report["n_not_run"] == 1
    assert report["reported_total_tokens"] == 0 and report["attempts_without_usage"] == 1
    call = json.loads((tmp_path / "run/calls.jsonl").read_text().splitlines()[0])
    assert call["wall_seconds"] >= 0


def test_unasked_conditional_reply_is_not_sent_and_counts_as_missed(tmp_path):
    from memrecagent.memory_quality_evaluation import evaluate_memory_quality

    cases = suite()
    cases["episodes"][0]["steps"][1].update(
        session_id="S1", reply_to="T1", text="只对我本人长期保存。"
    )
    report = evaluate_memory_quality(cases, complete=client(), output_dir=tmp_path / "run")
    assert report["provider_attempts"] == 1
    assert report["n_unasked_replies"] == 1
    assert report["strict_steps"] == 1


def test_budget_reserves_both_analyst_calls_before_any_request(tmp_path):
    from memrecagent.memory_quality_evaluation import evaluate_memory_quality

    called = []
    with pytest.raises(ValueError, match="预算"):
        evaluate_memory_quality(
            suite(),
            complete=lambda x: called.append(x),
            output_dir=tmp_path / "run",
            analysis_mode="analyst",
            max_calls=1,
        )
    assert not called and not (tmp_path / "run").exists()


def test_same_template_family_cannot_cross_dev_and_test():
    from memrecagent.memory_quality_data import validate_quality_suite

    data = suite()
    other = copy.deepcopy(data["episodes"][0])
    other.update(episode_id="holdout_self", split="test")
    data["episodes"].append(other)
    with pytest.raises(ValueError):
        validate_quality_suite(data)


@pytest.mark.parametrize(
    "change",
    [
        "duplicate_id",
        "future_reply",
        "future_gold",
        "bad_scope",
        "duplicate_head",
        "secret_source",
        "reused_session",
        "bad_time",
    ],
)
def test_invalid_semantic_or_temporal_suite_fails(change):
    from memrecagent.memory_quality_data import validate_quality_suite

    data = suite()
    episode = data["episodes"][0]
    step = episode["steps"][1]
    if change == "duplicate_id":
        step["turn_id"] = "T1"
    elif change == "future_reply":
        episode["steps"][0]["reply_to"] = "T2"
    elif change == "future_gold":
        episode["steps"][0]["expected"]["revisions"] = [["phone", "小米", "prefer"]]
    elif change == "bad_scope":
        episode["steps"][0]["expected"]["other_only"] = [["phone", "小米", "prefer"]]
    elif change == "duplicate_head":
        step["expected"]["heads"].append(step["expected"]["heads"][0])
    elif change == "secret_source":
        data["synthetic"] = False
    elif change == "reused_session":
        episode["steps"].append(
            copy.deepcopy(step) | {"turn_id": "T3", "session_id": "S1", "offset_seconds": 180}
        )
    elif change == "bad_time":
        step["offset_seconds"] = 60
    with pytest.raises(ValueError):
        validate_quality_suite(data)


def test_wrong_person_and_current_scope_writes_are_explicitly_counted():
    from memrecagent.memory_quality_metrics import score_quality_step, summarize_quality

    gold = expected()
    gold["other_only"] = [["phone", "三星", "prefer"]]
    gold["current_only"] = [["phone", "小米", "prefer"]]
    actual = dict(
        action="answer",
        writes=[["phone", "三星", "prefer"], ["phone", "小米", "prefer"]],
        heads=[],
        activated=[],
        other_context=[],
        revisions=[],
        status="ok",
    )
    score = score_quality_step(gold, actual)
    report = summarize_quality([dict(episode_id="E", split="dev", scenario="attribution", **score)])
    assert report["writes"]["actual"] == 2 and report["writes"]["precision"] == 0
    assert report["wrong_attribution_records"] == 1
    assert report["wrong_scope_records"] == 1
    assert report["wrong_attribution_per_written_record"] == 0.5


def test_no_writes_is_unknown_precision_and_zero_errors_has_nonzero_upper_bound():
    from memrecagent.memory_quality_metrics import score_quality_step, summarize_quality

    gold = expected()
    actual = dict(
        action="answer",
        writes=[],
        heads=[],
        activated=[],
        other_context=[],
        revisions=[],
        status="ok",
    )
    score = score_quality_step(gold, actual)
    report = summarize_quality([dict(episode_id="E", split="test", scenario="consult", **score)])
    assert report["writes"]["precision"] is None and report["writes"]["recall"] is None
    assert report["zero_error_upper_95_per_write"] is None
    gold["writes"] = actual["writes"] = [["phone", "小米", "prefer"]]
    score = score_quality_step(gold, actual)
    report = summarize_quality([dict(episode_id="E", split="test", scenario="self", **score)])
    assert report["zero_error_upper_95_per_write"] == pytest.approx(0.95)


def test_source_change_between_calls_stops_before_second_send(tmp_path, monkeypatch):
    from memrecagent import memory_quality_evaluation as evaluation

    original = evaluation.source_fingerprints
    called = []

    def hashes():
        actual = original()
        if called:
            actual["injected-change"] = "changed"
        return actual

    complete = client()

    def callback(messages):
        called.append(1)
        return complete(messages)

    monkeypatch.setattr(evaluation, "source_fingerprints", hashes)
    report = evaluation.evaluate_memory_quality(
        suite(), complete=callback, output_dir=tmp_path / "run"
    )
    assert len(called) == 1 and report["inputs_unchanged"] is False
    assert report["n_steps"] == 2 and report["strict_steps"] == 0


def test_quality_freeze_covers_every_production_module_including_attribute_dependencies():
    from hashlib import sha256
    from pathlib import Path

    from memrecagent import memory_quality_evaluation as evaluation

    fingerprints = evaluation.source_fingerprints()
    for path in Path(evaluation.__file__).parent.glob("*.py"):
        assert fingerprints.get(path.name) == sha256(path.read_bytes()).hexdigest(), path.name


def test_per_tag_denominators_and_polarity_do_not_hide_wrong_relation():
    from memrecagent.memory_quality_metrics import score_quality_step, summarize_quality

    gold = expected(writes=[["phone", "小米", "prefer"]])
    actual = dict(
        action="answer",
        writes=[["phone", "小米", "dislike"]],
        heads=[],
        activated=[],
        other_context=[],
        revisions=[],
        status="ok",
    )
    score = score_quality_step(gold, actual)
    report = summarize_quality(
        [dict(episode_id="E", split="test", scenario="polarity", tags=["negation"], **score)]
    )
    assert report["polarity_accuracy_given_expected_entity"] == 0
    assert report["by_tag"]["negation"]["writes"]["recall"] == 0


def test_analyst_failures_keep_both_call_records_on_the_step(tmp_path):
    from memrecagent.memory_quality_evaluation import evaluate_memory_quality

    report = evaluate_memory_quality(
        suite(),
        complete=client(bad_first=True),
        analysis_mode="analyst",
        output_dir=tmp_path / "run",
    )
    assert report["provider_attempts"] == 4
    row = json.loads((tmp_path / "run/steps.jsonl").read_text().splitlines()[0])
    assert len(row["analysis_calls"]) == 2


def test_alias_scoring_keeps_relative_preference_in_ram_then_drops_on_restart(tmp_path):
    from memrecagent.memory_quality_evaluation import evaluate_memory_quality

    data = suite()
    ep = data["episodes"][0]
    ep["people"] = [{"person_id": "uncle", "aliases": ["叔叔", "我叔叔"]}]
    ep["steps"][0].update(text="我叔叔一直喜欢小米手机。", expected=expected())
    ep["steps"][0]["expected"]["other_only"] = [["phone", "小米", "prefer"]]
    ep["steps"][1].update(
        text="给叔叔推荐手机。", session_id="S1", expected=expected(actions=["recommend"])
    )
    ep["steps"][1]["expected"]["other_context"] = [["uncle", "phone", "小米", "prefer"]]
    ep["steps"].append(
        copy.deepcopy(ep["steps"][1])
        | {
            "turn_id": "T3",
            "session_id": "S2",
            "offset_seconds": 180,
            "expected": expected(actions=["recommend"]),
        }
    )

    def complete(messages):
        payload = json.loads(messages[-1]["content"])
        latest = payload["turns"][-1]

        def quote(t, text):
            return {"turn_id": t, "quote": text}

        raw = envelope()
        if latest["turn_id"] == "T1":
            raw["intents"] = ["express_preference"]
            raw["candidate"]["assertions"] = [
                dict(
                    subject={"kind": "other", "references": [quote("T1", "叔叔")]},
                    category="phone",
                    entity="小米",
                    relation="prefer",
                    scope="long_term",
                    evidence=[quote("T1", latest["content"])],
                )
            ]
        else:
            raw["intents"] = ["recommend"]
            raw["candidate"]["request"].update(
                is_recommendation=True,
                category="phone",
                beneficiary={
                    "kind": "other",
                    "references": [
                        quote(
                            "T1"
                            if any(t["turn_id"] == "T1" for t in payload["turns"])
                            else latest["turn_id"],
                            "叔叔",
                        )
                    ],
                },
                evidence=[quote(latest["turn_id"], latest["content"])],
            )
        return ModelReply(json.dumps(raw, ensure_ascii=False), "fixed", None, None, 0.0)

    report = evaluate_memory_quality(data, complete=complete, output_dir=tmp_path / "run")
    assert report["strict_steps"] == 3
    assert report["writes"]["actual"] == 0


def test_no_token_field_does_not_masquerade_as_known_zero_usage(tmp_path):
    from memrecagent.memory_quality_evaluation import evaluate_memory_quality

    normal = client()

    def unknown(messages):
        response = normal(messages)
        return ModelReply(response.content, "fixed", None, {"prompt_tokens": 5}, 0.01)

    report = evaluate_memory_quality(suite(), complete=unknown, output_dir=tmp_path / "run")
    assert report["attempts_without_total_tokens"] == 2
    assert report["reported_total_tokens"] == 0


def test_wrong_relation_does_not_hide_pollution_from_an_other_only_entity():
    from memrecagent.memory_quality_metrics import score_quality_step

    gold = expected()
    gold["other_only"] = [["phone", "三星", "prefer"]]
    actual = dict(
        action="answer",
        writes=[["phone", "三星", "exclude"]],
        heads=[],
        activated=[],
        other_context=[],
        revisions=[],
        status="ok",
    )
    assert score_quality_step(gold, actual)["wrong_attribution_records"] == 1


def test_actual_clarification_is_observed_before_conditioned_confirmation(tmp_path):
    from memrecagent.memory_quality_evaluation import evaluate_memory_quality

    data = suite()
    pref = ["phone", "小米", "prefer"]
    second = data["episodes"][0]["steps"][1]
    second.update(
        session_id="S1",
        text="我现在不喜欢小米手机了。",
        expected=expected(heads=[pref], actions=["clarify"]),
    )
    third = copy.deepcopy(second) | {
        "turn_id": "T3",
        "offset_seconds": 180,
        "text": "只对我本次有效，不长期保存。",
        "reply_to": "T2",
        "expected": expected(heads=[pref]),
    }
    data["episodes"][0]["steps"].append(third)
    original = client()

    def complete(messages):
        payload = json.loads(messages[-1]["content"])
        turn = payload["turns"][-1]
        if turn["turn_id"] == "T1":
            return original(messages)
        raw = envelope()
        raw["intents"] = ["express_preference"]
        quote = {"turn_id": turn["turn_id"], "quote": turn["content"]}
        if turn["turn_id"] == "T2":
            raw["candidate"]["assertions"] = [
                dict(
                    subject={"kind": "self", "references": []},
                    category="phone",
                    entity="小米",
                    relation="dislike",
                    scope="unclear",
                    evidence=[quote],
                )
            ]
        else:
            assert payload["pending_question"]["turn_id"] == "Q-T2"
            raw["confirmation"] = dict(
                question_turn_id="Q-T2",
                outcome="confirm",
                subject="self",
                scope="current",
                person_reference=None,
                evidence=[quote],
            )
        return ModelReply(json.dumps(raw, ensure_ascii=False), "fixed", None, None, 0.0)

    report = evaluate_memory_quality(data, complete=complete, output_dir=tmp_path / "run")
    assert report["strict_steps"] == 3
    rows = [json.loads(r) for r in (tmp_path / "run/steps.jsonl").read_text().splitlines()]
    assert "本次" in rows[1]["actual"]["clarification"]
