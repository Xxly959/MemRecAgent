"""对话运行器必须使用实际记忆和实际出现的回复，失败不丢分母。"""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from memrecagent.deepseek_client import ModelReply, ProviderError
from memrecagent.memory_dialogue import evaluate_dialogues, load_dialogues

FIXTURE = Path(__file__).parent / "fixtures/memory_dialogue_cases.json"


def model(raw):
    return ModelReply(
        json.dumps(raw, ensure_ascii=False),
        "deepseek-flash",
        "synthetic",
        {"total_tokens": 10},
        0.1,
    )


def empty():
    return {
        "request": {
            "is_recommendation": False,
            "beneficiary": {"kind": "unknown", "references": []},
            "category": None,
            "requested_entity": None,
            "change_scope": "request",
            "evidence": [],
        },
        "assertions": [],
        "unresolved": [],
    }


def test_fixed_input_has_eighteen_steps_and_only_two_conditional_replies():
    cases = load_dialogues(FIXTURE)
    assert len(cases) == 3
    assert sum(len(e["steps"]) for e in cases) == 18
    assert sum(s["reply"] is not None for e in cases for s in e["steps"]) == 2


def test_unasked_reply_and_answers_never_enter_model_messages(tmp_path):
    cases = load_dialogues(FIXTURE)
    calls = []

    def complete(messages):
        calls.append(messages)
        return model(empty())

    report = evaluate_dialogues(cases, complete=complete, output_dir=tmp_path / "run")
    assert report["base_steps"] == 18 and report["model_calls"] == 18
    assert report["confirmation_calls"] == 0
    assert report["strict_steps"] < 18
    text = json.dumps(calls, ensure_ascii=False)
    assert "只用于这次，我自己的长期排除先不改。" not in text
    assert "是我哥喜欢，这次是给我哥挑，不是给我。" not in text
    assert "allowed_writes" not in text and "max_memory_age_days" not in text


def test_provider_error_stops_but_keeps_all_eighteen_rows(tmp_path):
    def complete(messages):
        raise ProviderError("http", 401)

    report = evaluate_dialogues(
        load_dialogues(FIXTURE), complete=complete, output_dir=tmp_path / "run"
    )
    assert report["model_calls"] == 1 and report["not_run"] == 17 and report["base_steps"] == 18
    assert report["provider_errors"] == 1 and report["strict_steps"] == 0
    rows = [json.loads(s) for s in (tmp_path / "run/steps.jsonl").read_text().splitlines()]
    assert len(rows) == 18


def test_actual_wrong_write_persists_in_next_model_input_instead_of_oracle_reset(tmp_path):
    calls = []

    def complete(messages):
        payload = json.loads(messages[1]["content"])
        calls.append(payload)
        if len(calls) != 1:
            return model(empty())
        turn = payload["turns"][-1]
        result = empty()
        result["assertions"] = [
            {
                "subject": {"kind": "self", "references": []},
                "category": "phone",
                "entity": "小米",
                "relation": "prefer",
                "scope": "long_term",
                "evidence": [{"turn_id": turn["turn_id"], "start": 0, "end": len(turn["content"])}],
            }
        ]
        return model(result)

    report = evaluate_dialogues(
        load_dialogues(FIXTURE), complete=complete, output_dir=tmp_path / "run"
    )
    assert calls[1]["own_memories"] == [
        {"category": "phone", "entity": "小米", "relation": "prefer"}
    ]
    assert report["wrong_written_records"] >= 1


def test_output_directory_and_old_fixture_are_not_overwritten(tmp_path):
    out = tmp_path / "run"
    out.mkdir()
    (out / "keep").write_text("keep")
    with pytest.raises(FileExistsError):
        evaluate_dialogues(
            load_dialogues(FIXTURE), complete=lambda _: pytest.fail("不得调用"), output_dir=out
        )
    assert (out / "keep").read_text() == "keep"


def faithful_fake(messages):
    """手写语义响应只验证运行器；不读取expected，也不作为模型效果。"""
    payload = json.loads(messages[1]["content"])
    if "reply" in payload:
        reply = payload["reply"]
        text = reply["content"]
        other = text.startswith("是我哥")
        person = None
        if other:
            start = text.index("我哥")
            person = {"turn_id": reply["turn_id"], "start": start, "end": start + 2}
        return model(
            {
                "outcome": "confirm",
                "subject": "other" if other else "self",
                "scope": "current",
                "person_reference": person,
                "evidence": [{"turn_id": reply["turn_id"], "start": 0, "end": len(text)}],
            }
        )
    turn = payload["turns"][-1]
    text = turn["content"]
    result = empty()
    evidence = [{"turn_id": turn["turn_id"], "start": 0, "end": len(text)}]

    def assertion(entity, relation, scope="long_term", who="self", anchor=None):
        refs = []
        if anchor:
            source = next(t for t in payload["turns"] if anchor in t["content"])
            start = source["content"].index(anchor)
            refs = [{"turn_id": source["turn_id"], "start": start, "end": start + len(anchor)}]
        result["assertions"] = [
            {
                "subject": {"kind": who, "references": refs},
                "category": "phone",
                "entity": entity,
                "relation": relation,
                "scope": scope,
                "evidence": evidence,
            }
        ]
        return refs

    def request(entity=None, who="self", refs=()):
        result["request"].update(
            is_recommendation=True,
            beneficiary={"kind": who, "references": list(refs)},
            category="phone",
            requested_entity=entity,
            evidence=evidence,
        )

    if text in (
        "以后推荐手机不要推荐小米，这是我自己的长期偏好。",
        "以后给我推荐手机不要推荐小米，长期记住。",
    ):
        assertion("小米", "exclude")
    elif text == "我爸喜欢小米，这次给他挑一部手机。":
        refs = assertion("小米", "prefer", who="other", anchor="我爸")
        request(who="other", refs=refs)
    elif text == "这次给我看看小米手机。":
        request("小米")
    elif text == "我现在喜欢小米了，但还没决定仅限这次还是长期。":
        assertion("小米", "prefer", scope="unclear")
    elif text == "永久取消我对小米的排除，也不是说我特别喜欢。":
        assertion("小米", "neutral")
    elif text == "我爸和我哥都在换手机，他们的想法还没确定。":
        pass
    elif text == "给我看看三星手机。":
        request("三星")
    elif text == "他喜欢小米，给他推荐一部手机。":
        assertion("小米", "prefer", who="unknown")
        request(who="unknown")
    elif text == "以后给我推荐手机也不要推荐苹果，长期记住。":
        assertion("苹果", "exclude")
    elif text in ("我长期喜欢三星手机。", "这是删除后的新表达：以后我喜欢三星手机，长期保存。"):
        assertion("三星", "prefer")
    elif text == "这次给我看的手机别选小米，只限本次。":
        assertion("小米", "exclude", scope="current")
        request()
    elif text == "长期取消我喜欢三星的态度，不代表讨厌三星。":
        assertion("三星", "neutral")
    else:
        request()
    return model(result)


def test_complete_fake_run_uses_real_state_confirmation_reset_expiry_and_delete(tmp_path):
    cases = load_dialogues(FIXTURE)
    report = evaluate_dialogues(cases, complete=faithful_fake, output_dir=tmp_path / "run")
    assert report["model_calls"] == 20 and report["confirmation_calls"] == 2
    assert report["strict_steps"] == 18 and report["strict_episodes"] == 3
    assert report["wrong_written_records"] == report["missing_updates"] == 0
    calls = [json.loads(s) for s in (tmp_path / "run/calls.jsonl").read_text().splitlines()]
    payloads = [json.loads(c["messages"][1]["content"]) for c in calls if c["kind"] == "extraction"]
    c3 = payloads[14]
    assert c3["own_memories"] == [] and c3["turns"][-1]["turn_id"] == "T3"
    reset = payloads[11]
    assert len(reset["turns"]) == 1 and reset["turns"][0]["turn_id"] == "T6"
    rows = [json.loads(s) for s in (tmp_path / "run/steps.jsonl").read_text().splitlines()]
    assert rows[14]["heads"] == [["phone", "三星", "prefer"]] and rows[14]["active"] == []
    assert rows[16]["heads"] == [] and rows[17]["heads"] == [["phone", "三星", "prefer"]]


def test_unclear_confirmation_never_commits_or_rewrites_the_future_reference(tmp_path):
    def complete(messages):
        payload = json.loads(messages[1]["content"])
        if "reply" not in payload:
            return faithful_fake(messages)
        reply = payload["reply"]
        return model(
            {
                "outcome": "unclear",
                "subject": None,
                "scope": None,
                "person_reference": None,
                "evidence": [
                    {"turn_id": reply["turn_id"], "start": 0, "end": len(reply["content"])}
                ],
            }
        )

    report = evaluate_dialogues(
        load_dialogues(FIXTURE), complete=complete, output_dir=tmp_path / "run"
    )
    assert report["confirmation_calls"] == 2 and report["confirmation_correct"] == 0
    rows = [json.loads(s) for s in (tmp_path / "run/steps.jsonl").read_text().splitlines()]
    assert rows[3]["final_action"] == "clarify" and rows[3]["written"] == []
    assert rows[3]["strict"] is False and rows[3]["reply_seen"] is True


def test_default_cli_does_not_read_key_or_create_run(tmp_path):
    env = os.environ.copy()
    env.pop("DEEPSEEK_API_KEY", None)
    result = subprocess.run(
        [
            sys.executable,
            "scripts/evaluate_memory_dialogues.py",
            "--output-dir",
            str(tmp_path / "run"),
        ],
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {
        "mode": "plan",
        "suite": "development",
        "episodes": 3,
        "base_steps": 18,
        "max_model_calls": 20,
        "api_calls": 0,
    }
    assert not (tmp_path / "run").exists()


def test_live_cli_without_key_does_not_initialize_experiment(tmp_path):
    env = os.environ.copy()
    env.pop("DEEPSEEK_API_KEY", None)
    result = subprocess.run(
        [
            sys.executable,
            "scripts/evaluate_memory_dialogues.py",
            "--live",
            "--output-dir",
            str(tmp_path / "run"),
        ],
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == 2
    assert "DEEPSEEK_API_KEY" in result.stderr
    assert not (tmp_path / "run").exists()


def test_library_evaluation_works_when_installed_without_repository_scripts(tmp_path):
    package_root = tmp_path / "installed"
    shutil.copytree(
        Path("src/memrecagent"),
        package_root / "memrecagent",
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    code = """import sys,json
sys.path.insert(0,sys.argv[1])
from pathlib import Path
from memrecagent.deepseek_client import ModelReply
from memrecagent.memory_dialogue import load_dialogues,evaluate_dialogues
value={"request":{
    "is_recommendation":False,"beneficiary":{"kind":"unknown","references":[]},
    "category":None,"requested_entity":None,"change_scope":"request","evidence":[]},
    "assertions":[],"unresolved":[]}
report=evaluate_dialogues(
    load_dialogues(Path(sys.argv[2])),
    complete=lambda _:ModelReply(json.dumps(value),"fake",None,None,0.0),
    output_dir=Path(sys.argv[3]))
print(report["base_steps"])
"""
    result = subprocess.run(
        [
            sys.executable,
            "-S",
            "-c",
            code,
            str(package_root),
            str(FIXTURE.resolve()),
            str(tmp_path / "run"),
        ],
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "18"


@pytest.mark.parametrize("anchor_word", ["这次", "喜欢"])
def test_wrong_person_anchor_cannot_receive_confirmation_credit(tmp_path, anchor_word):
    def complete(messages):
        response = faithful_fake(messages)
        payload = json.loads(messages[1]["content"])
        if "reply" in payload and payload["reply"]["content"].startswith("是我哥"):
            value = json.loads(response.content)
            reply = payload["reply"]
            start = reply["content"].index(anchor_word)
            value["person_reference"] = {
                "turn_id": reply["turn_id"],
                "start": start,
                "end": start + 2,
            }
            return model(value)
        return response

    report = evaluate_dialogues(
        load_dialogues(FIXTURE), complete=complete, output_dir=tmp_path / "run"
    )
    assert report["strict_steps"] < 18 and report["confirmation_correct"] == 1


def test_wrong_category_is_counted_as_wrong_persistent_write(tmp_path):
    calls = 0

    def complete(messages):
        nonlocal calls
        calls += 1
        response = faithful_fake(messages)
        if calls == 1:
            value = json.loads(response.content)
            value["assertions"][0]["category"] = "laptop"
            return model(value)
        return response

    report = evaluate_dialogues(
        load_dialogues(FIXTURE), complete=complete, output_dir=tmp_path / "run"
    )
    rows = [json.loads(s) for s in (tmp_path / "run/steps.jsonl").read_text().splitlines()]
    assert rows[0]["strict"] is False and rows[0]["wrong_written_records"] == 1
    assert report["wrong_written_records"] >= 1


def test_provider_protocol_error_also_stops_remaining_requests(tmp_path):
    def complete(messages):
        raise ProviderError("response_protocol")

    report = evaluate_dialogues(
        load_dialogues(FIXTURE), complete=complete, output_dir=tmp_path / "run"
    )
    assert report["model_calls"] == 1 and report["not_run"] == 17
