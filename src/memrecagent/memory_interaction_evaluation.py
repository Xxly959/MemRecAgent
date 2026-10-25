"""普通双方语境中的渐进偏好验收；脚本语境与真实模型响应分别留痕。"""

import json
import math
import platform
import re
from collections.abc import Callable
from dataclasses import asdict
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from pathlib import Path
from statistics import median

from memrecagent.deepseek_client import ModelReply, ProviderError
from memrecagent.memory_analyst import ANALYST_PROMPT, AnalystError
from memrecagent.memory_commit import CommitError
from memrecagent.memory_dialogue import (
    _candidate_semantics,
    _person_name,
    _state,
    _validate_semantic,
)
from memrecagent.memory_extraction import ExtractionError, Turn, unique_object
from memrecagent.memory_interaction import ConversationMemorySession, InteractionPreview
from memrecagent.memory_interaction_extraction import (
    INTENTS,
    INTERACTION_PROMPT,
    InteractionCandidate,
    decode_interaction,
)
from memrecagent.memory_store import JsonlMemoryStore

BASE = datetime(2026, 10, 21, tzinfo=UTC)
MAX_CALLS = 20
_ACTIONS = {"continue", "answer", "compare", "recommend", "clarify", "feedback_target_required"}
_SCOPE_QUESTIONS = {
    "这次偏好只适用于本次，还是需要长期保存？",
    "这次表达只适用于本次推荐，还是要长期修改之前的偏好？",
}


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def _hash(value: object) -> str:
    return sha256(_json(value).encode()).hexdigest()


def _text(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip()) and len(value) <= 4000


def _validate_clues(rows: object) -> None:
    if not isinstance(rows, list) or len(rows) > 32:
        raise ValueError("线索参考无效")
    for row in rows:
        if not isinstance(row, dict) or row.keys() != {"subject", "person", "category", "entity"}:
            raise ValueError("线索字段无效")
        _validate_semantic(
            {
                "request": {"is_recommendation": False},
                "assertions": [dict(row, relation="prefer", scope="current")],
            }
        )


def _validate_expected(expected: object) -> None:
    if not isinstance(expected, dict) or expected.keys() != {
        "intents",
        "topics",
        "assertions",
        "clues",
        "request",
        "unresolved",
        "confirmation",
        "actions",
        "heads",
        "context",
        "context_people",
        "writes",
        "working_clues",
    }:
        raise ValueError("交互参考字段无效")
    if (
        not isinstance(expected["intents"], list)
        or not expected["intents"]
        or any(i not in INTENTS for i in expected["intents"])
        or len(set(expected["intents"])) != len(expected["intents"])
        or not isinstance(expected["actions"], list)
        or not expected["actions"]
        or any(a not in _ACTIONS for a in expected["actions"])
        or not isinstance(expected["topics"], list)
        or any(not _text(t) for t in expected["topics"])
        or not isinstance(expected["unresolved"], list)
        or any(t not in {"subject", "scope", "entity", "relation"} for t in expected["unresolved"])
    ):
        raise ValueError("意图或动作参考无效")
    _validate_semantic({"request": expected["request"], "assertions": expected["assertions"]})
    _validate_clues(expected["clues"])
    _validate_clues(expected["working_clues"])
    if expected["request"]["is_recommendation"] != ("recommend" in expected["intents"]):
        raise ValueError("参考意图与请求矛盾")
    if expected["assertions"] and "express_preference" not in expected["intents"]:
        raise ValueError("参考态度缺少表达意图")
    for name in ("heads", "context", "writes", "context_people"):
        rows = expected[name]
        length = 4 if name == "context_people" else 3
        if not isinstance(rows, list) or any(
            not isinstance(row, list)
            or len(row) != length
            or any(not _text(cell) for cell in row)
            or row[-1] not in {"prefer", "dislike", "exclude", "neutral"}
            or row[-3] != "phone"
            for row in rows
        ):
            raise ValueError("状态或激活参考无效")
    confirmation = expected["confirmation"]
    if confirmation is not None and (
        not isinstance(confirmation, dict)
        or confirmation.keys() != {"outcome", "subject", "scope", "person"}
        or confirmation["outcome"] != "confirm"
        or confirmation["subject"] != "self"
        or confirmation["scope"] not in {"current", "long_term"}
        or confirmation["person"] is not None
    ):
        raise ValueError("当前单范围确认参考无效")


def _validate(cases: object) -> None:
    if not isinstance(cases, (tuple, list)) or len(cases) != 3:
        raise ValueError("需要三段冻结合成故事")
    identities, replies = set(), 0
    for episode in cases:
        if not isinstance(episode, dict) or episode.keys() != {"episode_id", "steps"}:
            raise ValueError("故事字段无效")
        identity, steps = episode["episode_id"], episode["steps"]
        if (
            not isinstance(identity, str)
            or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", identity)
            or identity in identities
            or not isinstance(steps, list)
            or len(steps) != 6
        ):
            raise ValueError("故事ID或轮次数无效")
        identities.add(identity)
        previous, turn_ids = -1, set()
        for step in steps:
            if not isinstance(step, dict) or step.keys() != {
                "turn_id",
                "offset_seconds",
                "text",
                "expected",
                "assistant",
                "reply",
                "delete_entity",
            }:
                raise ValueError("轮次字段无效")
            if (
                not _text(step["turn_id"])
                or step["turn_id"] in turn_ids
                or type(step["offset_seconds"]) is not int
                or not previous + 3 < step["offset_seconds"] <= 31536000
                or not _text(step["text"])
                or (step["delete_entity"] is not None and not _text(step["delete_entity"]))
            ):
                raise ValueError("轮次时间或内容无效")
            turn_ids.add(step["turn_id"])
            previous = step["offset_seconds"]
            _validate_expected(step["expected"])
            script = step["assistant"]
            if script is not None and (
                not isinstance(script, dict)
                or script.keys() != {"on_action", "text", "source"}
                or script["source"] != "scripted_assistant_context"
                or script["on_action"] not in _ACTIONS - {"clarify"}
                or not _text(script["text"])
            ):
                raise ValueError("脚本助手来源或动作无效")
            reply = step["reply"]
            if reply is not None:
                if (
                    not isinstance(reply, dict)
                    or reply.keys()
                    != {"turn_id", "offset_seconds", "text", "question_kind", "expected"}
                    or not _text(reply["turn_id"])
                    or reply["turn_id"] in turn_ids
                    or type(reply["offset_seconds"]) is not int
                    or not previous + 3 < reply["offset_seconds"] <= 31536000
                    or not _text(reply["text"])
                    or reply["question_kind"] != "single_scope"
                ):
                    raise ValueError("条件回复无效")
                _validate_expected(reply["expected"])
                if (
                    step["expected"]["actions"] != ["clarify"]
                    or reply["expected"]["confirmation"] is None
                ):
                    raise ValueError("条件回复必须对应参考澄清")
                turn_ids.add(reply["turn_id"])
                previous = reply["offset_seconds"]
                replies += 1
    if replies != 2:
        raise ValueError("冻结集需要两条条件回复")


def load_interactions(path: Path) -> tuple[dict, ...]:
    """只读取明确来源的固定合成集；所有参考在任何模型调用前检查。"""
    value = json.loads(Path(path).read_text(), object_pairs_hook=unique_object)
    if (
        not isinstance(value, dict)
        or value.keys() != {"schema_version", "synthetic", "reference_source", "episodes"}
        or type(value["schema_version"]) is not int
        or value["schema_version"] != 1
        or value["synthetic"] is not True
        or value["reference_source"] != "assistant_authored_from_approved_rules"
    ):
        raise ValueError("需要明确标注来源的合成输入")
    _validate(value["episodes"])
    return tuple(value["episodes"])


def _clue_view(clues, turns: tuple[Turn, ...]) -> list[dict]:
    known = {turn.turn_id: turn for turn in turns}
    rows = []
    for clue in clues:
        person = None
        if clue.subject.kind == "other":
            span = clue.subject.references[0]
            person = known[span.turn_id].content[span.start : span.end].strip()
        rows.append(
            dict(
                subject=clue.subject.kind, person=person, category=clue.category, entity=clue.entity
            )
        )
    return sorted(rows, key=_json)


def _projection(analysis: InteractionCandidate, turns: tuple[Turn, ...]) -> dict:
    semantic = _candidate_semantics(analysis.candidate, turns)
    confirmation = None
    if analysis.confirmation is not None:
        choice = analysis.confirmation.choice
        person = None
        if choice.person_reference is not None:
            span = choice.person_reference
            source = next(t for t in turns if t.turn_id == span.turn_id)
            person = source.content[span.start : span.end].strip()
        confirmation = dict(
            outcome=choice.outcome, subject=choice.subject, scope=choice.scope, person=person
        )
    return dict(
        semantic,
        intents=sorted(analysis.intents),
        topics=list(analysis.topic_entities),
        clues=_clue_view(analysis.clues, turns),
        unresolved=sorted(u.field for u in analysis.candidate.unresolved),
        confirmation=confirmation,
    )


def _field_view(projection: dict, field: str) -> list:
    """按实体对齐字段，人物保留原文先行词，不压缩成统一other。"""
    rows = []
    for assertion in projection["assertions"]:
        prefix = [assertion["category"], assertion["entity"]]
        if field == "subject":
            rows.append(prefix + [assertion["subject"], assertion["person"]])
        elif field != "entity":
            rows.append(prefix + [assertion[field]])
        else:
            rows.append(prefix)
    request = projection["request"]
    if request["is_recommendation"]:
        if field == "subject":
            rows.append(["request", request["beneficiary"], request["person"]])
        elif field == "scope":
            rows.append(["request", request["change_scope"]])
        elif field == "entity":
            rows.append(["request", request["category"], request["requested_entity"]])
    return sorted(rows, key=_json)


def evaluate_interactions(
    cases: tuple[dict, ...],
    *,
    complete: Callable,
    output_dir: Path,
    analysis_mode: str = "direct",
    pre_call_check: Callable[[], None] | None = None,
) -> dict:
    """一次评测只使用独立合成存储；固定18步及2条确认参考，不恢复为参考状态。"""
    if analysis_mode not in ("direct", "analyst"):
        raise ValueError("需要direct或analyst分析模式")
    if pre_call_check is not None and not callable(pre_call_check):
        raise ValueError("调用前检查必须可调用")
    calls_per_turn = 2 if analysis_mode == "analyst" else 1
    max_calls = MAX_CALLS * calls_per_turn
    _validate(cases)
    # 冻结调用方传入的可变字典，防止回调修改未来输入或参考。
    cases = tuple(json.loads(_json(cases)))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    source = Path(__file__).parent
    files = sorted(source.glob("memory*.py")) + [source / "deepseek_client.py"]
    script = source.parents[1] / "scripts/evaluate_memory_interactions.py"
    if script.is_file():
        files.append(script)
    snapshot = output_dir / "source"
    snapshot.mkdir()
    for file in files:
        (snapshot / file.name).write_bytes(file.read_bytes())
    manifest = dict(
        created_at=datetime.now(UTC).isoformat(),
        synthetic=True,
        reference_source="assistant_authored_from_approved_rules",
        cases_sha256=_hash(cases),
        prompt_sha256=_hash(INTERACTION_PROMPT),
        analyst_prompt_sha256=_hash(ANALYST_PROMPT),
        analysis_mode=analysis_mode,
        calls_per_turn=calls_per_turn,
        dependency_sha256={
            name: sha256((source.parents[1] / name).read_bytes()).hexdigest()
            for name in ("pyproject.toml", "uv.lock")
        },
        source_sha256={p.name: sha256(p.read_bytes()).hexdigest() for p in files},
        frozen_before_calls=True,
        max_calls=max_calls,
        base_steps=18,
        conditional_replies=2,
        requested_model="deepseek-flash",
        endpoint="https://api.deepseek.com/chat/completions",
        thinking="disabled",
        temperature=0,
        max_tokens=4096,
        automatic_retries=0,
        timeout_seconds=60,
        actual_cost=None,
        python=platform.python_version(),
        platform=platform.platform(),
        scope_unresolved_accepted_when_assertion_unclear=True,
        assistant_source="scripted_assistant_context",
        user_source="scripted_synthetic_user",
    )
    (output_dir / "manifest.json").write_text(_json(manifest) + "\n")
    (output_dir / "inputs.json").write_text(_json(cases) + "\n")
    calls, rows, events, analysis_records = [], [], [], []
    stopped = False
    preflight_failure = None

    def save() -> None:
        for name, entries in (
            ("calls", calls),
            ("steps", rows),
            ("events", events),
            ("analysis-records", analysis_records),
        ):
            (output_dir / (name + ".jsonl")).write_text("".join(_json(e) + "\n" for e in entries))

    def call(messages: list[dict[str, str]], identity: str, kind: str) -> ModelReply:
        nonlocal preflight_failure
        if len(calls) >= max_calls:
            raise ProviderError("budget")
        if pre_call_check is not None:
            try:
                pre_call_check()
            except Exception as error:
                # 宿主前置拒绝不属于已发送请求，也不回显任意检查异常原文。
                preflight_failure = (
                    "freeze_changed"
                    if isinstance(error, ProviderError) and error.code == "freeze_changed"
                    else "pre_call_check"
                )
                raise ProviderError(preflight_failure) from None
        stage = "direct"
        if analysis_mode == "analyst":
            stage = "analyst" if "memory_view" in json.loads(messages[-1]["content"]) else "draft"
        row = dict(
            identity=identity,
            kind=kind,
            stage=stage,
            messages=json.loads(_json(messages)),
            request_sha256=_hash(messages),
            status="attempted",
        )
        calls.append(row)
        save()
        try:
            response = complete(messages)
            if not isinstance(response, ModelReply):
                raise ProviderError("response_protocol")
            row.update(
                status="ok",
                response=response.content,
                model=response.model,
                fingerprint=response.fingerprint,
                usage=response.usage,
                latency_seconds=response.latency_seconds,
            )
            return response
        except ProviderError as error:
            row.update(status="provider_error", error_code=error.code, http_status=error.status)
            raise
        finally:
            save()

    for episode in cases:
        identity = episode["episode_id"]
        store = JsonlMemoryStore(output_dir / "stores" / identity)

        def session(
            store: JsonlMemoryStore = store, identity: str = identity
        ) -> ConversationMemorySession:
            return ConversationMemorySession(store, user_id="synthetic-user", session_id=identity)

        host = session()
        for step in episode["steps"]:
            row = dict(episode_id=identity, turn_id=step["turn_id"], status="not_run", strict=False)
            rows.append(row)
            if stopped:
                continue
            time = BASE + timedelta(seconds=step["offset_seconds"])
            as_of = time + timedelta(seconds=1)
            if step["delete_entity"] is not None:
                removed = store.delete_entity(
                    "synthetic-user",
                    category="phone",
                    entity=step["delete_entity"],
                    deleted_at=time - timedelta(seconds=1),
                )
                events.append(
                    dict(
                        episode_id=identity,
                        kind="trusted_delete_and_workspace_reset",
                        entity=step["delete_entity"],
                        removed_records=removed,
                    )
                )
                host = session()
            before = {m.memory_id for m in store.list_memories("synthetic-user", as_of=as_of)}
            preview, actual, reply_actual = None, None, None
            reply_seen = False
            initial_action = None
            current = Turn(step["turn_id"], "user", step["text"], time, time)

            def analyze(
                turn: Turn,
                kind: str,
                at: datetime,
                *,
                host: ConversationMemorySession = host,
                identity: str = identity,
            ) -> tuple[InteractionPreview, dict]:
                nonlocal preflight_failure
                preflight_failure = None
                # 预算不足以完成一次双阶段时，不先发送半轮请求。
                if len(calls) + calls_per_turn > max_calls:
                    raise ProviderError("budget")
                events.append(
                    dict(
                        episode_id=identity,
                        kind="scripted_synthetic_user",
                        turn_id=turn.turn_id,
                        content=turn.content,
                    )
                )
                try:
                    result = host.prepare_user(
                        turn,
                        as_of=at,
                        complete=lambda messages: call(
                            messages, identity + ":" + turn.turn_id, kind
                        ),
                        analysis_mode=analysis_mode,
                    )
                finally:
                    analysis_records.extend(
                        dict(identity=identity + ":" + turn.turn_id, kind=kind, **asdict(record))
                        for record in host.last_analysis_calls
                    )
                    save()
                parsed = decode_interaction(calls[-1]["response"], turns=host.turns)
                return result, _projection(parsed, host.turns)

            try:
                preview, actual = analyze(current, "base", as_of)
                initial_action = preview.response_action
                if preview.response_action == "clarify":
                    question = preview.memory_preview.proposal.decision.clarification
                    qtime = as_of + timedelta(seconds=1)
                    host.observe_assistant(
                        Turn("Q" + current.turn_id, "assistant", question, qtime, qtime)
                    )
                    events.append(
                        dict(
                            episode_id=identity,
                            kind="host_clarification",
                            turn_id="Q" + current.turn_id,
                            content=question,
                        )
                    )
                    scripted = step["reply"]
                    if scripted is not None and question in _SCOPE_QUESTIONS:
                        rtime = BASE + timedelta(seconds=scripted["offset_seconds"])
                        reply = Turn(scripted["turn_id"], "user", scripted["text"], rtime, rtime)
                        as_of = rtime + timedelta(seconds=1)
                        reply_seen = True
                        preview, reply_actual = analyze(reply, "confirmation", as_of)
                if preview.response_action != "clarify":
                    outcome = host.commit(preview.ticket_id, as_of=as_of)
                    row["commit_status"] = outcome.status
                ordinary = step["assistant"]
                if ordinary is not None and preview.response_action == ordinary["on_action"]:
                    at = as_of + timedelta(seconds=1)
                    assistant = Turn("A" + current.turn_id, "assistant", ordinary["text"], at, at)
                    host.observe_assistant(assistant)
                    events.append(
                        dict(
                            episode_id=identity,
                            kind=ordinary["source"],
                            turn_id=assistant.turn_id,
                            content=assistant.content,
                        )
                    )
                row["status"] = "ok"
            except ProviderError as error:
                row.update(
                    status="preflight_error" if preflight_failure else "provider_error",
                    error_code=preflight_failure or error.code,
                    http_status=error.status,
                )
                stopped = True
            except AnalystError as error:
                if preflight_failure:
                    row.update(status="preflight_error", error_code=preflight_failure)
                    stopped = True
                elif error.call is not None and error.call.status == "provider_error":
                    row.update(
                        status="provider_error",
                        error_code=error.code,
                        http_status=calls[-1].get("http_status") if calls else None,
                    )
                    stopped = True
                else:
                    row.update(status="processing_error", error_code=error.code)
            except (ExtractionError, CommitError) as error:
                row.update(status="processing_error", error_code=error.code)
            reference = step["expected"]
            # 未发生的未来回复没有授予写入权限；但确认验收仍保留固定分母。
            final = step["reply"]["expected"] if reply_seen else reference
            row["reference_branch"] = "conditional_reply" if reply_seen else "base"
            heads = _state(store.active_memories("synthetic-user", as_of=as_of))
            written = [
                m
                for m in store.list_memories("synthetic-user", as_of=as_of)
                if m.memory_id not in before
            ]
            context = _state(preview.context_memories) if preview is not None else []
            people = []
            if preview is not None and preview.response_action == "recommend":
                proposal = preview.memory_preview.proposal
                people = sorted(
                    [
                        [
                            _person_name(c.beneficiary_id, proposal.people, host.turns),
                            c.category,
                            c.entity,
                            c.relation,
                        ]
                        for c in proposal.decision.context_preferences
                        if c.beneficiary_id not in (None, "self")
                    ],
                    key=_json,
                )
            working = _clue_view(host.working_clues, host.turns)
            checks = {}
            for key in ("intents", "topics", "request", "assertions", "clues", "unresolved"):
                expected = reference[key]
                if key in {"intents", "assertions", "clues", "unresolved"}:
                    expected = sorted(expected, key=_json)
                checks[key] = actual is not None and actual[key] == expected
            for field in ("subject", "entity", "relation", "scope"):
                checks[field] = actual is not None and (
                    _field_view(actual, field) == _field_view(reference, field)
                )
            if (
                actual is not None
                and not reference["unresolved"]
                and any(a["scope"] == "unclear" for a in reference["assertions"])
            ):
                checks["unresolved"] = actual["unresolved"] in ([], ["scope"])
            wrong = sum(_state([m])[0] not in final["writes"] for m in written)
            missing = sum(item not in _state(written) for item in final["writes"])
            confirmation_ok = None
            if step["reply"] is not None:
                confirmation_ok = reply_actual is not None and all(
                    reply_actual[k]
                    == (
                        sorted(final[k], key=_json)
                        if k in {"intents", "assertions", "clues", "unresolved"}
                        else final[k]
                    )
                    for k in (
                        "intents",
                        "topics",
                        "request",
                        "assertions",
                        "clues",
                        "unresolved",
                        "confirmation",
                    )
                )
            checks.update(
                initial_action=initial_action in reference["actions"],
                final_action=preview is not None and preview.response_action in final["actions"],
                heads=heads == sorted(final["heads"]),
                activation=context == sorted(final["context"])
                and people == sorted(final["context_people"], key=_json),
                writes=wrong == 0 and missing == 0,
                working_clues=working == sorted(final["working_clues"], key=_json),
                confirmation=confirmation_ok is not False,
            )
            row.update(
                initial_action=initial_action,
                final_action=preview.response_action if preview is not None else None,
                semantic=actual,
                reply_semantic=reply_actual,
                reply_seen=reply_seen,
                heads=heads,
                context=context,
                context_people=people,
                working_clues=working,
                written=_state(written),
                wrong_written_records=wrong,
                missing_updates=missing,
                confirmation_correct=confirmation_ok,
                checks=checks,
                strict=row["status"] == "ok" and all(checks.values()),
                store_sha256=store.snapshot_token("synthetic-user"),
            )
            save()
    save()
    tokens = {}
    for entry in calls:
        for key, value in (entry.get("usage") or {}).items():
            tokens[key] = tokens.get(key, 0) + value
    latency = sorted(c["latency_seconds"] for c in calls if "latency_seconds" in c)
    report = dict(
        base_steps=18,
        confirmation_reference_count=2,
        model_calls=len(calls),
        analysis_mode=analysis_mode,
        stage_calls={
            stage: sum(c["stage"] == stage for c in calls)
            for stage in ("direct", "draft", "analyst")
        },
        attempted_steps=sum(r["status"] != "not_run" for r in rows),
        not_run=sum(r["status"] == "not_run" for r in rows),
        confirmation_calls=sum(c["kind"] == "confirmation" for c in calls),
        confirmation_turns=sum(r.get("reply_seen", False) for r in rows),
        confirmation_correct=sum(r.get("confirmation_correct") is True for r in rows),
        conditional_not_seen=sum(
            s["reply"] is not None and not r.get("reply_seen", False)
            for s, r in zip([s for e in cases for s in e["steps"]], rows, strict=True)
        ),
        strict_steps=sum(r["strict"] for r in rows),
        strict_episodes=sum(
            all(r["strict"] for r in rows if r["episode_id"] == e["episode_id"]) for e in cases
        ),
        provider_errors=sum(
            r["status"] == "provider_error" and r.get("error_code") != "budget" for r in rows
        ),
        budget_errors=sum(r.get("error_code") == "budget" for r in rows),
        processing_errors=sum(r["status"] == "processing_error" for r in rows),
        draft_processing_errors=sum(
            r["stage"] == "draft" and r["status"] == "processing_error" for r in analysis_records
        ),
        preflight_errors=sum(r["status"] == "preflight_error" for r in rows),
        wrong_written_records=sum(r.get("wrong_written_records", 0) for r in rows),
        missing_updates=sum(r.get("missing_updates", 0) for r in rows),
        scripted_assistant_observed=sum(e["kind"] == "scripted_assistant_context" for e in events),
        missed_clarification=sum(
            s["expected"]["actions"] == ["clarify"] and r.get("initial_action") != "clarify"
            for s, r in zip([s for e in cases for s in e["steps"]], rows, strict=True)
        ),
        extra_clarification=sum(
            "clarify" not in s["expected"]["actions"] and r.get("initial_action") == "clarify"
            for s, r in zip([s for e in cases for s in e["steps"]], rows, strict=True)
        ),
        tokens=tokens or None,
        usage_available_calls=sum(c.get("usage") is not None for c in calls),
        latency_p50=median(latency) if latency else None,
        latency_p95=latency[math.ceil(0.95 * len(latency)) - 1] if latency else None,
        actual_cost=None,
    )
    for metric in (
        "intents",
        "topics",
        "request",
        "assertions",
        "clues",
        "unresolved",
        "subject",
        "entity",
        "relation",
        "scope",
        "initial_action",
        "final_action",
        "heads",
        "activation",
        "writes",
        "working_clues",
    ):
        report[metric + "_correct"] = sum(r.get("checks", {}).get(metric, False) for r in rows)
    (output_dir / "report.json").write_text(_json(report) + "\n")
    return report
