"""固定三段合成故事的记忆闭环评测；仅实际出现的输入可以进入模型。"""

import json
import math
import platform
import re
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from pathlib import Path
from statistics import median

from memrecagent.deepseek_client import ModelReply, ProviderError
from memrecagent.memory_commit import CommitError, MemoryCommitSession
from memrecagent.memory_confirmation import (
    CONFIRMATION_PROMPT,
    build_confirmation_messages,
    decode_confirmation,
)
from memrecagent.memory_extraction import ExtractionError, Turn, decode_candidate, unique_object
from memrecagent.memory_extraction_evaluation import SYSTEM_PROMPT, build_messages
from memrecagent.memory_store import JsonlMemoryStore

BASE = datetime(2026, 10, 21, tzinfo=UTC)
MAX_CALLS = 20


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def _hash(value: object) -> str:
    return sha256(_json(value).encode()).hexdigest()


def _validate(cases: tuple[dict, ...] | list[dict]) -> None:
    if not isinstance(cases, (list, tuple)) or len(cases) != 3:
        raise ValueError("需要三段冻结故事")
    ids, replies = set(), 0
    for episode in cases:
        if not isinstance(episode, dict) or episode.keys() != {
            "episode_id",
            "max_memory_age_days",
            "steps",
        }:
            raise ValueError("故事字段无效")
        identity = episode["episode_id"]
        age = episode["max_memory_age_days"]
        if (
            not isinstance(identity, str)
            or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", identity)
            or identity in ids
            or (age is not None and (type(age) is not int or not 1 <= age <= 3650))
        ):
            raise ValueError("故事ID或年龄配置无效")
        ids.add(identity)
        steps = episode["steps"]
        if not isinstance(steps, list) or len(steps) != 6:
            raise ValueError("每段需要六个基础轮次")
        previous, turn_ids = -1, set()
        for step in steps:
            if not isinstance(step, dict) or step.keys() != {
                "turn_id",
                "offset_seconds",
                "text",
                "expected",
                "reply",
                "reset_session",
                "delete_entity",
                "restart",
            }:
                raise ValueError("轮次字段无效")
            if (
                not isinstance(step["turn_id"], str)
                or not step["turn_id"].strip()
                or step["turn_id"] in turn_ids
                or type(step["offset_seconds"]) is not int
                or not previous < step["offset_seconds"] <= 31536000
                or not isinstance(step["text"], str)
                or not step["text"].strip()
                or len(step["text"]) > 4000
                or type(step["reset_session"]) is not bool
                or type(step["restart"]) is not bool
                or (
                    step["delete_entity"] is not None and not isinstance(step["delete_entity"], str)
                )
            ):
                raise ValueError("轮次内容或时间无效")
            turn_ids.add(step["turn_id"])
            previous = step["offset_seconds"]
            _validate_expected(step["expected"])
            reply = step["reply"]
            if reply is not None:
                if not isinstance(reply, dict) or reply.keys() != {
                    "turn_id",
                    "offset_seconds",
                    "text",
                    "expected",
                    "confirmation",
                }:
                    raise ValueError("条件回复字段无效")
                if (
                    not isinstance(reply["turn_id"], str)
                    or not reply["turn_id"].strip()
                    or reply["turn_id"] in turn_ids
                    or type(reply["offset_seconds"]) is not int
                    or not previous + 1 < reply["offset_seconds"] <= 31536000
                    or not isinstance(reply["text"], str)
                    or not reply["text"].strip()
                ):
                    raise ValueError("条件回复无效")
                turn_ids.add(reply["turn_id"])
                previous = reply["offset_seconds"]
                _validate_expected(reply["expected"])
                confirmation = reply["confirmation"]
                if (
                    not isinstance(confirmation, dict)
                    or confirmation.keys()
                    != {
                        "outcome",
                        "subject",
                        "scope",
                        "person",
                    }
                    or confirmation["outcome"] != "confirm"
                    or confirmation["subject"] not in ("self", "other")
                    or confirmation["scope"] != "current"
                    or (confirmation["subject"] == "self" and confirmation["person"] is not None)
                    or (
                        confirmation["subject"] == "other"
                        and (
                            not isinstance(confirmation["person"], str)
                            or not confirmation["person"]
                            or confirmation["person"] not in reply["text"]
                        )
                    )
                ):
                    raise ValueError("当前冻结集确认参考无效")
                replies += 1
    if replies != 2:
        raise ValueError("冻结集需要两条条件回复")


def _validate_expected(value: object) -> None:
    required = {
        "actions",
        "heads",
        "active",
        "excluded",
        "beneficiary",
        "allowed_writes",
        "person",
    }
    if not isinstance(value, dict) or value.keys() not in (required, required | {"semantic"}):
        raise ValueError("参考标签字段无效")
    if "semantic" in value:
        _validate_semantic(value["semantic"])
    if (
        not isinstance(value["actions"], list)
        or not value["actions"]
        or any(a not in ("apply", "current", "clarify", "noop") for a in value["actions"])
        or value["beneficiary"] not in (None, "self", "other")
        or not isinstance(value["excluded"], list)
        or any(not isinstance(e, str) or not e for e in value["excluded"])
    ):
        raise ValueError("参考动作无效")
    if (
        value["beneficiary"] == "other"
        and (not isinstance(value["person"], str) or not value["person"])
    ) or (value["beneficiary"] != "other" and value["person"] is not None):
        raise ValueError("参考人物无效")
    for name in ("heads", "active", "allowed_writes"):
        if not isinstance(value[name], list) or any(
            not isinstance(row, list)
            or len(row) != 3
            or not isinstance(row[0], str)
            or not row[0]
            or not isinstance(row[1], str)
            or not row[1]
            or row[2] not in ("prefer", "dislike", "exclude", "neutral")
            for row in value[name]
        ):
            raise ValueError("参考状态无效")


def _validate_semantic(value: object) -> None:
    """新增语义参考在任何调用前核验；旧集无该字段，保留原评分口径。"""
    if not isinstance(value, dict) or value.keys() != {"request", "assertions"}:
        raise ValueError("语义参考字段无效")
    request, assertions = value["request"], value["assertions"]
    if not isinstance(request, dict) or type(request.get("is_recommendation")) is not bool:
        raise ValueError("语义请求无效")
    if not request["is_recommendation"]:
        if request.keys() != {"is_recommendation"}:
            raise ValueError("无请求的参考应归一化")
    elif (
        request.keys()
        != {
            "is_recommendation",
            "beneficiary",
            "person",
            "category",
            "requested_entity",
            "change_scope",
        }
        or request["beneficiary"] not in ("self", "other", "unknown")
        or request["change_scope"] not in ("request", "long_term", "unclear")
        or request["category"] != "phone"
        or (
            request["requested_entity"] is not None
            and (
                not isinstance(request["requested_entity"], str) or not request["requested_entity"]
            )
        )
    ):
        raise ValueError("语义请求字段无效")
    if not isinstance(assertions, list) or len(assertions) > 32:
        raise ValueError("语义态度无效")
    for row in assertions:
        if (
            not isinstance(row, dict)
            or row.keys() != {"subject", "person", "category", "entity", "relation", "scope"}
            or row["subject"] not in ("self", "other", "unknown")
            or row["category"] != "phone"
            or not isinstance(row["entity"], str)
            or not row["entity"]
            or row["relation"] not in ("prefer", "dislike", "exclude", "neutral")
            or row["scope"] not in ("long_term", "current", "unclear")
        ):
            raise ValueError("语义态度字段无效")
    people = [(row["subject"], row["person"]) for row in assertions]
    if request["is_recommendation"]:
        people.append((request["beneficiary"], request["person"]))
    if any(
        (kind == "other" and (not isinstance(person, str) or not person))
        or (kind != "other" and person is not None)
        for kind, person in people
    ):
        raise ValueError("语义人物参考无效")


def load_dialogues(path: Path) -> tuple[dict, ...]:
    """只接受明确标注的助手规则合成集，不接受真实用户对话。"""
    value = json.loads(Path(path).read_text(), object_pairs_hook=unique_object)
    if (
        not isinstance(value, dict)
        or value.keys()
        != {
            "schema_version",
            "synthetic",
            "reference_source",
            "episodes",
        }
        or type(value["schema_version"]) is not int
        or value["schema_version"] != 1
        or value["synthetic"] is not True
        or value["reference_source"] != "assistant_authored_from_approved_rules"
    ):
        raise ValueError("需要已标注来源的合成输入")
    _validate(value["episodes"])
    return tuple(value["episodes"])


def _state(rows) -> list[list[str]]:
    return sorted([[m.category, m.entity, m.relation] for m in rows])


def _person_name(person_id, people, turns) -> str | None:
    """本小样本按实际人物先行词评分，不能把父亲/哥哥都压成other。"""
    person = next((p for p in people if p.person_id == person_id), None)
    if person is None or not person.references:
        return None
    anchor = person.references[0]
    source = next((t for t in turns if t.turn_id == anchor.turn_id), None)
    return source.content[anchor.start : anchor.end].strip() if source is not None else None


def _candidate_semantics(candidate, turns: tuple[Turn, ...]) -> dict:
    """从实际候选投影语义；不从参考标签构造响应，不干预写入。"""
    known = {t.turn_id: t for t in turns}

    def person(subject):
        if subject.kind != "other":
            return None
        anchor = subject.references[0]
        return known[anchor.turn_id].content[anchor.start : anchor.end].strip()

    request = candidate.request
    request_view = {"is_recommendation": request.is_recommendation}
    if request.is_recommendation:
        request_view.update(
            beneficiary=request.beneficiary.kind,
            person=person(request.beneficiary),
            category=request.category,
            requested_entity=request.requested_entity,
            change_scope=request.change_scope,
        )
    assertions = [
        {
            "subject": a.subject.kind,
            "person": person(a.subject),
            "category": a.category,
            "entity": a.entity,
            "relation": a.relation,
            "scope": a.scope,
        }
        for a in candidate.assertions
        if a.relation != "none"
    ]
    return {"request": request_view, "assertions": sorted(assertions, key=_json)}


def _host(store: JsonlMemoryStore, identity: str, number: int, age: timedelta | None):
    return MemoryCommitSession(
        store, user_id="synthetic-user", session_id=f"{identity}-{number}", max_memory_age=age
    )


def evaluate_dialogues(cases: tuple[dict, ...], *, complete: Callable, output_dir: Path) -> dict:
    """只在本次产物根下使用合成存储；每步读取实际磁盘而非参考状态。"""
    _validate(cases)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    source = Path(__file__).parent
    files = sorted(source.glob("memory*.py")) + [source / "deepseek_client.py"]
    entrypoint = Path(__file__).resolve().parents[2] / "scripts/evaluate_memory_dialogues.py"
    if entrypoint.is_file():
        files.append(entrypoint)
    manifest = {
        "created_at": datetime.now(UTC).isoformat(),
        "data": "synthetic_only",
        "reference_source": "assistant_authored_from_approved_rules",
        "cases_sha256": _hash(cases),
        "extraction_prompt_sha256": _hash(SYSTEM_PROMPT),
        "confirmation_prompt_sha256": _hash(CONFIRMATION_PROMPT),
        "source_sha256": {p.name: sha256(p.read_bytes()).hexdigest() for p in files},
        "requested_model": "deepseek-flash",
        "thinking": "disabled",
        "temperature": 0,
        "max_tokens": 4096,
        "json_output": True,
        "automatic_retries": 0,
        "max_calls": MAX_CALLS,
        "base_steps": 18,
        "conditional_replies": 2,
        "semantic_scored_steps": sum(
            "semantic" in s["expected"] for e in cases for s in e["steps"]
        ),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "actual_cost": None,
        "endpoint": "https://api.deepseek.com/chat/completions",
        "timeout_seconds": 60,
        "max_response_bytes": 65536,
    }
    snapshot = output_dir / "source"
    snapshot.mkdir()
    for p in files:
        (snapshot / p.name).write_bytes(p.read_bytes())
    (output_dir / "manifest.json").write_text(_json(manifest) + "\n")
    (output_dir / "inputs.json").write_text(_json(cases) + "\n")
    calls, rows = [], []
    stopped = False

    def call(messages: list[dict[str, str]], kind: str, identity: str) -> ModelReply:
        if len(calls) >= MAX_CALLS:
            raise ProviderError("budget")
        entry = {
            "identity": identity,
            "kind": kind,
            "request_sha256": _hash(messages),
            "messages": messages,
            "status": "attempted",
        }
        calls.append(entry)
        try:
            response = complete(messages)
            if not isinstance(response, ModelReply):
                raise ProviderError("response_protocol")
            entry.update(
                status="ok",
                response=response.content,
                model=response.model,
                fingerprint=response.fingerprint,
                usage=response.usage,
                latency_seconds=response.latency_seconds,
            )
            return response
        except ProviderError as error:
            entry.update(status="provider_error", error_code=error.code, http_status=error.status)
            raise
        finally:
            (output_dir / "calls.jsonl").write_text("".join(_json(c) + "\n" for c in calls))

    for episode in cases:
        identity = episode["episode_id"]
        age = (
            timedelta(days=episode["max_memory_age_days"])
            if episode["max_memory_age_days"] is not None
            else None
        )
        root = output_dir / "stores" / identity
        store = JsonlMemoryStore(root)
        session_number = 1

        controller, turns, people = _host(store, identity, session_number, age), (), ()
        for step in episode["steps"]:
            row = {"episode_id": identity, "turn_id": step["turn_id"], "status": "not_run"}
            rows.append(row)
            if stopped:
                continue
            time = BASE + timedelta(seconds=step["offset_seconds"])
            as_of = time + timedelta(seconds=1)
            if step["reset_session"]:
                session_number += 1
                turns, people = (), ()
                controller = _host(store, identity, session_number, age)
            if step["delete_entity"] is not None:
                store.delete_entity(
                    "synthetic-user",
                    category="phone",
                    entity=step["delete_entity"],
                    deleted_at=time - timedelta(seconds=1),
                )
            if step["restart"]:
                store = JsonlMemoryStore(root)
                controller = _host(store, identity, session_number, age)
            before = {m.memory_id for m in store.list_memories("synthetic-user", as_of=as_of)}
            current = Turn(step["turn_id"], "user", step["text"], time, time)
            turns += (current,)
            expected = step["expected"]
            original_action, final_action, decision = None, None, None
            semantic = None
            reply_seen, confirmation_ok = False, None
            try:
                active = store.active_memories("synthetic-user", as_of=as_of, max_memory_age=age)
                response = call(
                    build_messages(turns, active), "extraction", identity + ":" + current.turn_id
                )
                candidate = decode_candidate(response.content, turns=turns)
                semantic = _candidate_semantics(candidate, turns)
                preview = controller.prepare(candidate, turns=turns, people=people, as_of=as_of)
                original_action = final_action = preview.action
                decision = preview.proposal.decision
                people = preview.proposal.people
                if preview.action == "clarify":
                    question = decision.clarification or "请明确本次偏好。"
                    turns += (Turn("Q" + current.turn_id, "assistant", question, as_of, as_of),)
                    scripted = step["reply"]
                    if scripted is not None:
                        reply_time = BASE + timedelta(seconds=scripted["offset_seconds"])
                        reply = Turn(
                            scripted["turn_id"], "user", scripted["text"], reply_time, reply_time
                        )
                        turns += (reply,)
                        reply_seen = True
                        expected = scripted["expected"]
                        as_of = reply_time + timedelta(seconds=1)
                        confirmation = call(
                            build_confirmation_messages(preview, reply=reply),
                            "confirmation",
                            identity + ":" + reply.turn_id,
                        )
                        parsed = decode_confirmation(confirmation.content, reply=reply)
                        projection = {
                            "outcome": parsed.outcome,
                            "subject": parsed.subject,
                            "scope": parsed.scope,
                            "person": (
                                reply.content[
                                    parsed.person_reference.start : parsed.person_reference.end
                                ].strip()
                                if parsed.person_reference is not None
                                else None
                            ),
                        }
                        row["confirmation"] = projection
                        confirmation_ok = projection == scripted["confirmation"]
                        if parsed.outcome == "confirm":
                            preview = controller.confirm(
                                preview.ticket_id,
                                reply=reply,
                                subject=parsed.subject,
                                scope="current" if parsed.subject == "other" else parsed.scope,
                                person_reference=parsed.person_reference,
                                as_of=as_of,
                            )
                            final_action, decision = preview.action, preview.proposal.decision
                            people = preview.proposal.people
                        elif parsed.outcome == "cancel":
                            controller.cancel(preview.ticket_id)
                            final_action, decision = "noop", None
                if final_action in ("apply", "current", "noop") and decision is not None:
                    row["commit_status"] = controller.commit(preview.ticket_id, as_of=as_of).status
                row["status"] = "ok"
            except ProviderError as error:
                row.update(status="provider_error", error_code=error.code, http_status=error.status)
                stopped = True
            except (ExtractionError, CommitError) as error:
                row.update(status="processing_error", error_code=error.code)
            history = store.list_memories("synthetic-user", as_of=as_of)
            heads = _state(store.active_memories("synthetic-user", as_of=as_of))
            active = _state(
                store.active_memories("synthetic-user", as_of=as_of, max_memory_age=age)
            )
            written = [m for m in history if m.memory_id not in before]
            excluded = list(decision.excluded_entities) if decision is not None else []
            who = (
                None
                if decision is None or decision.beneficiary_id is None
                else "self"
                if decision.beneficiary_id == "self"
                else "other"
            )
            person_name = (
                _person_name(decision.beneficiary_id, people, turns) if who == "other" else None
            )
            checks = {
                "initial_action": original_action in step["expected"]["actions"],
                "final_action": final_action in expected["actions"],
                "heads": heads == sorted(expected["heads"]),
                "active": active == sorted(expected["active"]),
                "excluded": sorted(excluded) == sorted(expected["excluded"]),
                "beneficiary": who == expected["beneficiary"],
                "person": person_name == expected["person"],
                "confirmation": confirmation_ok is not False,
            }
            if "semantic" in step["expected"]:
                reference = step["expected"]["semantic"]
                checks["semantic"] = semantic == {
                    "request": reference["request"],
                    "assertions": sorted(reference["assertions"], key=_json),
                }
            wrong = sum(
                [m.category, m.entity, m.relation] not in expected["allowed_writes"]
                for m in written
            )
            missing = sum(key not in _state(written) for key in expected["allowed_writes"])
            checks["writes"] = wrong == 0 and missing == 0
            row.update(
                initial_action=original_action,
                final_action=final_action,
                reply_seen=reply_seen,
                heads=heads,
                active=active,
                excluded=excluded,
                beneficiary=who,
                person=person_name,
                written=_state(written),
                wrong_written_records=wrong,
                missing_updates=missing,
                confirmation_correct=confirmation_ok,
                semantic=semantic,
                checks=checks,
                strict=row["status"] == "ok" and all(checks.values()),
                store_sha256=store.snapshot_token("synthetic-user"),
            )
            (output_dir / "steps.jsonl").write_text("".join(_json(r) + "\n" for r in rows))
    (output_dir / "steps.jsonl").write_text("".join(_json(r) + "\n" for r in rows))
    usage = [c["usage"] for c in calls if c.get("usage") is not None]
    tokens = {}
    for entry in usage:
        for key, value in entry.items():
            tokens[key] = tokens.get(key, 0) + value
    latency = sorted(c["latency_seconds"] for c in calls if c.get("latency_seconds") is not None)
    report = {
        "base_steps": 18,
        "attempted_steps": sum(r["status"] != "not_run" for r in rows),
        "not_run": sum(r["status"] == "not_run" for r in rows),
        "model_calls": len(calls),
        "confirmation_calls": sum(c["kind"] == "confirmation" for c in calls),
        "provider_errors": sum(r["status"] == "provider_error" for r in rows),
        "processing_errors": sum(r["status"] == "processing_error" for r in rows),
        "strict_steps": sum(r.get("strict", False) for r in rows),
        "semantic_scored_steps": sum(
            "semantic" in s["expected"] for e in cases for s in e["steps"]
        ),
        "semantic_correct": sum(r.get("checks", {}).get("semantic", False) for r in rows),
        "strict_episodes": sum(
            all(r.get("strict", False) for r in rows if r["episode_id"] == e["episode_id"])
            for e in cases
        ),
        "wrong_written_records": sum(r.get("wrong_written_records", 0) for r in rows),
        "missing_updates": sum(r.get("missing_updates", 0) for r in rows),
        "confirmation_correct": sum(r.get("confirmation_correct") is True for r in rows),
        "tokens": tokens if usage else None,
        "usage_available_calls": len(usage),
        "latency_p50": median(latency) if latency else None,
        "latency_p95": latency[math.ceil(0.95 * len(latency)) - 1] if latency else None,
        "actual_cost": None,
    }
    for metric in (
        "initial_action",
        "final_action",
        "heads",
        "active",
        "excluded",
        "beneficiary",
        "person",
        "writes",
    ):
        report[metric + "_correct"] = sum(r.get("checks", {}).get(metric, False) for r in rows)
    (output_dir / "report.json").write_text(_json(report) + "\n")
    return report
