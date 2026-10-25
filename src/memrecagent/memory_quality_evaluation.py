"""冻结合成语言评测；真实host提交/重启/删除，不从标准答案恢复状态。"""

import json
from collections.abc import Callable
from dataclasses import asdict
from datetime import UTC, datetime, timedelta
from functools import partial
from hashlib import sha256
from pathlib import Path
from time import perf_counter

from memrecagent.deepseek_client import ModelReply, ProviderError
from memrecagent.memory_analyst import AnalystError, _metadata
from memrecagent.memory_commit import CommitError
from memrecagent.memory_dialogue import _person_name, _state
from memrecagent.memory_extraction import ExtractionError, Turn
from memrecagent.memory_interaction import ConversationMemorySession
from memrecagent.memory_quality_data import validate_quality_suite
from memrecagent.memory_quality_metrics import score_quality_step, summarize_quality
from memrecagent.memory_store import JsonlMemoryStore

BASE = datetime(2026, 10, 23, tzinfo=UTC)


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def _source_paths() -> dict[str, Path]:
    source = Path(__file__).parent
    files = {p.name: p for p in sorted(source.glob("*.py"))}
    root = source.parents[1]
    for name in ("pyproject.toml", "uv.lock", "scripts/evaluate_memory_quality.py"):
        if (root / name).is_file():
            files[Path(name).name] = root / name
    return files


def source_fingerprints() -> dict[str, str]:
    """每次调用前后检查真正使用的源码与锁文件，变化停止发送。"""
    return {name: sha256(path.read_bytes()).hexdigest() for name, path in _source_paths().items()}


def evaluate_memory_quality(
    suite: dict,
    *,
    complete: Callable[[list[dict[str, str]]], ModelReply],
    output_dir: Path,
    analysis_mode: str = "direct",
    max_calls: int | None = None,
    evaluation_kind: str = "fixed_client",
) -> dict:
    """逐故事从空库开始；全部计划轮留分母，供应商/冻结失败停止整次运行。"""
    validate_quality_suite(suite)
    suite = json.loads(_json(suite))
    if analysis_mode not in ("direct", "analyst", "analyst_blind") or evaluation_kind not in (
        "fixed_client",
        "synthetic_live",
    ):
        raise ValueError("分析或评测模式无效")
    calls_per_turn = 1 if analysis_mode == "direct" else 2
    steps = sum(len(e["steps"]) for e in suite["episodes"])
    budget = steps * calls_per_turn if max_calls is None else max_calls
    if type(budget) is not int or not calls_per_turn <= budget <= steps * calls_per_turn:
        raise ValueError("调用预算必须覆盖完整单轮且不超过套件上限")
    output_dir = Path(output_dir)
    source = Path(__file__).parent.resolve()
    if (
        output_dir.resolve() == source
        or source in output_dir.resolve().parents
        or output_dir.resolve() in source.parents
    ):
        raise ValueError("产物不能覆盖源码")
    fingerprints = source_fingerprints()
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "source").mkdir()
    for name, path in _source_paths().items():
        (output_dir / "source" / name).write_bytes(path.read_bytes())
    (output_dir / "inputs.json").write_text(_json(suite) + "\n")
    manifest = {
        "schema_version": 1,
        "evaluation_kind": evaluation_kind,
        "label_source": suite["label_source"],
        "source_sha256": fingerprints,
        "inputs_sha256": sha256(_json(suite).encode()).hexdigest(),
        "analysis_mode": analysis_mode,
        "max_provider_attempts": budget,
        "planned_steps": steps,
        "automatic_retries": 0,
        "initial_memory": "empty_per_episode",
        "assistant_source": "host_clarification_or_explicit_scripted_context",
        "frozen_before_calls": True,
    }
    (output_dir / "manifest.json").write_text(_json(manifest) + "\n")
    calls, rows = [], []
    stopped = False

    def frozen() -> bool:
        return source_fingerprints() == fingerprints

    def save_calls() -> None:
        (output_dir / "calls.jsonl").write_text("".join(_json(c) + "\n" for c in calls))

    def call(messages: list[dict[str, str]], identity: str) -> ModelReply:
        if not frozen():
            raise ProviderError("freeze_changed")
        if len(calls) >= budget:
            raise ProviderError("budget")
        record = {
            "identity": identity,
            "status": "attempted",
            "messages": json.loads(_json(messages)),
            "usage": None,
        }
        calls.append(record)
        save_calls()
        started = perf_counter()
        try:
            response = complete(messages)
            if not isinstance(response, ModelReply):
                raise ProviderError("response_protocol")
            # 单次与双阶段共用同样的返回元数据守卫。
            _metadata(response)
            record.update(
                status="ok",
                response=response.content,
                model=response.model,
                fingerprint=response.fingerprint,
                usage=response.usage,
                latency_seconds=response.latency_seconds,
            )
            if not frozen():
                raise ProviderError("freeze_changed")
            return response
        except ProviderError as error:
            record.update(status="provider_error", error_code=error.code, http_status=error.status)
            raise
        finally:
            record["wall_seconds"] = perf_counter() - started
            save_calls()

    for episode in suite["episodes"]:
        store = JsonlMemoryStore(output_dir / "stores" / episode["episode_id"])
        host, session_id, last_question = None, None, None
        age = episode["max_memory_age_seconds"]
        age = timedelta(seconds=age) if age is not None else None
        aliases = {
            alias: person["person_id"]
            for person in episode["people"]
            for alias in person["aliases"]
        }
        for step in episode["steps"]:
            at = BASE + timedelta(seconds=step["offset_seconds"])
            as_of = at + timedelta(seconds=1)
            if session_id != step["session_id"]:
                session_id, last_question = step["session_id"], None
                host = ConversationMemorySession(
                    store, user_id="synthetic-user", session_id=session_id, max_memory_age=age
                )
            before = {m.memory_id: m for m in store.list_memories("synthetic-user", as_of=as_of)}
            actual = dict(
                action=None,
                writes=[],
                heads=[],
                activated=[],
                other_context=[],
                revisions=[],
                status="not_run",
            )
            error_code, preview, prepared = None, None, False
            if not stopped:
                if step["reply_to"] is not None and last_question != step["reply_to"]:
                    actual["status"] = "unasked_reply"
                else:
                    try:
                        if not frozen():
                            raise ProviderError("freeze_changed")
                        if len(calls) + calls_per_turn > budget:
                            raise ProviderError("budget")
                        if step["trusted_delete"] is not None:
                            store.delete_entity(
                                "synthetic-user",
                                category="phone",
                                entity=step["trusted_delete"],
                                deleted_at=at - timedelta(seconds=1),
                            )
                            host = ConversationMemorySession(
                                store,
                                user_id="synthetic-user",
                                session_id=session_id,
                                max_memory_age=age,
                            )
                            last_question = None
                        turn = Turn(step["turn_id"], "user", step["text"], at, at)
                        prepared = True
                        identity = episode["episode_id"] + ":" + step["turn_id"]
                        preview = host.prepare_user(
                            turn,
                            complete=partial(call, identity=identity),
                            as_of=as_of,
                            analysis_mode=analysis_mode,
                        )
                        actual["action"] = preview.response_action
                        last_question = None
                        if preview.response_action == "clarify":
                            question = preview.memory_preview.proposal.decision.clarification
                            actual["clarification"] = question
                            host.observe_assistant(
                                Turn(
                                    "Q-" + step["turn_id"],
                                    "assistant",
                                    question,
                                    at + timedelta(seconds=2),
                                    at + timedelta(seconds=2),
                                )
                            )
                            last_question = step["turn_id"]
                        else:
                            host.commit(preview.ticket_id, as_of=as_of)
                        script = step["assistant"]
                        if script is not None and script["on_action"] == preview.response_action:
                            host.observe_assistant(
                                Turn(
                                    "A-" + step["turn_id"],
                                    "assistant",
                                    script["text"],
                                    at + timedelta(seconds=2),
                                    at + timedelta(seconds=2),
                                )
                            )
                        actual["activated"] = _state(preview.context_memories)
                        proposal = preview.memory_preview.proposal
                        if preview.response_action == "recommend":
                            actual["other_context"] = sorted(
                                [
                                    [
                                        aliases.get(
                                            _person_name(
                                                c.beneficiary_id, proposal.people, host.turns
                                            ),
                                            "unknown_person",
                                        ),
                                        c.category,
                                        c.entity,
                                        c.relation,
                                    ]
                                    for c in proposal.decision.context_preferences
                                    if c.beneficiary_id not in (None, "self")
                                ]
                            )
                        actual["status"] = "ok"
                    except ProviderError as error:
                        error_code = error.code
                        actual["status"] = (
                            "preflight_error"
                            if error.code in ("freeze_changed", "budget")
                            else "provider_error"
                        )
                        stopped = True
                    except AnalystError as error:
                        error_code = error.code
                        provider = calls[-1] if calls else {}
                        if provider.get("status") == "provider_error":
                            actual["status"] = (
                                "preflight_error"
                                if provider.get("error_code") in ("freeze_changed", "budget")
                                else "provider_error"
                            )
                            stopped = True
                        else:
                            actual["status"] = "processing_error"
                    except (ExtractionError, CommitError) as error:
                        actual["status"], error_code = "processing_error", error.code
            visible = store.list_memories("synthetic-user", as_of=as_of)
            written = [m for m in visible if m.memory_id not in before]
            actual["writes"] = _state(written)
            actual["heads"] = _state(
                store.active_memories("synthetic-user", as_of=as_of, max_memory_age=age)
            )
            actual["revisions"] = _state(
                [
                    m
                    for m in written
                    if m.supersedes in before
                    and (before[m.supersedes].category, before[m.supersedes].entity)
                    == (m.category, m.entity)
                ]
            )
            scored = score_quality_step(step["expected"], actual)
            rows.append(
                dict(
                    episode_id=episode["episode_id"],
                    turn_id=step["turn_id"],
                    split=episode["split"],
                    scenario=episode["scenario"],
                    tags=step["tags"],
                    actual=actual,
                    error_code=error_code,
                    state_token=store.snapshot_token("synthetic-user"),
                    analysis_calls=[asdict(c) for c in host.last_analysis_calls]
                    if prepared
                    else [],
                    **scored,
                )
            )
            (output_dir / "steps.jsonl").write_text("".join(_json(r) + "\n" for r in rows))
    save_calls()
    report = summarize_quality(rows) | {
        "evaluation_kind": evaluation_kind,
        "analysis_mode": analysis_mode,
        "provider_attempts": len(calls),
        "max_provider_attempts": budget,
        "reported_total_tokens": sum((c["usage"] or {}).get("total_tokens", 0) for c in calls),
        "attempts_without_usage": sum(c["usage"] is None for c in calls),
        "attempts_without_total_tokens": sum(
            "total_tokens" not in (c["usage"] or {}) for c in calls
        ),
        "attempt_wall_seconds": sum(c["wall_seconds"] for c in calls),
        "reported_token_fields": {
            field: {
                "sum": sum((c["usage"] or {}).get(field, 0) for c in calls),
                "attempts_missing_field": sum(field not in (c["usage"] or {}) for c in calls),
            }
            for field in (
                "prompt_tokens",
                "completion_tokens",
                "total_tokens",
                "prompt_cache_hit_tokens",
                "prompt_cache_miss_tokens",
            )
        },
        "inputs_unchanged": frozen(),
        "cost_limit": "attempts include failed/unknown transport; missing usage is unknown",
    }
    (output_dir / "report.json").write_text(_json(report) + "\n")
    return report
