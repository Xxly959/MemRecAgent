"""外部合成对话的实际状态重放；四组路由可运行不等于算法效应可识别。"""

import json
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from math import log2
from pathlib import Path
from statistics import fmean
from time import perf_counter

import numpy as np

from memrecagent.conversation_products import ConversationProductSession
from memrecagent.deepseek_client import ModelReply, ProviderError
from memrecagent.entity_aliases import install_entity_aliases
from memrecagent.external_derived import safe_external_id
from memrecagent.external_evaluation import _semantic
from memrecagent.memory_analyst import AnalystError, _metadata
from memrecagent.memory_commit import CommitError
from memrecagent.memory_extraction import ExtractionError, Turn
from memrecagent.memory_products import ProductEntityFact
from memrecagent.memory_quality_evaluation import _source_paths
from memrecagent.memory_store import JsonlMemoryStore
from memrecagent.native_replay import NativeReplayEnvironment
from memrecagent.product_attributes import UNITS, ProductAttributeValue
from memrecagent.product_service import ProductService
from memrecagent.result_count import resolve_result_count

BASE = datetime(2026, 1, 1, tzinfo=UTC)
GROUPS = ("G00", "G10", "G01", "G11")


def resolve_scoring_cutoffs(
    tasks: list[dict], *, k: int, scoring_mode: str = "fixed"
) -> dict[str, dict]:
    """事前绑定评分K；只看请求原文，不取模型结果、标签或商品数。"""
    if type(k) is not int or not 1 <= k <= 50 or scoring_mode not in ("fixed", "request"):
        raise ValueError("invalid scoring configuration")
    cutoffs = {}
    for task in tasks:
        count = None
        if scoring_mode == "request":
            text = task["runtime"]["request_text"]
            if not isinstance(text, str):
                raise ValueError("invalid scoring request")
            count = resolve_result_count(text)
            if count.clarification is not None:
                raise ValueError("ambiguous scoring request count")
        explicit = count is not None and count.count is not None
        cutoffs[task["episode_id"]] = dict(
            k=count.count if explicit else k,
            source="explicit_request"
            if explicit
            else "fixed"
            if scoring_mode == "fixed"
            else "fallback",
            evidence_spans=[list(span) for span in count.spans] if explicit else [],
        )
    return cutoffs


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def graded_ranking_metrics(item_ids: list[str], reference: list[dict], *, k: int) -> dict:
    """0至3级gain只用于评分；不合格商品gain为0，失败空列表保留为0。"""
    if type(k) is not int or not 1 <= k <= 50 or len(set(item_ids)) != len(item_ids):
        raise ValueError("invalid ranking")
    scores = {r["item_id"]: (2 ** r["relevance"] - 1 if r["eligible"] else 0) for r in reference}
    relevant = {r["item_id"] for r in reference if r["eligible"] and r["relevance"] >= 2}
    ordered = item_ids[:k]
    ideal = sum(v / log2(n + 2) for n, v in enumerate(sorted(scores.values(), reverse=True)[:k]))
    dcg = sum(scores.get(i, 0) / log2(n + 2) for n, i in enumerate(ordered))
    found = [n for n, i in enumerate(ordered, 1) if i in relevant]
    eligible = {r["item_id"] for r in reference if r["eligible"]}
    return dict(
        ndcg=dcg / ideal if ideal else None,
        recall=len(set(ordered) & relevant) / len(relevant) if relevant else None,
        hit_rate=float(bool(found)) if relevant else None,
        mrr=(1 / found[0] if found else 0.0) if relevant else None,
        precision=len(set(ordered) & relevant) / k if relevant else None,
        constraint_violations=sum(i not in eligible for i in ordered),
        empty=not ordered,
    )


def _catalog(task: dict, *, seed: int) -> tuple[ProductService, tuple[ProductEntityFact, ...]]:
    items = task["runtime"]["items"]
    if (
        not isinstance(items, list)
        or not 1 <= len(items) <= 5000
        or len({i["item_id"] for i in items}) != len(items)
    ):
        raise ValueError("invalid task catalog")
    if any(i.keys() - ({"item_id", "brand", "category"} | UNITS.keys()) for i in items):
        raise ValueError("catalog contains unsupported fields")
    identity = sha256(_json(items).encode()).hexdigest()
    # 此计数仅是预注册的合成公共先验，未声称来自真实点击/购买/独立用户。
    counts = {
        i["item_id"]: 1
        + int(sha256(_json(["prior", seed, i["item_id"]]).encode()).hexdigest()[:8], 16) % 100
        for i in items
    }
    service = ProductService(
        "synthetic-external-" + identity,
        int((BASE - timedelta(days=1)).timestamp() * 1000),
        counts,
        {},
        {i["item_id"]: {"title": i["brand"] + " 虚构手机"} for i in items},
        {},
        "synthetic_seeded_public_prior",
        None,
    )
    facts = tuple(
        ProductEntityFact(
            i["item_id"],
            i["category"],
            i["brand"],
            "external-catalog:" + i["item_id"],
            BASE - timedelta(days=1),
            tuple(
                ProductAttributeValue(
                    a,
                    i[a],
                    UNITS[a],
                    "external-catalog:" + i["item_id"] + ":" + a,
                    BASE - timedelta(days=1),
                )
                for a in UNITS
                if a in i
            ),
        )
        for i in items
    )
    return service, facts


def _state(store: JsonlMemoryStore, user_id: str, as_of: datetime) -> list[tuple]:
    state = [
        ("self", m.category, "brand", m.entity, m.relation)
        for m in store.active_memories(user_id, as_of=as_of)
    ]
    state += [
        ("self", m.category, "attribute", m.attribute, m.operator, m.value, m.unit, m.is_hard)
        for m in store.active_attribute_memories(user_id, as_of=as_of)
    ]
    return sorted(state, key=_json)


def family_memory_contrast(cases: list[dict], *, seed: int) -> dict:
    """近似模板属于同一场景家族，整族配对重采样，少于两族不造置信区间。"""
    grouped: dict[str, list[float]] = {}
    for case in cases:
        if case["reference_status"] != "provisional":
            continue
        off = case["groups"]["G00"]["metrics"]["ndcg"]
        on = case["groups"]["G10"]["metrics"]["ndcg"]
        if off is not None and on is not None:
            grouped.setdefault(case["template_family"], []).append(on - off)
    differences = [fmean(values) for _, values in sorted(grouped.items())]
    interval = None
    if len(differences) > 1:
        rng = np.random.default_rng(seed)
        draws = rng.choice(differences, size=(1000, len(differences)), replace=True).mean(axis=1)
        interval = np.quantile(draws, [0.025, 0.975]).tolist()
    return dict(
        delta=fmean(differences) if differences else None,
        interval=interval,
        n_families=len(differences),
        n_cases=sum(len(v) for v in grouped.values()),
        unit="whole_primary_scenario_family",
    )


def evaluate_external_replay(
    suite: dict,
    *,
    complete: Callable[[list[dict[str, str]]], ModelReply],
    output_dir: Path,
    analysis_mode: str = "analyst",
    split: str = "dev",
    max_calls: int | None = None,
    evaluation_kind: str = "fixed_client",
    k: int = 3,
    seed: int = 20261024,
    scoring_mode: str = "fixed",
    native_environment: NativeReplayEnvironment | None = None,
) -> dict:
    """只向宿主传运行文本/目录；引用只在实际处理完以后评分，绝不回填状态。"""
    suite = json.loads(_json(suite))
    if any(
        not safe_external_id(e["episode_id"])
        or any(not safe_external_id(s[k]) for s in e["steps"] for k in ("turn_id", "session_id"))
        for e in suite["episodes"]
    ):
        raise ValueError("invalid external runtime identifier")
    if (
        suite.get("schema_version") != 1
        or suite.get("synthetic") is not True
        or analysis_mode not in ("direct", "analyst", "analyst_blind")
        or split not in ("dev", "test", "all")
        or evaluation_kind not in ("fixed_client", "synthetic_live")
    ):
        raise ValueError("invalid replay configuration")
    tasks = {t["episode_id"]: t for t in suite["tasks"] if split == "all" or t["split"] == split}
    if len(tasks) != sum(split == "all" or t["split"] == split for t in suite["tasks"]):
        raise ValueError("one task per episode required")
    episodes = [e for e in suite["episodes"] if e["episode_id"] in tasks]
    if not episodes or len(episodes) != len(tasks):
        raise ValueError("task episode missing")
    scoring_cutoffs = resolve_scoring_cutoffs(list(tasks.values()), k=k, scoring_mode=scoring_mode)
    scoring_protocol = "fixed_k_v1" if scoring_mode == "fixed" else "request_count_v1"
    native = native_environment
    if native is not None:
        if not isinstance(native, NativeReplayEnvironment) or set(native.users) != set(tasks):
            raise ValueError("native profile bindings must match selected episodes")
        if any(
            sha256(p.read_bytes()).hexdigest() != native.source_sha256[n]
            for n, p in native.source_files.items()
        ):
            raise ValueError("native environment changed before replay")
        for episode in episodes:
            for step in episode["steps"]:
                native.reference_state(step["reference"]["long_term_state_after"])
        catalogs = {sid: (native.service, native.facts) for sid in tasks}
    else:
        catalogs = {sid: _catalog(t, seed=seed) for sid, t in tasks.items()}
    for e in episodes:
        t = tasks[e["episode_id"]]
        match = [s for s in e["steps"] if s["turn_id"] == t["turn_id"]]
        if len(match) != 1 or match[0]["text"] != t["runtime"]["request_text"]:
            raise ValueError("task input mismatch")
    planned = sum(len(e["steps"]) for e in episodes)
    per_turn = 1 if analysis_mode == "direct" else 2
    budget = planned * per_turn if max_calls is None else max_calls
    if (
        type(budget) is not int
        or not per_turn <= budget <= planned * per_turn
        or type(seed) is not int
        or not 0 <= seed < 2**63
        or type(k) is not int
        or not 1 <= k <= 50
    ):
        raise ValueError("invalid budget or cutoff")
    output_dir = Path(output_dir)
    source = Path(__file__).parent.resolve()
    if (
        output_dir.resolve() == source
        or source in output_dir.resolve().parents
        or output_dir.resolve() in source.parents
    ):
        raise ValueError("output cannot cover source")

    def replay_paths():
        files = _source_paths()
        extra = Path(__file__).resolve().parents[2] / "scripts/evaluate_external_replay.py"
        if extra.is_file():
            files[extra.name] = extra
        if native is not None:
            files.update(native.source_files)
        return files

    paths = replay_paths()
    fingerprints = {name: sha256(p.read_bytes()).hexdigest() for name, p in paths.items()}
    output_dir.mkdir(parents=True, exist_ok=False)
    for name, p in paths.items():
        target = output_dir / "source" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(p.read_bytes())
    inputs = _json(suite)
    (output_dir / "inputs.json").write_text(inputs)
    manifest = dict(
        source_sha256=fingerprints,
        inputs_sha256=sha256(inputs.encode()).hexdigest(),
        analysis_mode=analysis_mode,
        split=split,
        max_calls=budget,
        planned_steps=planned,
        planned_cases=len(tasks),
        evaluation_kind=evaluation_kind,
        seed=seed,
        k=k,
        scoring_mode=scoring_mode,
        scoring_protocol=scoring_protocol,
        scoring_cutoffs=scoring_cutoffs,
        automatic_retries=0,
        initial_memory="empty_per_episode",
        public_prior="existing_bundle_positive_user_counts"
        if native
        else "synthetic_seeded_counts_not_observed_behavior",
        algorithm_profile="native_ids_existing_bundle" if native else "none; actual cold fallback",
        assistant_source="actual clarification or neutral scripted acknowledgement",
    )
    (output_dir / "manifest.json").write_text(_json(manifest))
    calls, rows, cases = [], [], []
    stopped = False

    def frozen():
        try:
            return {
                name: sha256(p.read_bytes()).hexdigest() for name, p in replay_paths().items()
            } == fingerprints and (output_dir / "inputs.json").read_text() == inputs
        except OSError:
            return False

    def save(name, values):
        (output_dir / name).write_text("".join(_json(v) + "\n" for v in values))

    def call(messages):
        if native is not None:
            native.validate_messages(messages)
        if not frozen():
            raise ProviderError("freeze_changed")
        if len(calls) >= budget:
            raise ProviderError("budget")
        row = dict(
            status="attempted",
            usage=None,
            request_sha256=sha256(_json(messages).encode()).hexdigest(),
            messages=json.loads(_json(messages)),
            identity=current_identity,
            stage="analyst"
            if "memory_view" in json.loads(messages[-1]["content"])
            else "draft"
            if analysis_mode != "direct"
            else "direct",
        )
        calls.append(row)
        save("calls.jsonl", calls)
        start = perf_counter()
        try:
            reply = complete(messages)
            if not isinstance(reply, ModelReply):
                raise ProviderError("response_protocol")
            _metadata(reply)
            row.update(
                status="ok",
                usage=reply.usage,
                model=reply.model,
                fingerprint=reply.fingerprint,
                response=reply.content,
            )
            if not frozen():
                raise ProviderError("freeze_changed")
            return reply
        except ProviderError as error:
            row.update(status="provider_error", error_code=error.code, http_status=error.status)
            raise
        except Exception:
            row.update(status="provider_error", error_code="client_error")
            raise ProviderError("client_error") from None
        finally:
            row["wall_seconds"] = perf_counter() - start
            save("calls.jsonl", calls)

    for episode in episodes:
        sid = episode["episode_id"]
        task = tasks[sid]
        scoring_cutoff = scoring_cutoffs[sid]
        score_k = scoring_cutoff["k"]
        service, facts = catalogs[sid]
        store_root = output_dir / "stores" / sid
        if native is not None:
            install_entity_aliases(store_root, native.aliases.to_document())
        store = JsonlMemoryStore(store_root)
        user = native.users[sid] if native else "synthetic-external:" + sid
        episode_transport = False
        h = None
        session = None
        question = None
        for step in episode["steps"]:
            current_identity = sid + ":" + step["turn_id"]
            at = (native.base_at if native else BASE) + timedelta(seconds=step["offset_seconds"])
            as_of = at + timedelta(seconds=1)
            if session != step["session_id"] and not stopped:
                session = step["session_id"]
                question = None
                h = ConversationProductSession(
                    service,
                    store,
                    facts,
                    facts_bundle_id=service.bundle_id,
                    user_id=user,
                    session_id=session,
                    category=native.category if native else "phone",
                )
            row = dict(
                episode_id=sid,
                turn_id=step["turn_id"],
                status="not_run",
                action=None,
                error_code=None,
            )
            result = None
            if not stopped:
                if step["reply_to"] is not None and question != step["reply_to"]:
                    row["status"] = "unasked_reply"
                elif len(calls) + per_turn > budget or not frozen():
                    row["error_code"] = (
                        "budget" if len(calls) + per_turn > budget else "freeze_changed"
                    )
                    stopped = True
                else:
                    try:
                        result = h.process_user(
                            Turn(step["turn_id"], "user", step["text"], at, at),
                            complete=call,
                            as_of=as_of,
                            strategy="bpr_cold_popularity",
                            k=k,
                            analysis_mode=analysis_mode,
                        )
                        row.update(
                            status="ok",
                            action=result["response_action"],
                            new_memory_ids=result["new_memory_ids"],
                        )
                    except (AnalystError, ExtractionError, CommitError, ProviderError) as error:
                        row.update(status="processing_error", error_code=error.code)
                        if isinstance(error, ProviderError) or (
                            calls and calls[-1]["status"] == "provider_error"
                        ):
                            row["status"] = "provider_error"
                            stopped = True
                            episode_transport = (
                                native is not None and calls[-1].get("error_code") == "transport"
                            )
                    except (ValueError, OSError, TypeError):
                        row.update(status="host_error", error_code="runtime_state_error")
                        stopped = True
                    question = None
            try:
                state = _state(store, user, as_of)
                if native is not None:
                    state = [
                        (a, b, "product" if c == "brand" else c, *tail) for a, b, c, *tail in state
                    ]
                row["heads"] = [list(s) for s in state]
            except (ValueError, OSError):
                state = None
                row["heads"] = None
                row["state_read_error"] = True
                episode_transport = False
                if row["status"] != "not_run":
                    row.update(status="host_error", error_code="runtime_state_error")
                stopped = True
            reference = (
                native.reference_state(step["reference"]["long_term_state_after"])
                if native
                else sorted(
                    (_semantic(f) for f in step["reference"]["long_term_state_after"]), key=_json
                )
            )
            row["state_match"] = state == reference
            if native is not None:
                row["action_match"] = row["action"] == step["reference"]["action"]
                row["result"] = result
                expected_subject = step["reference"].get("beneficiary")
                ranking = result.get("ranking") if result is not None else None
                actual_subject = (
                    "self"
                    if ranking and ranking.get("routing_reason") == "owner_profile"
                    else "other"
                    if ranking and ranking.get("routing_reason") == "non_self_public_baseline"
                    else None
                )
                row["subject_match"] = (
                    expected_subject is None or expected_subject == actual_subject
                )
                requested_item = step["reference"].get("requested_item")
                row["requested_item_match"] = requested_item is None or (
                    ranking is not None
                    and [i["item_id"] for i in ranking["items"]] == [requested_item]
                )
            row["reference_reviews"] = step["reference_reviews"]
            rows.append(row)
            save("steps.jsonl", rows)
            if step["turn_id"] == task["turn_id"]:
                groups = {}
                before = store.snapshot_token(user) if row["status"] == "ok" else None
                for group, mem, algorithm in (
                    ("G00", False, False),
                    ("G10", True, False),
                    ("G01", False, True),
                    ("G11", True, True),
                ):
                    rank = None
                    if row["status"] == "ok" and row["action"] == "recommend":
                        try:
                            rank = h.rank_current(
                                use_long_term_memory=mem,
                                strategy="bpr_cold_popularity" if algorithm else "popularity",
                            )
                        except (ValueError, KeyError):
                            pass
                    item_ids = [i["item_id"] for i in rank["items"]] if rank is not None else []
                    groups[group] = dict(
                        status=rank["status"] if rank is not None else "error",
                        item_ids=item_ids,
                        requested_strategy="bpr_cold_popularity" if algorithm else "popularity",
                        used_strategy=rank.get("used_strategy") if rank else None,
                        algorithm_effective=bool(
                            native
                            and rank
                            and rank.get("used_strategy") == "bpr"
                            and rank.get("routing_reason") == "owner_profile"
                            and user in native.service.models["bpr"].active_users
                        ),
                        ranking=rank,
                        metrics=graded_ranking_metrics(
                            item_ids, task["reference"]["item_scores"], k=score_k
                        ),
                    )
                try:
                    valid_state = before is None or store.snapshot_token(user) == before
                except (ValueError, OSError):
                    valid_state = False
                if not valid_state:
                    for arm in groups.values():
                        arm.update(
                            status="error",
                            item_ids=[],
                            metrics=graded_ranking_metrics(
                                [], task["reference"]["item_scores"], k=score_k
                            ),
                        )
                        arm["state_error"] = "runtime_state_changed"
                    stopped = True
                cases.append(
                    dict(
                        task_id=task["task_id"],
                        episode_id=sid,
                        template_family=episode["template_family"],
                        reference_status=task["reference"]["status"],
                        reference_review_codes=task["reference"]["review_codes"],
                        scoring_cutoff=scoring_cutoff,
                        groups=groups,
                    )
                )
                save("cases.jsonl", cases)
            if result is not None and row["status"] == "ok":
                text = result["clarification"] if row["action"] == "clarify" else "已收到。"
                h.observe_assistant(
                    Turn(
                        "A-" + step["turn_id"],
                        "assistant",
                        text,
                        at + timedelta(seconds=2),
                        at + timedelta(seconds=2),
                    )
                )
                question = step["turn_id"] if row["action"] == "clarify" else None
        # 本轮新原生协议：只停止传输失败所在故事，下一段全新故事可继续；不重发。
        if episode_transport and frozen() and len(calls) + per_turn <= budget:
            stopped = False
    good = [c for c in cases if c["reference_status"] == "provisional"]
    contrast = family_memory_contrast(cases, seed=seed)
    report = dict(
        scoring_mode=scoring_mode,
        scoring_protocol=scoring_protocol,
        scoring_k_distribution={
            str(n): sum(c["k"] == n for c in scoring_cutoffs.values())
            for n in sorted({c["k"] for c in scoring_cutoffs.values()})
        },
        planned_steps=planned,
        successful_steps=sum(r["status"] == "ok" for r in rows),
        state_matches=sum(r["status"] == "ok" and r["state_match"] for r in rows),
        processing_errors=sum(r["status"] == "processing_error" for r in rows),
        provider_errors=sum(r["status"] == "provider_error" for r in rows),
        host_errors=sum(r["status"] == "host_error" for r in rows),
        not_run=sum(r["status"] == "not_run" for r in rows),
        unasked_replies=sum(r["status"] == "unasked_reply" for r in rows),
        planned_cases=len(tasks),
        case_errors=sum(any(g["status"] == "error" for g in c["groups"].values()) for c in cases),
        analysis_calls=len(calls),
        api_calls=len(calls) if evaluation_kind == "synthetic_live" else 0,
        reported_tokens=sum(
            c["usage"]["total_tokens"]
            for c in calls
            if c["usage"] is not None and "total_tokens" in c["usage"]
        ),
        calls_without_total_tokens=sum(
            c["usage"] is None or "total_tokens" not in c["usage"] for c in calls
        ),
        reported_usage_totals={
            key: sum(c["usage"].get(key, 0) for c in calls if c["usage"] is not None)
            for key in sorted({key for c in calls if c["usage"] is not None for key in c["usage"]})
        },
        calls_without_reported_usage=sum(c["usage"] is None for c in calls),
        inputs_source_unchanged=frozen(),
        algorithm_effect=None,
        interaction_effect=None,
        algorithm_estimand_identified=False,
        provisional_reference_cases=len(good),
        review_required_cases=len(cases) - len(good),
        memory_public_ndcg_delta=contrast["delta"],
        memory_public_interval=contrast["interval"],
        memory_public_contrast=contrast,
        groups={
            g: {
                key: fmean(
                    c["groups"][g]["metrics"][key]
                    for c in cases
                    if c["groups"][g]["metrics"][key] is not None
                )
                if any(c["groups"][g]["metrics"][key] is not None for c in cases)
                else None
                for key in (
                    "ndcg",
                    "recall",
                    "hit_rate",
                    "mrr",
                    "precision",
                    "constraint_violations",
                )
            }
            for g in GROUPS
        },
        limitations=[
            "external AI synthetic labels unverified; exploratory only",
            "no matched behavior profiles; actual algorithm arms cold-fallback to public prior",
            "neutral scripted acknowledgements during memory replay; not full Pi answer evaluation",
        ],
    )
    if native is not None:
        report.update(
            strict_steps=sum(
                r["status"] == "ok"
                and r["state_match"]
                and r.get("action_match")
                and r.get("subject_match")
                and r.get("requested_item_match")
                for r in rows
            ),
            active_algorithm_case_arms=sum(
                g["algorithm_effective"] for c in cases for g in c["groups"].values()
            ),
            native_bundle_id=native.service.bundle_id,
            limitations=[
                "authored synthetic dialogue/rules; no independent real-user utility target",
                "profile IDs stay local; public catalog titles may appear in model messages",
                "shared analysis context ranking ablation; not whole-agent memory-off",
                "scripted acknowledgements; not complete Pi answer evaluation",
            ],
        )
    (output_dir / "report.json").write_text(_json(report))
    return report
