"""固定历史与已知前态下的共享草案分析评测，不冒充自动写入闭环。"""

import json
import random
import re
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime
from hashlib import sha256
from pathlib import Path
from statistics import fmean
from time import perf_counter

from memrecagent.deepseek_client import ModelReply, ProviderError
from memrecagent.memory import PreferenceMemory
from memrecagent.memory_analysis_control import analysis_projection, compare_shared_draft
from memrecagent.memory_analyst import AnalystError, Complete, _binding, _metadata, _validate_heads
from memrecagent.memory_extraction import ExtractionError, Turn, unique_object, validate_turns
from memrecagent.memory_interaction_extraction import build_interaction_messages, decode_interaction
from memrecagent.memory_quality_evaluation import _source_paths


def _json(value: object) -> str:
    return json.dumps(value, sort_keys=True, allow_nan=False, ensure_ascii=True)


def parse_shared_suite(value: dict, *, split: str = "dev") -> list[dict]:
    """参考是可严格解码的候选，全部先验证，绝不传给模型或补入实际草案。"""
    try:
        if (
            split not in ("dev", "test", "all")
            or not isinstance(value, dict)
            or value.keys() != {"schema_version", "synthetic", "label_source", "cases"}
            or type(value["schema_version"]) is not int
            or value["schema_version"] != 1
            or value["synthetic"] is not True
            or not isinstance(value["label_source"], str)
            or not value["label_source"].strip()
            or not isinstance(value["cases"], list)
            or not 1 <= len(value["cases"]) <= 200
            or len(_json(value).encode()) > 8 * 1024 * 1024
        ):
            raise ValueError
        identities, clusters, parsed = set(), {}, []
        for case in value["cases"]:
            if not isinstance(case, dict) or case.keys() != {
                "case_id",
                "cluster_id",
                "split",
                "runtime",
                "references",
            }:
                raise ValueError
            for key in ("case_id", "cluster_id"):
                if not isinstance(case[key], str) or not re.fullmatch(
                    r"[A-Za-z0-9_-]{1,64}", case[key]
                ):
                    raise ValueError
            if (
                case["case_id"] in identities
                or case["split"] not in ("dev", "test")
                or clusters.setdefault(case["cluster_id"], case["split"]) != case["split"]
            ):
                raise ValueError
            identities.add(case["case_id"])
            runtime = case["runtime"]
            if (
                not isinstance(runtime, dict)
                or runtime.keys()
                != {"turns", "as_of", "active_memories", "pending_question_turn_id"}
                or not isinstance(runtime["turns"], list)
                or not 1 <= len(runtime["turns"]) <= 64
            ):
                raise ValueError
            turns = []
            for t in runtime["turns"]:
                if not isinstance(t, dict) or t.keys() != {
                    "turn_id",
                    "role",
                    "content",
                    "occurred_at",
                    "available_at",
                }:
                    raise ValueError
                turns.append(
                    Turn(
                        t["turn_id"],
                        t["role"],
                        t["content"],
                        datetime.fromisoformat(t["occurred_at"]),
                        datetime.fromisoformat(t["available_at"]),
                    )
                )
            turns = tuple(turns)
            known = validate_turns(turns)
            time = _binding(
                "synthetic-shared",
                case["case_id"],
                datetime.fromisoformat(runtime["as_of"]),
                "frozen",
            )
            if sum(len(t.content) for t in turns) > 24000 or any(
                t.available_at >= time for t in turns
            ):
                raise ValueError
            if (
                not isinstance(runtime["active_memories"], list)
                or len(runtime["active_memories"]) > 256
            ):
                raise ValueError
            memories = tuple(PreferenceMemory.from_dict(m) for m in runtime["active_memories"])
            _validate_heads(memories, user_id="synthetic-shared", as_of=time)
            question_id = runtime["pending_question_turn_id"]
            if question_id is not None and (
                not isinstance(question_id, str) or question_id not in known
            ):
                raise ValueError
            question = known[question_id] if question_id is not None else None
            build_interaction_messages(turns, (), pending_question=question)
            refs = case["references"]
            if not isinstance(refs, list) or not 1 <= len(refs) <= 8:
                raise ValueError
            projections = []
            for ref in refs:
                decoded = decode_interaction(_json(ref), turns=turns)
                if (
                    decoded.confirmation is not None
                    and decoded.confirmation.question_turn_id != question_id
                ):
                    raise ValueError
                projection = analysis_projection(decoded, turns)
                if projection not in projections:
                    projections.append(projection)
            if split == "all" or case["split"] == split:
                parsed.append(
                    dict(
                        case_id=case["case_id"],
                        cluster_id=case["cluster_id"],
                        turns=turns,
                        as_of=time,
                        memories=memories,
                        question=question,
                        references=projections,
                    )
                )
        if not parsed:
            raise ValueError
        return parsed
    except (
        ValueError,
        TypeError,
        KeyError,
        ExtractionError,
        AnalystError,
        OverflowError,
        UnicodeError,
        RecursionError,
    ):
        raise ValueError("共享分析语料无效") from None


def evaluate_shared_analysis(
    value: dict,
    *,
    complete: Complete,
    output_dir: Path,
    max_calls: int | None = None,
    split: str = "dev",
    seed: int = 20261024,
    evaluation_kind: str = "fixed_client",
    input_file: Path | None = None,
) -> dict:
    """三次完整配对预算；调用、跳过、错误及参考匹配分别保存。"""
    value = json.loads(_json(value))
    cases = parse_shared_suite(value, split=split)
    limit = 3 * len(cases) if max_calls is None else max_calls
    if (
        type(limit) is not int
        or not 3 <= limit <= 3 * len(cases)
        or not callable(complete)
        or type(seed) is not int
        or not 0 <= seed < 2**63
        or evaluation_kind not in ("fixed_client", "synthetic_live")
    ):
        raise ValueError("共享分析配置无效")
    original = None
    if input_file is not None:
        input_file = Path(input_file).resolve()
        original = input_file.read_bytes()
        if _json(json.loads(original, object_pairs_hook=unique_object)) != _json(value):
            raise ValueError("共享分析输入不一致")
    output_dir = Path(output_dir)
    source = Path(__file__).parent.resolve()
    if (
        source == output_dir.resolve()
        or source in output_dir.resolve().parents
        or output_dir.resolve() in source.parents
    ):
        raise ValueError("输出不可覆盖源码")

    def paths() -> dict[str, Path]:
        result = _source_paths()
        entry = Path(__file__).resolve().parents[2] / "scripts/evaluate_shared_analysis.py"
        if entry.is_file():
            result[entry.name] = entry
        if input_file is not None:
            result["input_suite.json"] = input_file
        return result

    source_bytes = {name: p.read_bytes() for name, p in paths().items()}
    fingerprints = {name: sha256(content).hexdigest() for name, content in source_bytes.items()}
    inputs = _json(value)
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "inputs.json").write_text(inputs)
    for name, content in source_bytes.items():
        dest = output_dir / "source" / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(content)
    rng = random.Random(seed)
    rng.shuffle(cases)
    orders = [
        ("analyst", "analyst_blind") if (index + seed) % 2 == 0 else ("analyst_blind", "analyst")
        for index in range(len(cases))
    ]
    manifest = dict(
        source_sha256=fingerprints,
        inputs_sha256=sha256(inputs.encode()).hexdigest(),
        raw_input_sha256=sha256(original).hexdigest() if original is not None else None,
        planned_cases=len(cases),
        max_client_attempts=limit,
        seed=seed,
        split=split,
        order=[dict(case_id=c["case_id"], arm_order=o) for c, o in zip(cases, orders, strict=True)],
        initial_memory="provided fixed heads; not automatic history writes",
        scope="read-only candidate semantic match; no commits or Pi answers",
        evaluation_kind=evaluation_kind,
    )
    (output_dir / "manifest.json").write_text(_json(manifest))

    def frozen() -> bool:
        try:
            return {
                n: sha256(p.read_bytes()).hexdigest() for n, p in paths().items()
            } == fingerprints and (output_dir / "inputs.json").read_text() == inputs
        except OSError:
            return False

    records, rows = [], []
    stopped = False

    def save(name: str, values: list[dict]) -> None:
        (output_dir / name).write_text("".join(_json(v) + "\n" for v in values))

    for case, order in zip(cases, orders, strict=True):
        row = dict(case_id=case["case_id"], cluster_id=case["cluster_id"], arm_order=order, arms={})
        first = len(records)
        if sum(r["sent"] for r in records) + 3 > limit or not frozen():
            stopped = True
        result = None
        if not stopped:

            def recorded(
                messages: list[dict[str, str]], *, case_id: str = case["case_id"]
            ) -> ModelReply:
                record = dict(
                    case_id=case_id,
                    sent=False,
                    status="attempted",
                    usage=None,
                    messages=json.loads(_json(messages)),
                    request_sha256=sha256(_json(messages).encode()).hexdigest(),
                )
                records.append(record)
                started = perf_counter()
                try:
                    if not frozen():
                        record["controller_error"] = "freeze_changed"
                        raise ProviderError("freeze_changed")
                    if sum(r["sent"] for r in records) >= limit:
                        record["controller_error"] = "budget"
                        raise ProviderError("budget")
                    record["sent"] = True
                    save("calls.jsonl", records)
                    response = complete(messages)
                    if isinstance(response, ModelReply):
                        try:
                            meta = _metadata(response)
                        except AnalystError:
                            record["returned_invalid_metadata"] = True
                        else:
                            record.update(
                                model=meta["model"],
                                fingerprint=meta["fingerprint"],
                                usage=dict(meta["usage"]) if meta["usage"] is not None else None,
                            )
                        if (
                            isinstance(response.content, str)
                            and len(response.content.encode()) <= 65536
                        ):
                            record["response"] = response.content
                    record["status"] = "returned"
                    if not frozen():
                        record["controller_error"] = "freeze_changed"
                        raise ProviderError("freeze_changed")
                    return response
                except Exception:
                    record["status"] = "callback_error"
                    raise
                finally:
                    record["wall_seconds"] = perf_counter() - started
                    save("calls.jsonl", records)

            result = compare_shared_draft(
                case["turns"],
                complete=recorded,
                user_id="synthetic-shared",
                session_id=case["case_id"],
                as_of=case["as_of"],
                snapshot_token=fingerprints["memory_analysis_control.py"],
                active_memories=case["memories"],
                pending_question=case["question"],
                arm_order=order,
            )
            if len(records) - first != len(result.calls):
                raise RuntimeError("shared_call_accounting")
            for offset, call in enumerate(result.calls):
                records[first + offset].update(analysis_call=asdict(call), stage=call.stage)
            save("calls.jsonl", records)
            stopped = result.fatal
        arm_map = {a.mode: a for a in result.arms} if result is not None else {}
        for mode in ("direct", "analyst", "analyst_blind"):
            arm = arm_map.get(mode)
            projection = (
                analysis_projection(arm.analysis, case["turns"]) if arm and arm.analysis else None
            )
            row["arms"][mode] = dict(
                status=arm.status if arm else "not_run",
                projection=projection,
                call_index=first + arm.call_index if arm and arm.call_index is not None else None,
                error_code=arm.error_code if arm else None,
                matches_reference=arm is not None
                and arm.status == "ok"
                and projection in case["references"],
            )
        rows.append(row)
        save("cases.jsonl", rows)
    save("calls.jsonl", records)
    ready = frozen() and all(
        a["status"] in ("ok", "processing_error") for r in rows for a in r["arms"].values()
    )
    contrasts = {}
    for high, low in (
        ("analyst", "direct"),
        ("analyst_blind", "direct"),
        ("analyst_blind", "analyst"),
    ):
        groups = defaultdict(list)
        for r in rows:
            groups[r["cluster_id"]].append(
                int(r["arms"][high]["matches_reference"]) - int(r["arms"][low]["matches_reference"])
            )
        diffs = [fmean(v) for _, v in sorted(groups.items())]
        interval = None
        if ready and len(diffs) > 1:
            bootstrap = random.Random(seed)
            draws = sorted(fmean(bootstrap.choices(diffs, k=len(diffs))) for _ in range(1000))
            interval = [draws[24], draws[974]]
        contrasts[high + "_minus_" + low] = dict(
            delta=fmean(diffs),
            interval=interval,
            n_clusters=len(diffs),
            unit="whole_cluster",
            n_cases=len(rows),
        )
    report = dict(
        planned_cases=len(cases),
        callback_attempts=len(records),
        client_attempts=sum(r["sent"] for r in records),
        api_calls=sum(r["sent"] for r in records) if evaluation_kind == "synthetic_live" else 0,
        reported_total_tokens=sum((r["usage"] or {}).get("total_tokens", 0) for r in records),
        calls_without_total_tokens=sum(
            r["sent"] and "total_tokens" not in (r["usage"] or {}) for r in records
        ),
        comparison_ready=ready,
        inputs_source_unchanged=frozen(),
        evaluation_kind=evaluation_kind,
        arms={
            mode: dict(
                matches=sum(r["arms"][mode]["matches_reference"] for r in rows),
                not_run=sum(r["arms"][mode]["status"] == "not_run" for r in rows),
                errors=sum(r["arms"][mode]["status"] not in ("ok", "not_run") for r in rows),
            )
            for mode in ("direct", "analyst", "analyst_blind")
        },
        contrasts=contrasts,
        limitations=[
            "provided synthetic references; independent correctness not verified",
            "fixed prior heads; candidate semantics only, not commit/answer/user utility",
            "draft selects shared memory view; conditional direct-visibility effect",
            "shared draft costs counted once; arm-equivalent costs must not be summed",
        ],
    )
    (output_dir / "report.json").write_text(_json(report))
    return report
