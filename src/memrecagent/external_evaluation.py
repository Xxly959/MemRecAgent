"""外部合成评测数据的独立审计；不执行生成脚本或修改原始标签。"""

import math
from collections import Counter, defaultdict
from datetime import datetime

UNITS = {
    "price_cny": "CNY",
    "weight_g": "g",
    "battery_life_hours": "h",
    "screen_inches": "inch",
    "storage_gb": "GB",
}
RELATIONS = {"prefer", "dislike", "exclude", "neutral"}
FIELDS = (
    "long_term_add",
    "long_term_retract",
    "long_term_delete",
    "long_term_state_after",
    "session_only",
    "request_only",
)


def _constraint(fact: dict) -> tuple:
    attribute, op, value, unit = (fact[k] for k in ("attribute", "operator", "value", "unit"))
    if attribute == "screen_type":
        if op not in ("=", "==") or unit != "type" or not isinstance(value, str) or not value:
            raise ValueError("invalid enum constraint")
        return attribute, "==", value, unit
    if (
        attribute not in UNITS
        or unit != UNITS[attribute]
        or op not in ("<=", ">=", "==")
        or type(value) not in (int, float)
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ValueError("invalid constraint")
    return attribute, op, value, unit


def _key(fact: dict) -> tuple:
    base = fact["subject"], fact["category"], fact["target_type"]
    if not all(isinstance(v, str) and v for v in base) or fact["category"] != "phone":
        raise ValueError("invalid fact")
    if fact["target_type"] == "brand":
        if (
            not isinstance(fact["target"], str)
            or not fact["target"]
            or fact["relation"] not in RELATIONS
        ):
            raise ValueError("invalid brand")
        return base + (fact["target"],)
    if fact["target_type"] != "attribute" or type(fact["is_hard"]) is not bool:
        raise ValueError("invalid attribute")
    a, op, _, _ = _constraint(fact)
    return base + (a, op)


def _semantic(fact: dict) -> tuple:
    key = _key(fact)
    return key + (
        (fact["relation"],)
        if fact["target_type"] == "brand"
        else (_constraint(fact)[2], fact["unit"], fact["is_hard"])
    )


def _eligible(item: dict, constraints: set[tuple], excluded: set[str]) -> bool:
    for attribute in UNITS.keys() & item.keys():
        value = item[attribute]
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError("invalid item numeric")
    if item["brand"] in excluded:
        return False
    for attribute, op, limit, _ in constraints:
        value = item[attribute]
        if attribute == "screen_type":
            if not isinstance(value, str) or not value:
                raise ValueError("invalid item enum")
            if value != limit:
                return False
            continue
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError("invalid item numeric")
        if not {"<=": value <= limit, ">=": value >= limit, "==": value == limit}[op]:
            return False
    return True


def audit_external_dataset(stories: list[dict], tasks: list[dict]) -> dict:
    """检查结构、证据时序、状态和约束，始终不改原输入。"""
    if not isinstance(stories, list) or not isinstance(tasks, list):
        raise ValueError("数据需要故事和任务列表")
    issues, registry, snapshots = [], {}, {}
    counts = dict(
        stories=len(stories),
        turns=sum(
            len(s["turns"])
            for s in stories
            if isinstance(s, dict) and isinstance(s.get("turns"), list)
        ),
        tasks=len(tasks),
        items=sum(
            len(t["items"])
            for t in tasks
            if isinstance(t, dict) and isinstance(t.get("items"), list)
        ),
        ratings=sum(
            len(t["item_scores"])
            for t in tasks
            if isinstance(t, dict) and isinstance(t.get("item_scores"), list)
        ),
    )
    ineligible_positive = 0
    scenarios, grade_positions = Counter(), defaultdict(list)

    def issue(code, location, severity="error"):
        issues.append(dict(code=code, location=location, severity=severity))

    for story in stories:
        sid = story.get("story_id", "unknown") if isinstance(story, dict) else "unknown"
        try:
            if not isinstance(sid, str) or not sid or sid in registry:
                issue("duplicate_or_invalid_story_id", str(sid))
            registry[sid] = story
            scenarios[story["primary_scenario"]] += 1
            people = {p["person_id"] for p in story["people"]}
            turns, state, deleted, session_context = story["turns"], {}, {}, {}
            ids = [t["turn_id"] for t in turns]
            if len(set(ids)) != len(ids):
                issue("duplicate_turn_id", sid)
            last_at, last_session, closed = -1, None, set()
            for index, turn in enumerate(turns):
                loc, gold = f"{sid}/{turn['turn_id']}", turn["gold"]
                at = turn["day"] * 86400
                parsed = datetime.strptime(turn["time"], "%H:%M")
                at += parsed.hour * 3600 + parsed.minute * 60
                if type(turn["day"]) is not int or turn["day"] < 0 or at <= last_at:
                    issue("non_increasing_time", loc)
                last_at = at
                session = turn["session_id"]
                if session != last_session:
                    if session in closed:
                        issue("reopened_session", loc)
                    if last_session is not None:
                        closed.add(last_session)
                    session_context, last_session = {}, session
                if not isinstance(turn["text"], str) or not turn["text"].strip():
                    issue("invalid_text", loc)
                for field in FIELDS:
                    facts = gold[field]
                    if not isinstance(facts, list):
                        raise ValueError("invalid fact list")
                    keys = []
                    for fact in facts:
                        key = _key(fact)
                        keys.append(key)
                        if fact["subject"] not in people:
                            issue("unknown_subject", loc + "/" + field)
                        if field.startswith("long_term") and fact["subject"] != "self":
                            issue("invalid_long_term_owner", loc + "/" + field)
                        refs = fact["evidence_turn_ids"]
                        if (
                            not isinstance(refs, list)
                            or not refs
                            or len(set(refs)) != len(refs)
                            or any(r not in ids[: index + 1] for r in refs)
                        ):
                            issue("future_or_missing_evidence", loc + "/" + field)
                        if (
                            field == "long_term_add"
                            and key in deleted
                            and any(r in ids and ids.index(r) <= deleted[key] for r in refs)
                        ):
                            issue("deleted_evidence_reuse", loc)
                    if len(set(keys)) != len(keys):
                        issue("duplicate_fact_key", loc + "/" + field)
                if gold["needs_clarification"] and any(
                    gold[f]
                    for f in (
                        "long_term_add",
                        "long_term_retract",
                        "long_term_delete",
                        "session_only",
                        "request_only",
                    )
                ):
                    issue("clarification_has_mutation", loc)
                for field in ("long_term_retract", "long_term_delete"):
                    for fact in gold[field]:
                        key = _key(fact)
                        if key not in state or _semantic(state[key]) != _semantic(fact):
                            issue("missing_or_mismatched_retract_target", loc)
                        state.pop(key, None)
                        if field == "long_term_delete":
                            deleted[key] = index
                for fact in gold["long_term_add"]:
                    key = _key(fact)
                    if key in state and _semantic(state[key]) != _semantic(fact):
                        issue("implicit_replacement_without_retract", loc)
                    state[key] = fact
                if sorted(map(_semantic, state.values())) != sorted(
                    map(_semantic, gold["long_term_state_after"])
                ):
                    issue("gold_state_mismatch", loc)
                for fact in gold["session_only"]:
                    session_context[_key(fact)] = fact
                snapshots[(sid, turn["turn_id"])] = (turn, dict(state), dict(session_context))
        except (KeyError, TypeError, ValueError, AttributeError, OverflowError):
            issue("invalid_story_schema", str(sid))

    task_ids, item_ids = set(), set()
    for task in tasks:
        tid = task.get("task_id", "unknown") if isinstance(task, dict) else "unknown"
        try:
            if not isinstance(tid, str) or not tid or tid in task_ids:
                issue("duplicate_or_invalid_task_id", str(tid))
            task_ids.add(tid)
            turn, state, session = snapshots[(task["story_id"], task["turn_id"])]
            if (
                task["request_text"] != turn["text"]
                or task["knowledge_cutoff_turn"] != turn["turn_id"]
            ):
                issue("request_or_cutoff_mismatch", tid)
            person = task["for_person"]
            if person not in {p["person_id"] for p in registry[task["story_id"]]["people"]}:
                issue("unknown_task_subject", tid)
            # 请求同属性覆盖长期，人物会话信息不会跨session复制。
            effective = {k: f for k, f in state.items() if person == "self"}
            for fact in list(session.values()) + turn["gold"]["request_only"]:
                if fact["subject"] == person:
                    effective[_key(fact)] = fact
            attrs = [f for f in effective.values() if f["target_type"] == "attribute"]
            expected_constraints = {_constraint(f) for f in attrs if f["is_hard"]}
            try:
                supplied = [_constraint(c) for c in task["hard_constraints"]]
            except (KeyError, TypeError, ValueError):
                issue("invalid_constraint", tid)
                continue
            if len(set(supplied)) != len(supplied):
                issue("duplicate_constraint", tid)
            if set(supplied) != expected_constraints:
                issue("scoped_constraint_mismatch", tid, "review")
            excluded = {
                f["target"]
                for f in effective.values()
                if f["target_type"] == "brand" and f["relation"] == "exclude"
            }
            if set(task["excluded_brands"]) != excluded:
                issue("scoped_exclusion_mismatch", tid, "review")
            items, ratings = task["items"], task["item_scores"]
            local_ids = [i["item_id"] for i in items]
            if len(set(local_ids)) != len(local_ids) or item_ids.intersection(local_ids):
                issue("duplicate_item_id", tid)
            item_ids.update(local_ids)
            rating_ids = [r["item_id"] for r in ratings]
            if len(set(rating_ids)) != len(rating_ids):
                issue("duplicate_rating_id", tid)
            if set(rating_ids) != set(local_ids):
                issue("rating_item_set_mismatch", tid)
            scores = {r["item_id"]: r for r in ratings}
            brand_scores = defaultdict(set)
            for position, item in enumerate(items, 1):
                rating = scores[item["item_id"]]
                if (
                    type(rating["relevance"]) is not int
                    or not 0 <= rating["relevance"] <= 3
                    or type(rating["eligible"]) is not bool
                ):
                    issue("invalid_rating", tid)
                eligibility = _eligible(item, set(supplied), set(task["excluded_brands"]))
                if rating["eligible"] != eligibility:
                    issue("eligibility_mismatch", tid + "/" + item["item_id"])
                if bool(rating["violated_constraints"]) == eligibility:
                    issue("violation_list_mismatch", tid + "/" + item["item_id"])
                if not eligibility and rating["relevance"] > 0:
                    ineligible_positive += 1
                if eligibility:
                    brand_scores[item["brand"]].add(rating["relevance"])
                grade_positions[position].append(rating["relevance"])
            if not attrs and any(len(values) > 1 for values in brand_scores.values()):
                issue("within_brand_grade_needs_review", tid, "review")
        except (KeyError, TypeError, ValueError, AttributeError, OverflowError):
            issue("invalid_task_schema", str(tid))
    return dict(
        schema_version=1,
        counts=counts,
        scenarios=dict(scenarios),
        issues=issues,
        structural_valid=not any(i["severity"] == "error" for i in issues),
        human_review_required=any(i["severity"] == "review" for i in issues),
        semantic_gold_independently_verified=False,
        ineligible_positive_raw_ratings=ineligible_positive,
        ranking_gain_policy="eligible_only",
        mean_grade_by_input_position={str(k): sum(v) / len(v) for k, v in grade_positions.items()},
        limitations=[
            "结构检查不证明语义标签正确",
            "请求延续与隐含属性须人工复核",
            "不执行生成脚本或模型调用",
        ],
    )
