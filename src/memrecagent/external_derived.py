"""显式政策下派生外部合成语料；运行输入与待复核参考始终分开。"""

import json
import re
from copy import deepcopy
from hashlib import sha256

from memrecagent.external_evaluation import (
    FIELDS,
    _constraint,
    _eligible,
    _key,
    _semantic,
    audit_external_dataset,
)
from memrecagent.product_attributes import UNITS


def safe_external_id(value: object) -> bool:
    """外部ID将成为本地目录片段，不能接受路径分隔或上级目录。"""
    return isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9_-]{1,64}", value) is not None


def value_digest(value: object) -> str:
    """规范JSON摘要仅作政策旧值前置条件，不是语义正确证明。"""
    return sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


def _policies(stories: list[dict], policy: dict) -> tuple[list[dict], list[dict], dict]:
    if (
        not isinstance(policy, dict)
        or policy.keys() != {"schema_version", "policy_id", "seed", "patches", "turn_reviews"}
        or type(policy["schema_version"]) is not int
        or policy["schema_version"] != 1
        or not isinstance(policy["policy_id"], str)
        or not policy["policy_id"].strip()
        or type(policy["seed"]) is not int
        or not 0 <= policy["seed"] < 2**63
        or not isinstance(policy["patches"], list)
        or len(policy["patches"]) > 360
        or not isinstance(policy["turn_reviews"], list)
        or len(policy["turn_reviews"]) > 360
    ):
        raise ValueError("invalid derivation policy")
    derived = deepcopy(stories)
    registry = {(s["story_id"], t["turn_id"]): t for s in derived for t in s["turns"]}
    if len(registry) != sum(len(s["turns"]) for s in derived):
        raise ValueError("duplicate story/turn")
    changes, touched = [], set()
    allowed = set(FIELDS) | {
        "action",
        "needs_clarification",
        "clarification_reason",
        "acceptable_clarification",
        "must_not_infer",
        "reason",
    }
    for patch in policy["patches"]:
        if not isinstance(patch, dict) or patch.keys() != {
            "story_id",
            "turn_id",
            "field",
            "before_sha256",
            "after",
            "reason",
        }:
            raise ValueError("invalid policy patch")
        key = patch["story_id"], patch["turn_id"]
        target = (*key, patch["field"])
        if (
            key not in registry
            or patch["field"] not in allowed
            or target in touched
            or not isinstance(patch["reason"], str)
            or not patch["reason"].strip()
        ):
            raise ValueError("invalid patch target")
        gold = registry[key]["gold"]
        before = gold[patch["field"]]
        if value_digest(before) != patch["before_sha256"]:
            raise ValueError("stale policy patch")
        touched.add(target)
        changes.append(
            dict(
                story_id=key[0],
                turn_id=key[1],
                field=patch["field"],
                before=deepcopy(before),
                after=deepcopy(patch["after"]),
                reason=patch["reason"],
            )
        )
        gold[patch["field"]] = deepcopy(patch["after"])
    reviews = {}
    for review in policy["turn_reviews"]:
        if not isinstance(review, dict) or review.keys() != {"story_id", "turn_id", "reason"}:
            raise ValueError("invalid turn review")
        key = review["story_id"], review["turn_id"]
        if (
            key not in registry
            or not isinstance(review["reason"], str)
            or not review["reason"].strip()
        ):
            raise ValueError("unknown review target")
        reviews.setdefault(key, []).append(review["reason"])
    return derived, changes, reviews


def _snapshots(stories: list[dict]) -> dict:
    snapshots = {}
    for story in stories:
        session, context = None, {}
        for turn in story["turns"]:
            if session != turn["session_id"]:
                session, context = turn["session_id"], {}
            for fact in turn["gold"]["session_only"]:
                context[_key(fact)] = fact
            state = {_key(f): f for f in turn["gold"]["long_term_state_after"]}
            snapshots[(story["story_id"], turn["turn_id"])] = turn, state, dict(context)
    return snapshots


def derive_external_suite(stories: list[dict], tasks: list[dict], policy: dict) -> dict:
    """无模型、无训练、无写盘；所有参考修改留下可核验的原值和理由。"""
    if any(
        not safe_external_id(s["story_id"])
        or any(not safe_external_id(t[k]) for t in s["turns"] for k in ("turn_id", "session_id"))
        for s in stories
    ) or any(not safe_external_id(t["task_id"]) for t in tasks):
        raise ValueError("invalid external identifier")
    source_hashes = {
        "stories": value_digest(stories),
        "tasks": value_digest(tasks),
        "policy": value_digest(policy),
    }
    corrected, changes, reviews = _policies(stories, policy)
    if any(
        type(t["gold"]["needs_clarification"]) is not bool
        or t["gold"]["action"]
        not in ("answer", "compare", "recommend", "clarify", "continue", "manage_memory")
        for s in corrected
        for t in s["turns"]
    ):
        raise ValueError("invalid reference action or clarification flag")
    if not audit_external_dataset(corrected, tasks)["structural_valid"]:
        raise ValueError("unresolved raw reference errors")
    for task in tasks:
        if any(
            item.keys() - ({"item_id", "brand", "category"} | UNITS.keys())
            for item in task["items"]
        ):
            raise ValueError("unsupported catalog fields; labels cannot enter runtime")
    snapshots = _snapshots(corrected)
    derived_tasks = deepcopy(tasks)
    constraint_changes = set()
    for task in derived_tasks:
        turn, state, context = snapshots[(task["story_id"], task["turn_id"])]
        person = task["for_person"]
        effective = {k: f for k, f in state.items() if person == "self"}
        for fact in list(context.values()) + turn["gold"]["request_only"]:
            if fact["subject"] == person:
                effective[_key(fact)] = fact
        constraints = sorted(
            {
                _constraint(f)
                for f in effective.values()
                if f["target_type"] == "attribute" and f["is_hard"]
            }
        )
        exclusions = sorted(
            {
                f["target"]
                for f in effective.values()
                if f["target_type"] == "brand" and f["relation"] == "exclude"
            }
        )
        supplied = sorted({_constraint(f) for f in task["hard_constraints"]})
        if constraints != supplied or exclusions != sorted(task["excluded_brands"]):
            constraint_changes.add(task["task_id"])
            changes.append(
                dict(
                    task_id=task["task_id"],
                    field="reference_constraints",
                    before=dict(
                        hard_constraints=deepcopy(task["hard_constraints"]),
                        excluded_brands=deepcopy(task["excluded_brands"]),
                    ),
                    after=dict(hard_constraints=constraints, excluded_brands=exclusions),
                    reason=(
                        "current same-direction requirement overrides prior reference; "
                        "relevance is not repaired"
                    ),
                )
            )
        task["hard_constraints"] = [
            dict(zip(("attribute", "operator", "value", "unit"), c, strict=True))
            for c in constraints
        ]
        task["excluded_brands"] = exclusions
        items = {i["item_id"]: i for i in task["items"]}
        for score in task["item_scores"]:
            eligible = _eligible(items[score["item_id"]], set(constraints), set(exclusions))
            if score["eligible"] != eligible or task["task_id"] in constraint_changes:
                before = dict(
                    eligible=score["eligible"],
                    violated_constraints=deepcopy(score["violated_constraints"]),
                )
                score["eligible"] = eligible
                score["violated_constraints"] = [] if eligible else ["derived_reference_constraint"]
                after = dict(
                    eligible=score["eligible"],
                    violated_constraints=deepcopy(score["violated_constraints"]),
                )
                if before != after:
                    changes.append(
                        dict(
                            task_id=task["task_id"],
                            item_id=score["item_id"],
                            field="reference_eligibility",
                            before=before,
                            after=after,
                            reason=(
                                "recomputed after explicit scoped-reference correction; "
                                "original relevance preserved"
                            ),
                        )
                    )
    audit = audit_external_dataset(corrected, derived_tasks)
    if not audit["structural_valid"]:
        raise ValueError("unresolved structural reference errors")
    family_names = {s["primary_scenario"] for s in corrected}
    if not family_names or any(not isinstance(f, str) or not f for f in family_names):
        raise ValueError("invalid family")
    families = sorted(family_names, key=lambda f: value_digest(["split", policy["seed"], f]))
    # 缺原始模板ID，整场景家族分组比随机拆同家族更保守；分布差异须公开。
    assignment = {
        f: "dev" if n < (len(families) + 1) // 2 else "test" for n, f in enumerate(families)
    }
    episodes = []
    split_by_story = {}
    for story in corrected:
        sid = story["story_id"]
        split = assignment[story["primary_scenario"]]
        split_by_story[sid] = split
        steps, previous, prior = [], None, {}
        for turn in story["turns"]:
            hour, minute = map(int, turn["time"].split(":"))
            reply = (
                previous["turn_id"]
                if previous is not None
                and previous["session_id"] == turn["session_id"]
                and previous["gold"]["needs_clarification"]
                else None
            )
            added = turn["gold"]["long_term_add"]
            optional = [
                f for f in added if _key(f) in prior and _semantic(prior[_key(f)]) == _semantic(f)
            ]
            transition = dict(
                required_upserts=deepcopy([f for f in added if f not in optional]),
                optional_refreshes=deepcopy(optional),
                deletions=deepcopy(turn["gold"]["long_term_delete"]),
                retractions=deepcopy(turn["gold"]["long_term_retract"]),
            )
            steps.append(
                dict(
                    turn_id=turn["turn_id"],
                    session_id=turn["session_id"],
                    offset_seconds=turn["day"] * 86400 + hour * 3600 + minute * 60,
                    text=turn["text"],
                    reply_to=reply,
                    reference=deepcopy(turn["gold"]),
                    reference_transition=transition,
                    reference_reviews=reviews.get((sid, turn["turn_id"]), []),
                )
            )
            prior = {_key(f): f for f in turn["gold"]["long_term_state_after"]}
            previous = turn
        episodes.append(
            dict(
                episode_id=sid,
                scenario=story["primary_scenario"],
                template_family="scenario:" + story["primary_scenario"],
                split=split,
                people=deepcopy(story["people"]),
                steps=steps,
            )
        )
    raw_tasks = {t["task_id"]: t for t in tasks}
    output_tasks = []
    for task in derived_tasks:
        tid, sid = task["task_id"], task["story_id"]
        ids = {
            i["item_id"]: "p_" + value_digest(["id", policy["seed"], tid, i["item_id"]])[:24]
            for i in task["items"]
        }
        ordered = sorted(
            task["items"], key=lambda i: value_digest(["order", policy["seed"], tid, i["item_id"]])
        )
        review_codes = {
            i["code"]
            for i in audit["issues"]
            if i["severity"] == "review" and i["location"].split("/")[0] == tid
        }
        if tid in constraint_changes:
            review_codes.add("scoped_constraint_mismatch")
        if any(key[0] == sid for key in reviews):
            review_codes.add("story_semantics_review")
        output_tasks.append(
            dict(
                task_id=tid,
                episode_id=sid,
                turn_id=task["turn_id"],
                split=split_by_story[sid],
                runtime=dict(
                    request_text=task["request_text"],
                    items=[deepcopy(i) | {"item_id": ids[i["item_id"]]} for i in ordered],
                ),
                reference=dict(
                    status="review_required" if review_codes else "provisional",
                    review_codes=sorted(review_codes),
                    for_person=task["for_person"],
                    hard_constraints=task["hard_constraints"],
                    excluded_brands=task["excluded_brands"],
                    original_ids={v: k for k, v in ids.items()},
                    item_scores=[
                        deepcopy(r) | {"item_id": ids[r["item_id"]]} for r in task["item_scores"]
                    ],
                    raw_item_scores=deepcopy(raw_tasks[tid]["item_scores"]),
                ),
            )
        )
    return dict(
        schema_version=1,
        synthetic=True,
        metadata=dict(
            source_kind="external_AI_synthetic_unverified",
            policy_id=policy["policy_id"],
            source_hashes=source_hashes,
            semantic_gold_independently_verified=False,
            algorithm_estimand_identified=False,
            behavior_history_available=False,
            ranking_gain_policy="eligible_only",
            split_policy=(
                "whole_primary_scenario_family; missing original template IDs; not blind holdout"
            ),
            family_assignment=assignment,
            conditional_reply_policy=(
                "inferred immediate same-session reply after reference "
                "clarification; send only if actually asked"
            ),
            api_calls=0,
        ),
        episodes=episodes,
        tasks=output_tasks,
        changes=changes,
        reference_audit=audit,
    )
