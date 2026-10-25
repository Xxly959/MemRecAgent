"""按最终写入事实与实际状态评分；无写入的精确率保持未知。"""

from collections import Counter
from math import expm1, log


def score_quality_step(expected: dict, actual: dict) -> dict:
    """事实多重集防重复记功；人物/范围污染只统计错误长期写入。"""
    gold = Counter(map(tuple, expected["writes"]))
    written = Counter(map(tuple, actual["writes"]))
    correct = sum((gold & written).values())
    wrong = written - gold
    expected_entities = {row[:2]: row[2] for row in gold}
    checks = {
        "action": actual["action"] in expected["actions"],
        **{
            key: sorted(actual[key]) == sorted(expected[key])
            for key in ("writes", "heads", "activated", "other_context", "revisions")
        },
    }

    def pollution(key):
        forbidden = set(map(tuple, expected[key]))
        entity_only = {row[:2] for row in forbidden} - expected_entities.keys()
        return sum(
            count for row, count in wrong.items() if row in forbidden or row[:2] in entity_only
        )

    return {
        "status": actual["status"],
        "strict": actual["status"] == "ok" and all(checks.values()),
        "checks": checks,
        "expected_writes": sum(gold.values()),
        "actual_writes": sum(written.values()),
        "correct_writes": correct,
        "wrong_writes": sum(wrong.values()),
        "missing_writes": sum((gold - written).values()),
        "polarity_eligible": sum(
            count for row, count in written.items() if row[:2] in expected_entities
        ),
        "polarity_correct": sum(
            count for row, count in written.items() if expected_entities.get(row[:2]) == row[2]
        ),
        "wrong_attribution_records": pollution("other_only"),
        "wrong_scope_records": pollution("current_only"),
        "quoted_pollution_records": pollution("quoted_only"),
        "other_attribution_opportunity": bool(expected["other_only"]),
        "expected_revisions": len(expected["revisions"]),
        "correct_revisions": sum(
            (
                Counter(map(tuple, expected["revisions"]))
                & Counter(map(tuple, actual["revisions"]))
            ).values()
        ),
        "clarification_expected": expected["actions"] == ["clarify"],
        "clarification_actual": actual["action"] == "clarify",
    }


def _ratio(a: int, b: int) -> float | None:
    return a / b if b else None


def _aggregate(rows: list[dict]) -> dict:
    totals = {
        key: sum(r[key] for r in rows)
        for key in (
            "expected_writes",
            "actual_writes",
            "correct_writes",
            "wrong_writes",
            "missing_writes",
            "wrong_attribution_records",
            "wrong_scope_records",
            "quoted_pollution_records",
            "expected_revisions",
            "correct_revisions",
            "polarity_eligible",
            "polarity_correct",
        )
    }
    episodes = {r["episode_id"] for r in rows}
    writes = totals["actual_writes"]
    opportunities = sum(r["other_attribution_opportunity"] for r in rows)
    attributed_steps = sum(r["wrong_attribution_records"] > 0 for r in rows)
    clarification_tp = sum(r["clarification_expected"] and r["clarification_actual"] for r in rows)
    return {
        "n_steps": len(rows),
        "strict_steps": sum(r["strict"] for r in rows),
        "n_episodes": len(episodes),
        "strict_episodes": sum(
            all(r["strict"] for r in rows if r["episode_id"] == e) for e in episodes
        ),
        "writes": {
            "expected": totals["expected_writes"],
            "actual": writes,
            "correct": totals["correct_writes"],
            "wrong": totals["wrong_writes"],
            "missing": totals["missing_writes"],
            "precision": _ratio(totals["correct_writes"], writes),
            "recall": _ratio(totals["correct_writes"], totals["expected_writes"]),
        },
        **{
            key: totals[key]
            for key in (
                "wrong_attribution_records",
                "wrong_scope_records",
                "quoted_pollution_records",
            )
        },
        "wrong_attribution_per_written_record": _ratio(totals["wrong_attribution_records"], writes),
        "n_attribution_opportunities": opportunities,
        "attribution_error_step_rate": _ratio(attributed_steps, opportunities),
        "revision_recall": _ratio(totals["correct_revisions"], totals["expected_revisions"]),
        "n_expected_revisions": totals["expected_revisions"],
        "polarity_accuracy_given_expected_entity": _ratio(
            totals["polarity_correct"], totals["polarity_eligible"]
        ),
        "clarification": {
            "true_positive": clarification_tp,
            "precision": _ratio(clarification_tp, sum(r["clarification_actual"] for r in rows)),
            "recall": _ratio(clarification_tp, sum(r["clarification_expected"] for r in rows)),
        },
        # 零事件的单侧95%二项上界；相关对话并非独立伯努利试验，不作人群外推。
        "zero_error_upper_95_per_write": -expm1(log(0.05) / writes)
        if writes and not totals["wrong_writes"]
        else None,
        "zero_error_bound_limit": "one-sided binomial descriptive bound; writes may be correlated",
        **{
            f"n_{name}": sum(r["status"] == status for r in rows)
            for name, status in (
                ("processing_errors", "processing_error"),
                ("provider_errors", "provider_error"),
                ("preflight_errors", "preflight_error"),
                ("not_run", "not_run"),
                ("unasked_replies", "unasked_reply"),
            )
        },
        "checks_correct": {k: sum(r["checks"][k] for r in rows) for k in rows[0]["checks"]},
    }


def summarize_quality(rows: list[dict]) -> dict:
    """所有计划轮保留；类别指标是事实微平均，完整故事单列。"""
    if not rows:
        raise ValueError("评分需要计划轮次")
    result = _aggregate(rows)
    result["by_split"] = {
        s: _aggregate([r for r in rows if r["split"] == s])
        for s in sorted({r["split"] for r in rows})
    }
    result["by_scenario"] = {
        s: _aggregate([r for r in rows if r["scenario"] == s])
        for s in sorted({r["scenario"] for r in rows})
    }
    result["by_tag"] = {
        tag: _aggregate([r for r in rows if tag in r.get("tags", [])])
        for tag in sorted({tag for r in rows for tag in r.get("tags", [])})
    }
    return result
