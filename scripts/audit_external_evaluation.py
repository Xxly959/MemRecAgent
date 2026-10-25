"""只读验收外部JSONL，封存原始字节；零模型调用，不改标签或划分。"""

import argparse
import json
import sys
from hashlib import sha256
from pathlib import Path

from memrecagent.external_evaluation import audit_external_dataset
from memrecagent.memory_extraction import unique_object


def decode(raw: bytes) -> list[dict]:
    """拒绝重复键/非有限数字，保留输入原文仅在显式产物目录。"""
    if not 0 < len(raw) <= 4 * 1024 * 1024:
        raise ValueError("size")

    def invalid(value):
        raise ValueError("non-finite")

    lines = raw.decode("utf-8").splitlines()
    if not 1 <= len(lines) <= 1000 or any(not line.strip() for line in lines):
        raise ValueError("lines")
    return [
        json.loads(line, object_pairs_hook=unique_object, parse_constant=invalid) for line in lines
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stories", type=Path, required=True)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.output_dir.exists():
            raise ValueError("exists")
        raw_stories, raw_tasks = args.stories.read_bytes(), args.tasks.read_bytes()
        stories, tasks = decode(raw_stories), decode(raw_tasks)
        report = audit_external_dataset(stories, tasks)
        report["requested_shape_checks"] = {
            "60_stories": len(stories) == 60,
            "360_turns": report["counts"]["turns"] == 360,
            "30_tasks": len(tasks) == 30,
            "240_items_and_ratings": report["counts"]["items"]
            == report["counts"]["ratings"]
            == 240,
            "6_stories_per_10_scenarios": len(report["scenarios"]) == 10
            and set(report["scenarios"].values()) == {6},
            "6_turns_per_story": all(len(s["turns"]) == 6 for s in stories),
            "2_or_3_sessions": all(
                2 <= len({t["session_id"] for t in s["turns"]}) <= 3 for s in stories
            ),
            "8_items_per_task": all(len(t["items"]) == len(t["item_scores"]) == 8 for t in tasks),
        }
        manifest = dict(
            schema_version=1,
            api_calls=0,
            raw_inputs_unchanged=True,
            source_origin=(
                "user-provided external AI-generated synthetic data; "
                "independent semantic labels unverified"
            ),
            hashes={
                "stories.jsonl": sha256(raw_stories).hexdigest(),
                "recommendation_tasks.jsonl": sha256(raw_tasks).hexdigest(),
            },
            source_hashes={
                p.name: sha256(p.read_bytes()).hexdigest()
                for p in (
                    Path(__file__),
                    Path(__file__).parents[1] / "src/memrecagent/external_evaluation.py",
                )
            },
            split_assigned=False,
        )
        args.output_dir.mkdir(parents=True, exist_ok=False)
        (args.output_dir / "stories.jsonl").write_bytes(raw_stories)
        (args.output_dir / "recommendation_tasks.jsonl").write_bytes(raw_tasks)
        (args.output_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
        )
        (args.output_dir / "report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n"
        )
    except (ValueError, TypeError, KeyError, OSError, RecursionError):
        print(json.dumps({"error": "invalid_external_input"}), file=sys.stderr)
        return 2
    print(
        json.dumps(
            dict(
                output_dir=str(args.output_dir),
                counts=report["counts"],
                structural_valid=report["structural_valid"],
                human_review_required=report["human_review_required"],
                issues=len(report["issues"]),
                api_calls=0,
            ),
            ensure_ascii=False,
        )
    )
    return 0 if report["structural_valid"] and not report["human_review_required"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
