"""外部合成数据的显式政策派生；默认仅计划，--derive保存不可覆盖产物。"""

import argparse
import json
import sys
from collections import Counter
from hashlib import sha256
from pathlib import Path

from audit_external_evaluation import decode

from memrecagent.external_derived import derive_external_suite
from memrecagent.memory_extraction import unique_object


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stories", type=Path, required=True)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--derive", action="store_true")
    args = parser.parse_args()
    try:
        paths = {
            "stories.jsonl": args.stories,
            "tasks.jsonl": args.tasks,
            "policy.json": args.policy,
        }
        raw = {name: path.read_bytes() for name, path in paths.items()}
        if len(raw["policy.json"]) > 512 * 1024:
            raise ValueError("policy size")
        policy = json.loads(raw["policy.json"], object_pairs_hook=unique_object)
        suite = derive_external_suite(
            decode(raw["stories.jsonl"]), decode(raw["tasks.jsonl"]), policy
        )
        summary = dict(
            api_calls=0,
            episodes=len(suite["episodes"]),
            turns=sum(len(e["steps"]) for e in suite["episodes"]),
            tasks=len(suite["tasks"]),
            items=sum(len(t["runtime"]["items"]) for t in suite["tasks"]),
            split_episodes=dict(Counter(e["split"] for e in suite["episodes"])),
            split_tasks=dict(Counter(t["split"] for t in suite["tasks"])),
            reference_status=dict(Counter(t["reference"]["status"] for t in suite["tasks"])),
            changes=len(suite["changes"]),
            behavior_history_available=False,
            algorithm_estimand_identified=False,
            semantic_gold_independently_verified=False,
        )
        if not args.derive:
            print(json.dumps(summary | {"mode": "plan"}, ensure_ascii=False))
            return 0
        if args.output_dir.exists():
            raise ValueError("exists")
        root = Path(__file__).resolve().parents[1]
        sources = sorted((root / "src/memrecagent").glob("*.py")) + [
            Path(__file__).resolve(),
            root / "scripts/audit_external_evaluation.py",
            root / "pyproject.toml",
            root / "uv.lock",
        ]
        source_bytes = {str(path.relative_to(root)): path.read_bytes() for path in sources}
        output = json.dumps(
            suite, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False
        ).encode()
        manifest = dict(
            schema_version=1,
            api_calls=0,
            input_sha256={name: sha256(content).hexdigest() for name, content in raw.items()},
            derived_sha256=sha256(output).hexdigest(),
            source_sha256={
                name: sha256(content).hexdigest() for name, content in source_bytes.items()
            },
            raw_inputs_unchanged=all(
                paths[name].read_bytes() == content for name, content in raw.items()
            ),
        )
        if not manifest["raw_inputs_unchanged"]:
            raise ValueError("changed")
        args.output_dir.mkdir(parents=True, exist_ok=False)
        for name, content in raw.items():
            target = args.output_dir / "raw" / name
            target.parent.mkdir(exist_ok=True)
            target.write_bytes(content)
        for name, content in source_bytes.items():
            target = args.output_dir / "source" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
        (args.output_dir / "derived-suite.json").write_bytes(output)
        for name, value in (("manifest.json", manifest), ("report.json", summary)):
            (args.output_dir / name).write_text(json.dumps(value, ensure_ascii=False, indent=2))
        print(json.dumps(summary | {"mode": "derived"}, ensure_ascii=False))
        return 0
    except (ValueError, OSError, TypeError, KeyError, AttributeError, RecursionError):
        print(json.dumps({"error": "invalid_derivation_input"}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
