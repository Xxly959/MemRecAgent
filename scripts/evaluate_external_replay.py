"""重放已冻结派生语料；默认零调用，live需要环境Key及显式调用上限。"""

import argparse
import json
import os
import sys
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path

from memrecagent.deepseek_client import DeepSeekClient, ProviderError
from memrecagent.external_replay import evaluate_external_replay, resolve_scoring_cutoffs
from memrecagent.memory_extraction import unique_object
from memrecagent.native_replay import load_native_environment


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("dev", "test", "all"), default="dev")
    parser.add_argument(
        "--analysis-mode", choices=("direct", "analyst", "analyst_blind"), default="analyst"
    )
    parser.add_argument("--max-calls", type=int)
    parser.add_argument("--seed", type=int, default=20261024)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--scoring-mode", choices=("fixed", "request"), default="fixed")
    parser.add_argument("--native-environment", type=Path)
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args()
    try:
        raw = args.suite.read_bytes()
        if not 0 < len(raw) <= 16 * 1024 * 1024:
            raise ValueError("size")
        manifest = json.loads(
            (args.suite.parent / "manifest.json").read_bytes(), object_pairs_hook=unique_object
        )
        if sha256(raw).hexdigest() != manifest["derived_sha256"]:
            raise ValueError("suite mismatch")
        suite = json.loads(raw, object_pairs_hook=unique_object)
        tasks = [t for t in suite["tasks"] if args.split == "all" or t["split"] == args.split]
        selected = {t["episode_id"] for t in tasks}
        episodes = [e for e in suite["episodes"] if e["episode_id"] in selected]
        steps = sum(len(e["steps"]) for e in episodes)
        cutoffs = resolve_scoring_cutoffs(tasks, k=args.k, scoring_mode=args.scoring_mode)
        native = (
            load_native_environment(args.native_environment) if args.native_environment else None
        )
        if native is not None:
            if set(native.users) != selected:
                raise ValueError("native profile selection mismatch")
            for episode in episodes:
                for step in episode["steps"]:
                    native.reference_state(step["reference"]["long_term_state_after"])
        summary = dict(
            api_calls=0,
            split=args.split,
            analysis_mode=args.analysis_mode,
            episodes=len(episodes),
            planned_steps=steps,
            planned_cases=len(tasks),
            planned_max_calls=steps * (1 if args.analysis_mode == "direct" else 2),
            algorithm_estimand_identified=False,
            automatic_retries=0,
            native_environment=native is not None,
            active_bpr_bindings=sum(
                user in native.service.models["bpr"].active_users and user in native.service.seen
                for user in native.users.values()
            )
            if native
            else 0,
            scoring_mode=args.scoring_mode,
            scoring_cutoffs=cutoffs,
        )
        if not episodes:
            raise ValueError("empty selection")
        if not args.live:
            print(json.dumps(summary))
            return 0
        key = os.environ.get("DEEPSEEK_API_KEY")
        if not key or args.max_calls is None or args.output_dir.exists():
            raise ValueError("live needs key, explicit budget and fresh output")
        client = DeepSeekClient(key)

        def complete(messages):
            response = client.complete(messages)
            if key in json.dumps(asdict(response), ensure_ascii=False):
                raise ProviderError("response_content")
            return response

        report = evaluate_external_replay(
            suite,
            complete=complete,
            output_dir=args.output_dir,
            analysis_mode=args.analysis_mode,
            split=args.split,
            max_calls=args.max_calls,
            evaluation_kind="synthetic_live",
            k=args.k,
            seed=args.seed,
            scoring_mode=args.scoring_mode,
            native_environment=native,
        )
        print(
            json.dumps(
                {
                    k: report[k]
                    for k in (
                        "planned_steps",
                        "scoring_mode",
                        "scoring_protocol",
                        "scoring_k_distribution",
                        "successful_steps",
                        "planned_cases",
                        "case_errors",
                        "api_calls",
                        "reported_tokens",
                        "calls_without_reported_usage",
                        "calls_without_total_tokens",
                        "reported_usage_totals",
                    )
                }
            )
        )
        return (
            0
            if report["successful_steps"] == report["planned_steps"] and not report["case_errors"]
            else 1
        )
    except (ValueError, TypeError, KeyError, OSError, ProviderError):
        print(json.dumps({"error": "invalid_external_replay_input"}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
