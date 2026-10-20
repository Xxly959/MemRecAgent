"""Compare a fixed cold-user popularity fallback using saved model predictions."""

import argparse
import json
import platform
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from uuid import uuid4

import torch
from evaluate_movielens import capture_source, save_json

from memrecagent.fallback import recommend_with_fallback
from memrecagent.metrics import evaluate_users

ROOT = Path(__file__).resolve().parents[1]
HYBRID = "bpr_cold_popularity"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="Saved four-model run")
    parser.add_argument(
        "--output-root", type=Path, default=ROOT / "artifacts/movielens_cold_fallback"
    )
    args = parser.parse_args()
    try:
        names = ["report.json", "predictions.json", "bpr_validation.pt", "bpr_test.pt"]
        source_bytes = {name: (args.run / name).read_bytes() for name in names}
        source_report = json.loads(source_bytes["report.json"])
        source_predictions = json.loads(source_bytes["predictions.json"])
        k = source_report["config"]["k"]
        stages, predictions, subgroups = {}, {}, {}
        for name in ["validation", "test"]:
            stage = source_predictions[name]
            # Each stage uses its own fitting vocabulary; final refit may add users.
            checkpoint = torch.load(
                args.run / f"bpr_{name}.pt", weights_only=True, map_location="cpu"
            )
            history_users = set(checkpoint["users"])
            bpr, popular = stage["predictions"]["bpr"], stage["predictions"]["popularity"]
            if bpr.keys() != popular.keys():
                raise ValueError("BPR and popularity must have the same users")
            routes = {u: "bpr" if u in history_users else "popularity" for u in sorted(bpr)}
            combined = {
                u: recommend_with_fallback(u, history_users, bpr[u], popular[u], k)
                for u in sorted(bpr)
            }
            rankings = {**stage["predictions"], HYBRID: combined}
            # Future labels enter only after strategy selection and list construction.
            relevant = {u: set(items) for u, items in stage["relevant"].items()}
            metrics = {
                model: asdict(evaluate_users(ranks, relevant, k))
                for model, ranks in rankings.items()
            }
            if "stages" in source_report:
                for model in stage["predictions"]:
                    if metrics[model] != source_report["stages"][name]["metrics"][model]:
                        raise ValueError("Frozen report and predictions disagree")
            if "users" in stage:
                for u in routes:
                    if stage["users"][u]["cold_user"] != (u not in history_users):
                        raise ValueError("Saved cold-user flags and fitting vocabulary disagree")
            cold = {u for u in routes if u not in history_users}
            stages[name] = {
                "metrics": metrics,
                "routing": {
                    "n_users": len(routes),
                    "n_history_users": len(history_users),
                    "n_popularity_routes": len(cold),
                    "n_bpr_routes": len(routes) - len(cold),
                    "n_popularity_routes_with_relevant": sum(bool(relevant[u]) for u in cold),
                },
            }
            predictions[name] = {
                "predictions": rankings,
                "relevant": stage["relevant"],
                "routes": routes,
            }
            subgroups[name] = {}
            for group, is_cold in [("warm", False), ("cold", True)]:
                answers = {u: items for u, items in relevant.items() if (u in cold) == is_cold}
                subgroups[name][group] = {
                    model: asdict(evaluate_users({u: ranks[u] for u in answers}, answers, k))
                    for model, ranks in rankings.items()
                }
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        parser.error(str(exc))

    created = datetime.now(UTC)
    run = args.output_root / (created.strftime("%Y%m%dT%H%M%S%fZ-") + uuid4().hex[:8])
    run.mkdir(parents=True, exist_ok=False)
    code = capture_source(run)
    save_json(run / "predictions.json", predictions)
    report = {
        "created_at_utc": created.isoformat(),
        "protocol": {
            "rule": "absent from fitting user vocabulary -> popularity; otherwise original BPR",
            "k": k,
            "model_retraining": False,
            "new_epoch_selection": False,
            "candidate_filtering": (
                "reuse original filtered full-catalog rankings; no filling or score cutoff"
            ),
            "routing_inputs": "stage-specific checkpoint user vocabulary and user ID only",
            "source_bpr_selection": source_report.get("selection", {}).get("bpr"),
            "test_reused": source_report["config"].get("test_reused", True),
            "scope": (
                "fixed exploratory routing comparison on already-viewed data; not a new blind test"
            ),
            "history_without_positive": "keeps original BPR, even if its user vector was inactive",
        },
        "source": {
            "run": str(args.run.resolve()),
            "sha256": {name: sha256(value).hexdigest() for name, value in source_bytes.items()},
        },
        "stages": stages,
        "subgroups": subgroups,
        "code": code,
        "environment": {
            "python": sys.version,
            "torch": torch.__version__,
            "platform": platform.platform(),
        },
    }
    save_json(run / "report.json", report)
    print(f"Run directory: {run.resolve()}")
    for stage, result in stages.items():
        print(f"\n{stage} routing:", result["routing"])
        for model, metrics in result["metrics"].items():
            print(model, metrics)
    print("Frozen weights and epoch selection; only cold-user routing changed.")
    print(f"Report: {run / 'report.json'}")


if __name__ == "__main__":
    main()
