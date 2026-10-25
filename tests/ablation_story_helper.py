"""仅固定客户端工程故事：从空库经实际分析/提交，再供真实Pi工具读取。"""

import argparse
import json
from dataclasses import asdict
from datetime import timedelta
from pathlib import Path

from test_memory import T0
from test_memory_interaction_extraction import envelope
from test_product_service import service

from memrecagent.deepseek_client import ModelReply
from memrecagent.memory_analyst import AnalystError
from memrecagent.memory_extraction import Turn
from memrecagent.memory_interaction import ConversationMemorySession
from memrecagent.memory_products import ProductEntityFact
from memrecagent.memory_store import JsonlMemoryStore
from memrecagent.product_bundle import save_bundle


def build(path: Path, *, invalid: bool = False) -> None:
    """固定向量不是训练结果；输入/响应完整留痕，不声称自然语言识别。"""
    path.mkdir(parents=True, exist_ok=True)
    if any(path.iterdir()):
        raise ValueError("工程故事必须使用新空目录")
    base = service()
    base.seen = {"u": set()}
    manifest = save_bundle(base, path / "bundle", provenance={"kind": "synthetic-fixed-vectors"})
    facts = tuple(
        ProductEntityFact(k, "phone", f"品牌{k}", f"synthetic:{k}", T0 - timedelta(days=1))
        for k in base.counts
    )
    (path / "facts.json").write_text(
        json.dumps(
            {
                "bundle_id": manifest["bundle_id"],
                "facts": [asdict(f) | {"available_at": f.available_at.isoformat()} for f in facts],
            },
            ensure_ascii=False,
        )
    )
    store = JsonlMemoryStore(path / "memory")
    host = ConversationMemorySession(store, user_id="u", session_id="S1")
    turn = Turn("T1", "user", "以后长期排除品牌A手机，长期喜欢品牌B手机。", T0, T0)
    raw = envelope()
    raw["intents"] = ["express_preference"]
    raw["candidate"]["assertions"] = [
        dict(
            subject={"kind": "self", "references": []},
            category="phone",
            entity=entity,
            relation=relation,
            scope="long_term",
            evidence=[{"turn_id": "T1", "quote": turn.content}],
        )
        for entity, relation in (("品牌A", "exclude"), ("品牌B", "prefer"))
    ]
    calls = []

    def complete(messages):
        response = "bad-json" if invalid else json.dumps(raw, ensure_ascii=False)
        calls.append(dict(messages=messages, response=response, client="fixed-scripted"))
        return ModelReply(response, "fixed-scripted", None, None, 0.0)

    before = store.snapshot_token("u")
    try:
        preview = host.prepare_user(
            turn, complete=complete, as_of=T0 + timedelta(seconds=1), analysis_mode="analyst"
        )
        host.commit(preview.ticket_id, as_of=T0 + timedelta(seconds=1))
        status = "committed"
    except AnalystError:
        status = "processing_error"
    (path / "history-trace.json").write_text(
        json.dumps(
            dict(
                evidence_kind="fixed_client_engineering",
                actual_api_calls=0,
                initial_memory="empty",
                status=status,
                before=before,
                after=store.snapshot_token("u"),
                calls=calls,
                analysis_calls=[asdict(c) for c in host.last_analysis_calls],
                stored_records=len(store.list_memories("u", as_of=T0 + timedelta(days=1))),
            ),
            ensure_ascii=False,
        )
    )
    for group, memory, algorithm in (
        ("G00", False, False),
        ("G10", True, False),
        ("G01", False, True),
        ("G11", True, True),
    ):
        request = dict(
            user_id="u",
            session_id="S2",
            beneficiary_id="self",
            category="phone",
            as_of=(T0 + timedelta(days=1)).isoformat(),
            k=3,
            strategy="bpr" if algorithm else "popularity",
            use_long_term_memory=memory,
            normalisation="minmax",
        )
        (path / f"request-{group}.json").write_text(json.dumps(request, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    parser.add_argument("--invalid", action="store_true")
    args = parser.parse_args()
    build(args.path, invalid=args.invalid)
