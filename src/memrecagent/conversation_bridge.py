"""持续会话桥接：可信宿主原始轮次→MemoryAnalyst→受控提交→排名。"""

import argparse
import json
import os
import sys
from copy import deepcopy
from datetime import datetime
from hashlib import sha256
from pathlib import Path
from uuid import UUID

from memrecagent.agent_bridge import _ID, _error, _json, serve_bridge
from memrecagent.conversation_products import ConversationProductSession
from memrecagent.deepseek_client import DeepSeekClient
from memrecagent.memory_analyst import AnalystError
from memrecagent.memory_commit import CommitError
from memrecagent.memory_extraction import Turn
from memrecagent.memory_product_tool import _read
from memrecagent.memory_products import decode_product_entity_fact
from memrecagent.memory_store import JsonlMemoryStore
from memrecagent.product_bundle import load_bundle


class ConversationBridge:
    """只保证本实例串行回执；失败也缓存，避免重放重复模型消费或提交。"""

    def __init__(self, host: ConversationProductSession, *, complete) -> None:
        self._host, self._complete = host, complete
        self._receipts: dict[str, tuple[str, str]] = {}
        self._comparison: dict | None = None
        self._receipt_state = host.state_token()

    @staticmethod
    def _expired(response: dict, request_id: str) -> dict:
        """只留安全成本元信息；旧原文/商品/记录ID不再作为回执返回。"""
        data = response.get("result", {}) if response.get("ok") else response.get("error", {})
        result = _error("conversation_state", request_id)
        result["error"]["memory_commit_status"] = "unknown"
        calls = data.get("analysis_calls")
        if isinstance(calls, list) and len(calls) <= 2:
            fields = {
                "stage",
                "status",
                "model",
                "fingerprint",
                "usage",
                "latency_seconds",
                "error_code",
            }
            result["error"]["analysis_calls"] = [
                {key: deepcopy(value) for key, value in call.items() if key in fields}
                for call in calls
                if isinstance(call, dict)
            ]
        else:
            result["error"]["analysis_status"] = "unknown"
        return result

    def _expire_receipts(self) -> None:
        for identity, (fingerprint, snapshot) in self._receipts.items():
            self._receipts[identity] = (
                fingerprint,
                _json(self._expired(json.loads(snapshot), identity)),
            )

    def _sync_receipts(self) -> bool:
        try:
            current = self._host.refresh_read_state()
        except Exception:
            self._expire_receipts()
            self._comparison = None
            self._receipt_state = None
            return False
        if self._receipt_state is not None and current != self._receipt_state:
            self._expire_receipts()
            if (
                self._comparison is not None
                and self._comparison["result"]["state_token"] != current
            ):
                self._comparison = None
        self._receipt_state = current
        return True

    def _checked_comparison(self, token: str | None = None) -> dict:
        cached = self._comparison
        if cached is None:
            raise CommitError("stale_recommendation")
        try:
            current = self._host.recommendation_token()
            if (
                current != cached["request_token"]
                or cached["result"]["state_token"] != self._host.state_token()
            ):
                raise CommitError("stale_recommendation")
        except CommitError:
            self._comparison = None
            raise
        if token is not None and token != current:
            raise CommitError("stale_recommendation")
        return cached

    def handle(self, value: object) -> dict:
        request_id = value.get("id") if isinstance(value, dict) else None
        if not isinstance(request_id, str) or not _ID.fullmatch(request_id):
            return _error("invalid_request")
        if (
            set(value) != {"version", "id", "method", "params"}
            or type(value["version"]) is not int
            or value["version"] != 1
            or not isinstance(value["method"], str)
            or not isinstance(value["params"], dict)
        ):
            return _error("invalid_request", request_id)
        try:
            fingerprint = sha256(_json(value).encode()).hexdigest()
        except Exception:
            return _error("invalid_request", request_id)
        fresh = self._sync_receipts()
        if request_id in self._receipts:
            old, response = self._receipts[request_id]
            if old != fingerprint:
                return _error("id_conflict", request_id)
            cached = json.loads(response)
            if (
                value["method"]
                in {"comparison_binding", "rank_current", "check_comparison", "feedback_display"}
                and cached["ok"]
            ):
                token = (
                    cached["result"]["request_token"]
                    if value["method"] == "comparison_binding"
                    else value["params"]["request_token"]
                )
                try:
                    if value["method"] == "feedback_display":
                        if self._host.recommendation_token() != token:
                            raise CommitError("stale")
                    else:
                        self._checked_comparison(token)
                except CommitError:
                    return _error("conversation_state", request_id)
            self._sync_receipts()
            # 解码缓存期间也可能发生删除；只发布再次核验后仍保留的回执。
            current = self._receipts[request_id][1]
            return cached if current == response else json.loads(current)
        if len(self._receipts) >= 256:
            return _error("session_full", request_id)
        if not fresh:
            response = _error("conversation_state", request_id)
            response["error"].update(analysis_calls=[], new_memory_ids=[])
            self._receipts[request_id] = (fingerprint, _json(response))
            return response
        params, method = value["params"], value["method"]
        fields = {"turn_id", "content", "occurred_at", "available_at"}
        needed = fields | {"as_of", "strategy", "k"} if method == "user_turn" else fields
        if method == "feedback_display":
            needed = {"request_token", "displayed_at"}
        if method == "feedback_submit":
            needed = {"display_token", "index", "kind", "scope", "event_id", "at"}
        if method == "feedback_undo":
            needed = {"event_id", "action_id", "at"}
        if method == "check_state":
            needed = {"state_token"}
        if method == "comparison_binding":
            needed = set()
        if method == "rank_current":
            needed = {"request_token", "use_long_term_memory", "strategy"}
        if method == "check_comparison":
            needed = {"request_token", "state_token"}
        if method not in {
            "user_turn",
            "assistant_turn",
            "check_state",
            "comparison_binding",
            "rank_current",
            "check_comparison",
            "feedback_display",
            "feedback_submit",
            "feedback_undo",
        }:
            return _error("unknown_method", request_id)
        if set(params) != needed:
            return _error("invalid_request", request_id)
        if method in {"rank_current", "check_comparison"}:
            token = params["request_token"]
            if not isinstance(token, str) or len(token) != 32:
                return _error("invalid_request", request_id)
            try:
                if UUID(hex=token).hex != token:
                    raise ValueError
            except ValueError:
                return _error("invalid_request", request_id)
        if method == "rank_current" and (
            type(params["use_long_term_memory"]) is not bool
            or not isinstance(params["strategy"], str)
        ):
            return _error("invalid_request", request_id)
        analysis_started = False
        feedback_started = False
        try:
            if method == "feedback_display":
                result = self._host.register_feedback_display(
                    params["request_token"],
                    displayed_at=datetime.fromisoformat(params["displayed_at"]),
                )
            elif method in {"feedback_submit", "feedback_undo"}:
                self._comparison = None
                feedback_started = True
                if method == "feedback_submit":
                    result = self._host.submit_feedback(
                        params["display_token"],
                        index=params["index"],
                        kind=params["kind"],
                        scope=params["scope"],
                        event_id=params["event_id"],
                        at=datetime.fromisoformat(params["at"]),
                    )
                else:
                    result = self._host.undo_feedback(
                        params["event_id"],
                        action_id=params["action_id"],
                        at=datetime.fromisoformat(params["at"]),
                    )
            elif method == "comparison_binding":
                cached = self._checked_comparison()
                result = dict(
                    request_token=cached["request_token"],
                    prompt=cached["prompt"],
                    state_token=cached["result"]["state_token"],
                )
            elif method == "check_comparison":
                cached = self._checked_comparison(params["request_token"])
                if params["state_token"] != cached["result"]["state_token"]:
                    raise CommitError("stale")
                result = {"valid": True}
            elif method == "rank_current":
                cached = self._checked_comparison(params["request_token"])
                ranking = self._host.rank_current(
                    use_long_term_memory=params["use_long_term_memory"],
                    strategy=params["strategy"],
                    request_token=params["request_token"],
                )
                self._checked_comparison(params["request_token"])
                if ranking["status"] != "ready":
                    raise CommitError("ranking_unavailable")
                result = deepcopy(cached["result"]) | dict(
                    ranking=ranking,
                    state_token=ranking["state_token"],
                    analysis_calls=[],
                    new_memory_ids=[],
                )
            elif method == "check_state":
                if (
                    not isinstance(params["state_token"], str)
                    or params["state_token"] != self._host.state_token()
                ):
                    raise CommitError("stale")
                result = {"valid": True}
            else:
                if not isinstance(params["content"], str) or len(params["content"]) > 8000:
                    raise ValueError("输入无效")
                turn = Turn(
                    params["turn_id"],
                    "user" if method == "user_turn" else "assistant",
                    params["content"],
                    datetime.fromisoformat(params["occurred_at"]),
                    datetime.fromisoformat(params["available_at"]),
                )
                if method == "assistant_turn":
                    self._host.observe_assistant(turn)
                    result = {"observed": True}
                else:
                    analysis_started = True
                    self._comparison = None
                    result = self._host.process_user(
                        turn,
                        complete=self._complete,
                        as_of=datetime.fromisoformat(params["as_of"]),
                        strategy=params["strategy"],
                        k=params["k"],
                    )
                    if result["state_token"] != self._host.state_token():
                        raise CommitError("stale")
                    if result["response_action"] == "recommend":
                        self._comparison = dict(
                            request_token=self._host.recommendation_token(),
                            prompt=turn.content,
                            result=deepcopy(result),
                        )
            response = {"version": 1, "id": request_id, "ok": True, "result": result}
        except AnalystError:
            response = _error("analysis_failed", request_id)
        except CommitError:
            response = _error("conversation_state", request_id)
        except Exception:
            response = _error("conversation_failed", request_id)
        if not response["ok"] and feedback_started:
            response["error"]["feedback_commit_status"] = "unknown"
        if not response["ok"] and analysis_started:
            response["error"]["analysis_calls"] = self._host.last_analysis_calls
            response["error"]["new_memory_ids"] = self._host.last_new_memory_ids
        if not self._sync_receipts():
            response = self._expired(response, request_id)
        snapshot = _json(response)
        if len(snapshot.encode()) > 65536:
            oversized = _error("response_too_large", request_id)
            oversized["error"]["analysis_calls"] = (
                self._host.last_analysis_calls if analysis_started else []
            )
            oversized["error"]["new_memory_ids"] = (
                self._host.last_new_memory_ids if analysis_started else []
            )
            snapshot = _json(oversized)
            if len(snapshot.encode()) > 65536:
                oversized = _error("response_too_large", request_id)
                oversized["error"]["analysis_status"] = "unknown"
                oversized["error"]["memory_commit_status"] = "unknown"
                snapshot = _json(oversized)
        self._receipts[request_id] = (fingerprint, snapshot)
        return json.loads(snapshot)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", required=True)
    for name in ("bundle", "memory-root", "facts"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    for name in ("user-id", "session-id"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--category", default="phone")
    args = parser.parse_args()
    try:
        client = DeepSeekClient(os.environ.get("DEEPSEEK_API_KEY", ""), timeout=20)
        _, mapping = _read(args.facts, 2 * 1024 * 1024)
        if (
            set(mapping) != {"bundle_id", "facts"}
            or not isinstance(mapping["facts"], list)
            or not 0 < len(mapping["facts"]) <= 5000
        ):
            raise ValueError("映射无效")
        facts = tuple(decode_product_entity_fact(row) for row in mapping["facts"])
        host = ConversationProductSession(
            load_bundle(args.bundle),
            JsonlMemoryStore(args.memory_root),
            facts,
            facts_bundle_id=mapping["bundle_id"],
            user_id=args.user_id,
            session_id=args.session_id,
            category=args.category,
        )
        bridge = ConversationBridge(host, complete=client.complete)
    except Exception:
        sys.stderr.write("conversation_start_failed\n")
        return 2
    return serve_bridge(bridge, sys.stdin.buffer, sys.stdout.buffer)


if __name__ == "__main__":
    raise SystemExit(main())
