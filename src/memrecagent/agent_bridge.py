"""绑定商品服务的只读桥接；模型输入不携带身份或目录路径。"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import BinaryIO, Protocol

from memrecagent.product_bundle import load_bundle
from memrecagent.product_service import ProductService

MAX_FRAME_BYTES = 65536
MAX_RECEIPTS = 256
_ID = re.compile(r"[A-Za-z0-9_.:-]{1,128}")


def _error(code: str, request_id: str | None = None) -> dict:
    return {"version": 1, "id": request_id, "ok": False, "error": {"code": code}}


def _json(value: object) -> str:
    text = json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)
    text.encode("utf-8")
    return text


def _valid_health(value: object) -> bool:
    if not isinstance(value, dict) or set(value) != {
        "status",
        "bundle_id",
        "as_of_ms",
        "n_catalog_items",
        "strategies",
    }:
        return False
    return (
        value["status"] == "ready"
        and isinstance(value["bundle_id"], str)
        and bool(value["bundle_id"])
        and all(
            type(value[key]) is int and value[key] >= 0 for key in ("as_of_ms", "n_catalog_items")
        )
        and isinstance(value["strategies"], list)
        and bool(value["strategies"])
        and all(isinstance(item, str) and bool(item) for item in value["strategies"])
    )


class CatalogBridge:
    """成功回执最多保留256个；只保证当前实例内的串行重放。"""

    def __init__(self, service: ProductService) -> None:
        self._service = service
        self._receipts: dict[str, tuple[str, str]] = {}

    def handle(self, value: object) -> dict:
        """严格校验信封后执行真实health，并返回独立JSON快照。"""
        request_id = None
        if isinstance(value, dict):
            candidate = value.get("id")
            if isinstance(candidate, str) and _ID.fullmatch(candidate):
                request_id = candidate
        if (
            not isinstance(value, dict)
            or set(value) != {"version", "id", "method", "params"}
            or type(value["version"]) is not int
            or value["version"] != 1
            or request_id is None
            or not isinstance(value["method"], str)
            or not value["method"]
            or not isinstance(value["params"], dict)
        ):
            return _error("invalid_request", request_id)
        try:
            fingerprint = _json(value)
        except (ValueError, TypeError, UnicodeError, RecursionError):
            return _error("invalid_request", request_id)
        previous = self._receipts.get(request_id)
        if previous is not None:
            if previous[0] != fingerprint:
                return _error("id_conflict", request_id)
            return json.loads(previous[1])
        if value["method"] != "catalog_status":
            return _error("unknown_method", request_id)
        if value["params"]:
            return _error("invalid_request", request_id)
        if len(self._receipts) >= MAX_RECEIPTS:
            return _error("session_full", request_id)
        try:
            result = self._service.health()
            if not _valid_health(result):
                return _error("service_error", request_id)
            snapshot = _json({"version": 1, "id": request_id, "ok": True, "result": result})
        except Exception:
            # 后端异常不携带到协议，失败不占用成功回执容量。
            return _error("service_error", request_id)
        self._receipts[request_id] = (fingerprint, snapshot)
        return json.loads(snapshot)


def _unique_object(pairs: list[tuple[str, object]]) -> dict:
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("duplicate_key")
        value[key] = item
    return value


def _reject_constant(value: str) -> None:
    raise ValueError("non_finite")


class JsonlHandler(Protocol):
    """共享有界帧读取；具体目录/会话协议各自核验请求。"""

    def handle(self, value: object) -> dict: ...


def serve_bridge(bridge: JsonlHandler, input: BinaryIO, output: BinaryIO) -> int:
    """串行读取有界UTF8帧，每个响应立即flush；坏帧不回显原文。"""
    while True:
        # 第一个确定超限的字节即拒绝，不等待发送方补LF或关闭stdin。
        raw = input.readline(MAX_FRAME_BYTES + 1)
        if not raw:
            return 0
        if len(raw) == MAX_FRAME_BYTES + 1 and raw.endswith(b"\r"):
            # 只有这个位置的CR仍可能属于合法的最大帧CRLF。
            raw += input.read(1)
        complete = raw.endswith(b"\n")
        payload = raw[:-1] if complete else raw
        if payload.endswith(b"\r"):
            payload = payload[:-1]
        fatal = False
        if len(payload) > MAX_FRAME_BYTES:
            response = _error("frame_too_large")
            fatal = True
        elif not complete:
            response = _error("frame_truncated")
            fatal = True
        else:
            try:
                value = json.loads(
                    payload.decode("utf-8"),
                    object_pairs_hook=_unique_object,
                    parse_constant=_reject_constant,
                )
            except (ValueError, UnicodeError, RecursionError):
                response = _error("invalid_json")
            else:
                response = bridge.handle(value)
        output.write((_json(response) + "\n").encode("utf-8"))
        output.flush()
        if fatal:
            return 2


def main(argv: list[str] | None = None) -> int:
    """目录只接受可信宿主CLI绑定，启动失败不泄露路径或异常。"""
    parser = argparse.ArgumentParser(description="只读商品目录JSONL桥接")
    parser.add_argument("--bundle", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        bridge = CatalogBridge(load_bundle(args.bundle))
    except Exception:
        print("bundle_load_failed", file=sys.stderr)
        return 2
    return serve_bridge(bridge, sys.stdin.buffer, sys.stdout.buffer)


if __name__ == "__main__":
    raise SystemExit(main())
