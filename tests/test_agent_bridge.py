"""只读桥接的真实领域调用、严格信封与实例内回执。"""

from unittest.mock import Mock

import pytest
from test_product_service import service

from memrecagent.agent_bridge import CatalogBridge


def request(**changes):
    return {"version": 1, "id": "r1", "method": "catalog_status", "params": {}, **changes}


def error(code, request_id="r1"):
    return {"version": 1, "id": request_id, "ok": False, "error": {"code": code}}


def test_real_health_is_called_and_replayed_once():
    engine = service()
    engine.health = Mock(wraps=engine.health)
    bridge = CatalogBridge(engine)
    first = bridge.handle(request())
    assert first == {
        "version": 1,
        "id": "r1",
        "ok": True,
        "result": {
            "status": "ready",
            "bundle_id": "synthetic-v1",
            "as_of_ms": 1656633600000,
            "n_catalog_items": 3,
            "strategies": ["popularity", "mf", "bpr", "bpr_cold_popularity"],
        },
    }
    assert bridge.handle(dict(reversed(list(request().items())))) == first
    assert engine.health.call_count == 1


@pytest.mark.parametrize(
    "value,code,request_id",
    [
        (None, "invalid_request", None),
        ([], "invalid_request", None),
        ({"id": "r1"}, "invalid_request", "r1"),
        (request(version=True), "invalid_request", "r1"),
        (request(version=1.0), "invalid_request", "r1"),
        (request(version=2), "invalid_request", "r1"),
        (request(extra="secret"), "invalid_request", "r1"),
        (request(id=""), "invalid_request", None),
        (request(id="a" * 129), "invalid_request", None),
        (request(id="../private/path"), "invalid_request", None),
        (request(id=True), "invalid_request", None),
        (request(method=""), "invalid_request", "r1"),
        (request(method=1), "invalid_request", "r1"),
        (request(method="delete_memory"), "unknown_method", "r1"),
        (request(params=[]), "invalid_request", "r1"),
        (request(params={"path": "/private/secret"}), "invalid_request", "r1"),
        (request(params={"user_id": "someone"}), "invalid_request", "r1"),
        (request(params={"bad": float("nan")}), "invalid_request", "r1"),
    ],
)
def test_invalid_requests_do_not_call_backend_or_poison_id(value, code, request_id):
    engine = service()
    engine.health = Mock(wraps=engine.health)
    bridge = CatalogBridge(engine)
    assert bridge.handle(value) == error(code, request_id)
    assert engine.health.call_count == 0
    assert bridge.handle(request())["ok"] is True
    assert engine.health.call_count == 1


@pytest.mark.parametrize("changes", [{"method": "other"}, {"params": {"path": "other"}}])
def test_successful_id_conflict_precedes_method_and_param_checks(changes):
    engine = service()
    engine.health = Mock(wraps=engine.health)
    bridge = CatalogBridge(engine)
    bridge.handle(request())
    assert bridge.handle(request(**changes)) == error("id_conflict")
    assert engine.health.call_count == 1


def test_caller_and_backend_mutation_cannot_poison_cached_snapshot():
    engine = service()
    health = engine.health()
    engine.health = Mock(return_value=health)
    bridge = CatalogBridge(engine)
    first = bridge.handle(request())
    first["result"]["strategies"].append("caller mutation")
    health["strategies"].append("backend mutation")
    replay = bridge.handle(request())
    assert replay["result"]["strategies"] == ["popularity", "mf", "bpr", "bpr_cold_popularity"]
    replay["result"]["strategies"].clear()
    assert len(bridge.handle(request())["result"]["strategies"]) == 4


def test_success_capacity_keeps_old_receipts_and_does_not_execute_new_id():
    engine = service()
    engine.health = Mock(wraps=engine.health)
    bridge = CatalogBridge(engine)
    for index in range(256):
        assert bridge.handle(request(id=f"r{index}"))["ok"] is True
    assert bridge.handle(request(id="new")) == error("session_full", "new")
    assert bridge.handle(request(id="r0"))["ok"] is True
    assert engine.health.call_count == 256


def test_backend_failure_is_sanitized_and_does_not_consume_id():
    engine = service()
    real_health = engine.health
    engine.health = Mock(side_effect=RuntimeError("credential /private/path raw-dialogue"))
    bridge = CatalogBridge(engine)
    assert bridge.handle(request()) == error("service_error")
    engine.health = real_health
    assert bridge.handle(request())["ok"] is True


@pytest.mark.parametrize(
    "change",
    [
        {"extra": "secret"},
        {"status": "failed"},
        {"bundle_id": ""},
        {"bundle_id": "\ud800"},
        {"as_of_ms": True},
        {"as_of_ms": -1},
        {"n_catalog_items": 1.0},
        {"n_catalog_items": float("inf")},
        {"strategies": []},
        {"strategies": [""]},
        {"strategies": [1]},
    ],
)
def test_invalid_backend_result_never_becomes_success(change):
    engine = service()
    health = engine.health()
    engine.health = Mock(return_value={**health, **change})
    assert CatalogBridge(engine).handle(request()) == error("service_error")
