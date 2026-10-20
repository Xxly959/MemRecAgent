"""HTTP边界必须校验版本、类型与额外字段，并保留真实推理结果。"""

import pytest
from fastapi.testclient import TestClient
from test_product_service import service

from memrecagent.product_api import create_app


def test_health_and_recommendation_expose_versions_actual_scores_and_snapshot_limits():
    with TestClient(create_app(service())) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["bundle_id"] == "synthetic-v1"
        assert health.json()["n_catalog_items"] == 3
        result = client.post(
            "/recommendations",
            json={
                "user_id": "u",
                "bundle_id": "synthetic-v1",
                "strategy": "mf",
                "k": 2,
            },
        )
        assert result.status_code == 200
        body = result.json()
        assert [item["item_id"] for item in body["items"]] == ["C", "B"]
        assert [item["score"] for item in body["items"]] == [-1.0, -2.0]
        assert body["as_of_ms"] == 1656633600000
        assert body["metadata_snapshot"]["available_at"] is None


@pytest.mark.parametrize(
    "changes",
    [
        {"k": True},
        {"k": 0},
        {"k": 51},
        {"k": 2.0},
        {"user_id": "   "},
        {"strategy": "unknown"},
        {"as_of_ms": 1},
        {"relevant": {"u": ["C"]}},
    ],
)
def test_invalid_types_future_labels_or_arbitrary_time_are_rejected(changes):
    with TestClient(create_app(service())) as client:
        response = client.post(
            "/recommendations",
            json={
                "user_id": "u",
                "bundle_id": "synthetic-v1",
                "strategy": "mf",
                **changes,
            },
        )
        assert response.status_code == 422


def test_version_mismatch_has_no_silent_upgrade_and_cold_strategy_is_explicit():
    with TestClient(create_app(service())) as client:
        wrong = client.post(
            "/recommendations", json={"user_id": "u", "bundle_id": "other", "strategy": "mf"}
        )
        assert wrong.status_code == 409
        cold = client.post(
            "/recommendations",
            json={
                "user_id": "new",
                "bundle_id": "synthetic-v1",
                "strategy": "bpr_cold_popularity",
                "k": 2,
            },
        )
        assert cold.status_code == 200
        assert cold.json()["requested_strategy"] == "bpr_cold_popularity"
        assert cold.json()["used_strategy"] == "popularity"
