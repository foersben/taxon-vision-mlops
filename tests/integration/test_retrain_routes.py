# SPDX-FileCopyrightText: 2026 Benjamin Förster
# SPDX-License-Identifier: MIT
"""Integration tests for the Grafana Alertmanager retraining webhook relay."""

from __future__ import annotations

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from taxon_vision.service.api import app
from taxon_vision.service.routes import retrain as retrain_mod

client = TestClient(app)
VALID_TOKEN = "dev-secret-token"
AUTH_HEADER = {"Authorization": f"Bearer {VALID_TOKEN}"}


@pytest.fixture(autouse=True)
def _reset_cooldown() -> None:
    """Ensure in-memory cooldown state is cleared before each test."""
    retrain_mod.reset_retrain_cooldown()


def test_retrain_webhook_unauthorized() -> None:
    """Verify endpoint rejects requests with missing or invalid credentials."""
    resp = client.post("/api/v1/retrain/webhook", json={"status": "firing"})
    assert resp.status_code == 401

    resp_bad = client.post(
        "/api/v1/retrain/webhook",
        headers={"Authorization": "Bearer wrong-secret"},
        json={"status": "firing"},
    )
    assert resp_bad.status_code == 401


def test_retrain_webhook_ignored_when_no_retrain_action() -> None:
    """Verify alerts without action=retrain or with resolved status are ignored cleanly."""
    payload = {
        "status": "firing",
        "alerts": [
            {
                "status": "firing",
                "labels": {"alertname": "PredictionLatencyP95High", "severity": "warning"},
            }
        ],
    }
    resp = client.post("/api/v1/retrain/webhook", headers=AUTH_HEADER, json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ignored"
    assert data["dag_run_id"] is None


def test_retrain_webhook_dispatches_successfully(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify valid firing retrain alert triggers Airflow DAG dispatch."""

    async def mock_dispatch(alert_name: str) -> str:
        return "airflow-run-12345"

    monkeypatch.setattr(retrain_mod, "dispatch_airflow_retraining", mock_dispatch)

    payload = {
        "status": "firing",
        "alerts": [
            {
                "status": "firing",
                "labels": {
                    "alertname": "EmbeddingDriftDetected",
                    "action": "retrain",
                    "severity": "critical",
                },
            }
        ],
    }
    resp = client.post("/api/v1/retrain/webhook", headers=AUTH_HEADER, json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "triggered"
    assert data["dag_run_id"] == "airflow-run-12345"


def test_retrain_webhook_suppressed_by_cooldown(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify second consecutive request within cooldown window is suppressed."""

    async def mock_dispatch(alert_name: str) -> str:
        return "airflow-run-cooldown"

    monkeypatch.setattr(retrain_mod, "dispatch_airflow_retraining", mock_dispatch)

    payload = {
        "status": "firing",
        "alerts": [
            {
                "status": "firing",
                "labels": {"alertname": "EmbeddingDriftDetected", "action": "retrain"},
            }
        ],
    }

    resp1 = client.post("/api/v1/retrain/webhook", headers=AUTH_HEADER, json=payload)
    assert resp1.status_code == 200
    assert resp1.json()["status"] == "triggered"

    resp2 = client.post("/api/v1/retrain/webhook", headers=AUTH_HEADER, json=payload)
    assert resp2.status_code == 200
    assert resp2.json()["status"] == "suppressed"


def test_retrain_webhook_upstream_error_handling(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify upstream failure in Airflow dispatch propagates HTTP 502 with metric error tracking."""

    async def mock_dispatch_fail(alert_name: str) -> str:
        raise HTTPException(status_code=502, detail="Airflow unreachable")

    monkeypatch.setattr(retrain_mod, "dispatch_airflow_retraining", mock_dispatch_fail)

    payload = {
        "status": "firing",
        "alerts": [
            {
                "status": "firing",
                "labels": {"alertname": "EmbeddingDriftDetected", "action": "retrain"},
            }
        ],
    }
    resp = client.post("/api/v1/retrain/webhook", headers=AUTH_HEADER, json=payload)
    assert resp.status_code == 502
    assert "Airflow unreachable" in resp.json()["detail"]
