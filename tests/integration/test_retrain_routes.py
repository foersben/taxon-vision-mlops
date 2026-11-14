# SPDX-FileCopyrightText: 2026 Benjamin Förster
# SPDX-License-Identifier: MIT
"""Integration tests for the Grafana Alertmanager retraining webhook relay."""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from taxon_vision.service.api import app
from taxon_vision.service.routes import retrain as retrain_mod

client = TestClient(app)
VALID_TOKEN = "dev-secret-token"
AUTH_HEADER = {"Authorization": f"Bearer {VALID_TOKEN}"}


@pytest.fixture(autouse=True)
def _reset_cooldown(monkeypatch: pytest.MonkeyPatch) -> None:
    """Configure webhook secret explicitly and clear in-memory cooldown state before each test."""
    monkeypatch.setenv("TAXON_RETRAIN_WEBHOOK_SECRET", VALID_TOKEN)
    retrain_mod.reset_retrain_cooldown()


def test_retrain_webhook_unconfigured_secret_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify webhook rejects calls with HTTP 503 when TAXON_RETRAIN_WEBHOOK_SECRET is unset."""
    monkeypatch.delenv("TAXON_RETRAIN_WEBHOOK_SECRET", raising=False)
    resp = client.post("/api/v1/retrain/webhook", headers=AUTH_HEADER, json={"status": "firing"})
    assert resp.status_code == 503


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


def _patch_airflow_env_and_transport(
    monkeypatch: pytest.MonkeyPatch,
    handler: object,
) -> None:
    """Configure deterministic Airflow env vars and bind a MockTransport AsyncClient."""
    monkeypatch.setenv("TAXON_AIRFLOW_SIMULATION", "false")
    monkeypatch.setenv("TAXON_AIRFLOW_URL", "http://airflow-test:8080")
    monkeypatch.setenv("TAXON_AIRFLOW_USER", "admin")
    monkeypatch.setenv("TAXON_AIRFLOW_PASSWORD", "secret")

    real_async_client = httpx.AsyncClient

    def _client_factory(*args: object, **kwargs: object) -> httpx.AsyncClient:
        return real_async_client(transport=httpx.MockTransport(handler))  # type: ignore[arg-type]

    monkeypatch.setattr(retrain_mod.httpx, "AsyncClient", _client_factory)


def test_dispatch_airflow_retraining_v2_jwt(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify Airflow 3 dispatch exchanges credentials for JWT and sends logical_date=None."""
    recorded: list[tuple[str, dict[str, object], str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        auth_hdr = request.headers.get("Authorization", "")
        recorded.append((request.url.path, body, auth_hdr))
        if request.url.path == "/auth/token":
            assert body == {"username": "admin", "password": "secret"}
            return httpx.Response(201, json={"access_token": "jwt-abc-123"})
        if request.url.path == "/api/v2/dags/model_retraining/dagRuns":
            assert auth_hdr == "Bearer jwt-abc-123"
            assert body == {
                "logical_date": None,
                "conf": {"trigger_source": "grafana", "alert": "DriftHigh"},
            }
            return httpx.Response(200, json={"dag_run_id": "manual__2026-11-11"})
        return httpx.Response(404)

    _patch_airflow_env_and_transport(monkeypatch, handler)
    run_id = asyncio.run(retrain_mod.dispatch_airflow_retraining("DriftHigh"))
    assert run_id == "manual__2026-11-11"
    assert [path for path, _, _ in recorded] == [
        "/auth/token",
        "/api/v2/dags/model_retraining/dagRuns",
    ]


def test_dispatch_airflow_retraining_v1_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify dispatcher falls back cleanly to legacy Airflow v1 API when /auth/token is absent."""
    paths_called: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        paths_called.append(request.url.path)
        if request.url.path == "/auth/token":
            return httpx.Response(404)
        if request.url.path == "/api/v1/dags/model_retraining/dagRuns":
            assert request.headers.get("Authorization", "").startswith("Basic ")
            body = json.loads(request.content.decode("utf-8"))
            assert body == {"conf": {"trigger_source": "grafana", "alert": "DriftHigh"}}
            return httpx.Response(200, json={"dag_run_id": "legacy__run_1"})
        return httpx.Response(500)

    _patch_airflow_env_and_transport(monkeypatch, handler)
    run_id = asyncio.run(retrain_mod.dispatch_airflow_retraining("DriftHigh"))
    assert run_id == "legacy__run_1"
    assert paths_called == ["/auth/token", "/api/v1/dags/model_retraining/dagRuns"]


def test_dispatch_airflow_retraining_auth_failure_fails_fast(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify HTTP 401 from Airflow /auth/token raises HTTP 502 without hitting legacy v1."""
    paths_called: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        paths_called.append(request.url.path)
        if request.url.path == "/auth/token":
            return httpx.Response(401, json={"detail": "Invalid credentials"})
        return httpx.Response(500)

    _patch_airflow_env_and_transport(monkeypatch, handler)
    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(retrain_mod.dispatch_airflow_retraining("DriftHigh"))
    assert exc_info.value.status_code == 502
    assert paths_called == ["/auth/token"]


def test_dispatch_airflow_retraining_missing_credentials_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify dispatcher raises HTTP 503 rather than falling back to default credentials when unset."""
    monkeypatch.setenv("TAXON_AIRFLOW_SIMULATION", "false")
    monkeypatch.delenv("TAXON_AIRFLOW_USER", raising=False)
    monkeypatch.delenv("TAXON_AIRFLOW_PASSWORD", raising=False)
    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(retrain_mod.dispatch_airflow_retraining("DriftHigh"))
    assert exc_info.value.status_code == 503
