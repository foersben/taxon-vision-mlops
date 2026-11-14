# SPDX-FileCopyrightText: 2026 Benjamin Förster
# SPDX-License-Identifier: MIT
"""Integration tests for drift evaluation endpoints and Prometheus exposition."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime

import pytest
from fastapi.testclient import TestClient

from taxon_vision.service.api import app
from taxon_vision.service.telemetry_store import (
    BoundedInMemoryTelemetryStore,
    DriftRunRecord,
    set_telemetry_store,
)

client = TestClient(app)


@pytest.fixture(autouse=True)
def isolate_telemetry_store() -> Iterator[BoundedInMemoryTelemetryStore]:
    """Provide an isolated in-memory telemetry store for route integration tests."""
    store = BoundedInMemoryTelemetryStore(maxlen=500)
    set_telemetry_store(store)
    yield store
    set_telemetry_store(None)


def test_get_latest_drift_empty() -> None:
    """Verify GET /api/v1/drift/latest returns None when no drift runs have occurred."""
    resp = client.get("/api/v1/drift/latest")
    assert resp.status_code == 200
    assert resp.json() is None


def test_get_latest_drift_populated(isolate_telemetry_store: BoundedInMemoryTelemetryStore) -> None:
    """Verify GET /api/v1/drift/latest returns the most recent drift run record summary."""
    isolate_telemetry_store.record_drift_run(
        DriftRunRecord(
            id=1,
            created_at=datetime.now(UTC),
            window_size=120,
            w1_distance=0.08,
            w1_threshold=0.15,
            mmd2=0.002,
            mmd2_p_value=0.25,
            drift_detected=False,
            mlflow_run_id="run_12345",
        )
    )

    resp = client.get("/api/v1/drift/latest")
    assert resp.status_code == 200
    data = resp.json()
    assert data["window_size"] == 120
    assert data["w1_distance"] == 0.08
    assert data["w1_threshold"] == 0.15
    assert data["drift_detected"] is False
    assert data["mlflow_run_id"] == "run_12345"


def test_evaluate_drift_unconfigured_token_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify POST /api/v1/drift/evaluate fails closed (HTTP 503) when token secret is unset."""
    monkeypatch.delenv("TAXON_INTERNAL_API_TOKEN", raising=False)
    resp = client.post("/api/v1/drift/evaluate")
    assert resp.status_code == 503


def test_evaluate_drift_unauthorized(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify POST /api/v1/drift/evaluate rejects missing or invalid bearer tokens."""
    monkeypatch.setenv("TAXON_INTERNAL_API_TOKEN", "secret-test-token")

    # Missing token
    resp = client.post("/api/v1/drift/evaluate")
    assert resp.status_code == 401
    assert "Invalid or missing internal authorization token" in resp.json()["detail"]

    # Wrong token
    resp = client.post(
        "/api/v1/drift/evaluate",
        headers={"Authorization": "Bearer wrong-token"},
    )
    assert resp.status_code == 401


def test_evaluate_drift_authorized_success(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify POST /api/v1/drift/evaluate executes successfully with valid bearer token."""
    monkeypatch.setenv("TAXON_INTERNAL_API_TOKEN", "secret-test-token")

    # In test environment, evaluate_recent_drift will return insufficient_data cleanly
    resp = client.post(
        "/api/v1/drift/evaluate",
        headers={"Authorization": "Bearer secret-test-token"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "drift_detected" in data
    assert "status" in data
    assert "w1_distance" in data
    assert "w1_threshold" in data


def test_metrics_scrape_includes_drift_telemetry(isolate_telemetry_store: BoundedInMemoryTelemetryStore) -> None:
    """Verify Prometheus scrape endpoint exposes custom drift metrics when runs exist."""
    isolate_telemetry_store.record_drift_run(
        DriftRunRecord(
            id=42,
            created_at=datetime.now(UTC),
            window_size=500,
            w1_distance=0.18,
            w1_threshold=0.15,
            mmd2=0.015,
            mmd2_p_value=0.003,
            drift_detected=True,
            mlflow_run_id="run_drift_alert",
        )
    )

    resp = client.get("/metrics")
    assert resp.status_code == 200
    text = resp.text
    assert "taxon_drift_w1_distance" in text
    assert "taxon_drift_w1_threshold" in text
    assert "taxon_drift_detected 1.0" in text
    assert "taxon_drift_mmd2_p_value" in text
    assert "taxon_drift_last_run_timestamp_seconds" in text
