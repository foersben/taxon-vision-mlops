# SPDX-FileCopyrightText: 2026 Benjamin Förster
# SPDX-License-Identifier: MIT
"""Unit tests for prediction telemetry persistence and FastAPI background dispatch.

Why:
    Online distribution drift monitoring and historical telemetry auditing require that every successful inference request deterministically captures prediction metadata and feature embeddings without blocking the HTTP serving thread. These tests verify the contract of bounded in-memory test isolation, ring-buffer eviction, fail-fast database outage detection, and asynchronous FastAPI background task queuing.

How:
    Tests unit behaviour of BoundedInMemoryTelemetryStore (ring-buffer eviction, chronological recency retrieval, flattening invariants, and empty matrix fallbacks), verifies fail-fast gating on offline databases, and tests FastAPI /predict route integration using a test fake.
"""

from __future__ import annotations

import io
import os
import uuid
from collections.abc import Generator
from unittest.mock import MagicMock

import numpy as np
import numpy.typing as npt
import pytest
from fastapi.testclient import TestClient
from PIL import Image
from psycopg_pool import ConnectionPool

from taxon_vision.config import reset_settings
from taxon_vision.monitoring.telemetry import TELEMETRY_WRITE_FAILURES
from taxon_vision.service.api import app
from taxon_vision.service.telemetry_store import (
    BoundedInMemoryTelemetryStore,
    PostgresTelemetryStore,
    PredictionRecord,
    close_telemetry_store,
    get_telemetry_store,
    set_telemetry_store,
)

client = TestClient(app)


class FakeTelemetryStore:
    """In-memory test double implementing TelemetryStore protocol for test assertions."""

    def __init__(self) -> None:
        self.records: list[PredictionRecord] = []

    def record(self, rec: PredictionRecord) -> None:
        self.records.append(rec)

    def recent_embeddings(self, limit: int) -> npt.NDArray[np.float32]:
        embs = [r.embedding.flatten() for r in reversed(self.records) if r.embedding is not None]
        if not embs:
            return np.empty((0, 0), dtype=np.float32)
        return np.ascontiguousarray(embs[:limit], dtype=np.float32)


@pytest.fixture(autouse=True)
def clean_telemetry_store() -> Generator[None]:
    """Ensure clean global telemetry store state before and after each test execution."""
    set_telemetry_store(None)
    reset_settings()
    yield
    close_telemetry_store()
    reset_settings()


def _make_dummy_record(
    obs_id: str,
    top1: str = "Turdus merula",
    confidence: float = 0.95,
    embedding: np.ndarray | None = None,
) -> PredictionRecord:
    """Construct a synthetic PredictionRecord instance for unit tests."""
    return PredictionRecord(
        observation_id=obs_id,
        extractor="dinov2_small",
        model_version="v0.1.0",
        top1_taxon=top1,
        top1_confidence=confidence,
        conformal_set=[top1],
        ood_energy=-15.2,
        is_ood=False,
        requires_review=False,
        latency_ms=12.4,
        embedding=embedding,
    )


def test_bounded_telemetry_store_empty() -> None:
    """Verify that an empty store returns an empty 2D array of shape (0, 0)."""
    store = BoundedInMemoryTelemetryStore(maxlen=10)
    embs = store.recent_embeddings(limit=10)
    assert isinstance(embs, np.ndarray)
    assert embs.shape == (0, 0)
    assert embs.dtype == np.float32


def test_bounded_telemetry_store_record_and_recency() -> None:
    """Verify reverse chronological retrieval and filtering of null embeddings."""
    store = BoundedInMemoryTelemetryStore(maxlen=10)

    emb1 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    emb2 = np.array([4.0, 5.0, 6.0], dtype=np.float32)

    rec1 = _make_dummy_record("obs-1", embedding=emb1)
    rec_null = _make_dummy_record("obs-null", embedding=None)
    rec2 = _make_dummy_record("obs-2", embedding=emb2)

    store.record(rec1)
    store.record(rec_null)
    store.record(rec2)

    # limit=1 should return the newest non-null record (emb2)
    latest = store.recent_embeddings(limit=1)
    assert latest.shape == (1, 3)
    np.testing.assert_allclose(latest[0], emb2)

    # limit=5 should return [emb2, emb1] in reverse ingestion order
    all_recent = store.recent_embeddings(limit=5)
    assert all_recent.shape == (2, 3)
    np.testing.assert_allclose(all_recent[0], emb2)
    np.testing.assert_allclose(all_recent[1], emb1)


def test_bounded_telemetry_store_ring_buffer_eviction() -> None:
    """Verify that exceeding maxlen evicts the oldest records in O(1) time without memory leaks."""
    store = BoundedInMemoryTelemetryStore(maxlen=2)

    store.record(_make_dummy_record("obs-1"))
    store.record(_make_dummy_record("obs-2"))
    store.record(_make_dummy_record("obs-3"))

    assert len(store.records) == 2
    assert [r.observation_id for r in store.records] == ["obs-2", "obs-3"]


def test_bounded_telemetry_store_flattening_invariance() -> None:
    """Verify that multi-dimensional embeddings (e.g. 1x4) are flattened to 1D vectors."""
    store = BoundedInMemoryTelemetryStore(maxlen=5)
    unflattened = np.array([[0.1, 0.2, 0.3, 0.4]], dtype=np.float32)
    rec = _make_dummy_record("obs-flat", embedding=unflattened)

    store.record(rec)
    matrix = store.recent_embeddings(limit=1)
    assert matrix.shape == (1, 4)
    assert matrix.dtype == np.float32


def test_postgres_telemetry_store_write_failure_increments_counter() -> None:
    """Verify that database write errors increment Prometheus telemetry failure counter."""
    mock_pool = MagicMock()
    mock_pool.connection.side_effect = RuntimeError("Database unreachable")

    store = PostgresTelemetryStore(mock_pool)
    rec = _make_dummy_record("obs-fail")

    before_count = TELEMETRY_WRITE_FAILURES._value.get()
    store.record(rec)
    after_count = TELEMETRY_WRITE_FAILURES._value.get()

    assert after_count == before_count + 1


def test_postgres_telemetry_store_recent_embeddings_raises_on_error() -> None:
    """Verify that recent_embeddings raises RuntimeError on database query failure to prevent silent error masking."""
    mock_pool = MagicMock()
    mock_pool.connection.side_effect = RuntimeError("Connection dropped")

    store = PostgresTelemetryStore(mock_pool)
    with pytest.raises(RuntimeError, match="Failed to retrieve recent embeddings"):
        store.recent_embeddings(limit=10)


def test_get_telemetry_store_fails_fast_when_offline(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify that get_telemetry_store raises RuntimeError when DB is offline and fallback is false."""
    monkeypatch.setenv("TAXON_TELEMETRY_DATABASE_URL", "postgresql://invalid-host.local:5432/bad_db")
    monkeypatch.setenv("TAXON_TELEMETRY_ALLOW_IN_MEMORY_FALLBACK", "false")
    reset_settings()
    close_telemetry_store()

    with pytest.raises(RuntimeError, match="Failed to connect to PostgreSQL"):
        get_telemetry_store()


def test_get_telemetry_store_activates_bounded_fallback_when_gated(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify that BoundedInMemoryTelemetryStore activates when fallback is explicitly enabled."""
    monkeypatch.setenv("TAXON_TELEMETRY_DATABASE_URL", "postgresql://invalid-host.local:5432/bad_db")
    monkeypatch.setenv("TAXON_TELEMETRY_ALLOW_IN_MEMORY_FALLBACK", "true")
    reset_settings()
    close_telemetry_store()

    store = get_telemetry_store()
    assert isinstance(store, BoundedInMemoryTelemetryStore)


def test_predict_endpoint_queues_background_telemetry_on_success() -> None:
    """Verify that successful /predict invocations dispatch a PredictionRecord to telemetry store."""
    mock_store = FakeTelemetryStore()
    set_telemetry_store(mock_store)

    buf = io.BytesIO()
    Image.new("RGB", (32, 32), color="purple").save(buf, format="JPEG")
    buf.seek(0)

    resp = client.post("/predict", files={"file": ("test.jpg", buf, "image/jpeg")})
    assert resp.status_code == 200

    # Starlette TestClient runs background tasks before returning response
    assert len(mock_store.records) == 1
    rec = mock_store.records[0]
    assert rec.top1_taxon is not None
    assert rec.latency_ms > 0
    assert rec.conformal_set is not None
    assert rec.observation_id is not None
    assert len(rec.observation_id) == 36  # UUID string format


def test_predict_endpoint_zero_telemetry_on_invalid_input() -> None:
    """Verify that rejected uploads (HTTP 422) do not enqueue telemetry records."""
    mock_store = FakeTelemetryStore()
    set_telemetry_store(mock_store)

    # Empty file payload triggers 422
    empty_buf = io.BytesIO(b"")
    resp = client.post("/predict", files={"file": ("empty.jpg", empty_buf, "image/jpeg")})
    assert resp.status_code == 422

    # Invalid corrupted image bytes trigger 422
    corrupt_buf = io.BytesIO(b"not an image")
    resp2 = client.post("/predict", files={"file": ("corrupt.jpg", corrupt_buf, "image/jpeg")})
    assert resp2.status_code == 422

    # Assert no telemetry records were persisted
    assert len(mock_store.records) == 0


@pytest.mark.integration
def test_postgres_telemetry_store_roundtrip() -> None:
    """Verify write and read roundtrip against live PostgreSQL instance when database URL is configured."""
    dsn = os.getenv("TAXON_TELEMETRY_DATABASE_URL") or os.getenv("TAXON_TELEMETRY_DSN")
    if not dsn:
        pytest.skip("TAXON_TELEMETRY_DATABASE_URL not set; skipping live PostgreSQL round-trip integration test.")

    conninfo = dsn.replace("postgresql+psycopg://", "postgresql://", 1)
    with ConnectionPool(conninfo=conninfo, min_size=1, max_size=2) as pool:
        store = PostgresTelemetryStore(pool)
        test_emb = np.random.randn(384).astype(np.float32)
        unique_id = str(uuid.uuid4())
        rec = _make_dummy_record(unique_id, embedding=test_emb)
        store.record(rec)

        recent = store.recent_embeddings(limit=1)
        assert recent.shape == (1, 384)
        np.testing.assert_allclose(recent[0], test_emb, rtol=1e-5, atol=1e-5)
