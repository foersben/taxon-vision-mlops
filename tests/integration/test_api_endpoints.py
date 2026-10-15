# SPDX-FileCopyrightText: 2026 Benjamin Förster
# SPDX-License-Identifier: MIT
"""Integration tests for FastAPI REST inference, health, and training endpoints."""

from __future__ import annotations

import io
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from taxon_vision.service.api import app
from taxon_vision.service.routes import predict as predict_mod
from taxon_vision.service.routes.predict import _build_live_prediction

client = TestClient(app)


def test_health_endpoint() -> None:
    """Verify liveness probe returns HTTP 200 with healthy status."""
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "healthy"


def test_health_ready_endpoint() -> None:
    """Verify readiness probe returns HTTP 200 with ready status."""
    resp = client.get("/health/ready")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ready"


def test_predict_endpoint() -> None:
    """Verify /api/v1/predict rejects non-image bytes with HTTP 400."""
    dummy_img = io.BytesIO(b"fake image bytes")
    resp = client.post("/api/v1/predict", files={"file": ("test.jpg", dummy_img, "image/jpeg")})
    assert resp.status_code == 400
    assert "not a decodable image" in resp.json()["detail"]


def test_top_level_predict_endpoint() -> None:
    """Verify /predict top-level route alias matches API response structure for valid image."""
    buf = io.BytesIO()
    Image.new("RGB", (32, 32), color="blue").save(buf, format="JPEG")
    buf.seek(0)
    resp = client.post("/predict", files={"file": ("test.jpg", buf, "image/jpeg")})
    assert resp.status_code == 200
    data = resp.json()
    assert "top_prediction" in data
    assert "conformal_prediction_set" in data


def test_training_endpoint_sync() -> None:
    """Verify synchronous model training endpoint returns completed job summary."""
    payload = {
        "extractor": "mobilenetv4_conv_small",
        "epochs": 2,
        "batch_size": 16,
        "learning_rate": 0.001,
        "background": False,
    }
    resp = client.post("/training", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "completed"
    assert data["epochs_trained"] == 2
    assert "final_val_loss" in data
    assert "final_val_accuracy" in data
    assert data["final_val_accuracy"] is not None


def test_train_alias_endpoint() -> None:
    """Verify legacy /train alias triggers training pipeline correctly."""
    resp = client.post("/train", json={"epochs": 1})
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "completed"
    assert data["epochs_trained"] == 1


def test_training_endpoint_background() -> None:
    """Verify asynchronous training endpoint dispatches background task."""
    resp = client.post("/training", json={"epochs": 1, "background": True})
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "started"


def test_real_image_prediction_if_sample_exists() -> None:
    """Verify prediction endpoint with sample disk observation fixture if present."""
    sample_path = Path("data/sample/406185885_taxon_47120.jpg")
    if sample_path.exists():
        with open(sample_path, "rb") as f:
            resp = client.post("/predict", files={"file": ("sample.jpg", f, "image/jpeg")})
        assert resp.status_code == 200
        data = resp.json()
        assert data["top_prediction"]["scientific_name"] is not None
        assert "conformal_prediction_set" in data
        assert isinstance(data["is_ood_flagged"], bool)
        assert isinstance(data["requires_human_review"], bool)


def test_predict_endpoint_model_not_ready(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify prediction endpoint surfaces ModelNotReadyError as HTTP 503."""
    from taxon_vision.service.inference import ModelNotReadyError

    def mock_run_prediction(*args: object, **kwargs: object) -> None:
        raise ModelNotReadyError("Model head checkpoint is not available")

    monkeypatch.setattr(predict_mod, "run_prediction", mock_run_prediction)
    buf = io.BytesIO()
    Image.new("RGB", (32, 32), color="red").save(buf, format="JPEG")
    buf.seek(0)
    resp = client.post("/api/v1/predict", files={"file": ("unready.jpg", buf, "image/jpeg")})
    assert resp.status_code == 503
    assert "Model head checkpoint is not available" in resp.json()["detail"]


def test_predict_endpoint_valid_image() -> None:
    """Verify end-to-end prediction on a valid in-memory RGB JPEG image."""
    buf = io.BytesIO()
    Image.new("RGB", (32, 32), color="green").save(buf, format="JPEG")
    buf.seek(0)
    resp = client.post("/api/v1/predict", files={"file": ("valid.jpg", buf, "image/jpeg")})
    assert resp.status_code == 200
    data = resp.json()
    assert "top_prediction" in data
    assert data["top_prediction"]["scientific_name"] is not None
    assert "conformal_prediction_set" in data
    assert "latency_ms" in data


def test_build_live_prediction_multiple_candidates() -> None:
    """Verify _build_live_prediction formats multiple conformal candidate taxa."""
    pred = {
        "scientific_name": "Danaus plexippus",
        "common_name": "Monarch Butterfly",
        "confidence": 0.85,
        "conformal_set": ["Danaus plexippus", "Apis mellifera"],
        "is_ood": False,
        "requires_human_review": False,
    }
    resp = _build_live_prediction(pred, latency_ms=12.3)
    assert resp.top_prediction.scientific_name == "Danaus plexippus"
    assert len(resp.top_candidates) == 2
    assert resp.top_candidates[1].scientific_name == "Apis mellifera"
    assert resp.latency_ms == 12.3


def test_predict_endpoint_internal_server_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify prediction endpoint surfaces unexpected engine crashes as HTTP 500."""

    def mock_run_prediction(*args: object, **kwargs: object) -> None:
        raise RuntimeError("Simulated internal GPU memory fault")

    monkeypatch.setattr(predict_mod, "run_prediction", mock_run_prediction)
    buf = io.BytesIO()
    Image.new("RGB", (32, 32), color="red").save(buf, format="JPEG")
    buf.seek(0)
    resp = client.post("/api/v1/predict", files={"file": ("error.jpg", buf, "image/jpeg")})
    assert resp.status_code == 500
    assert "Inference execution failed" in resp.json()["detail"]
