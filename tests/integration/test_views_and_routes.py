# SPDX-FileCopyrightText: 2026 Benjamin Förster
# SPDX-License-Identifier: MIT
"""Integration tests for HTMX/Jinja2 server-rendered views, CAM explanation, and telemetry."""

from __future__ import annotations

import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from taxon_vision.monitoring.telemetry import PREDICTION_COUNTER
from taxon_vision.service.api import app

client = TestClient(app)


def test_index_view() -> None:
    """Verify primary dashboard index view renders HTML with title."""
    resp = client.get("/")
    assert resp.status_code == 200
    assert "TaxonVision" in resp.text


def test_predict_htmx_view() -> None:
    """Verify HTMX prediction route returns rendered partial card."""
    dummy_img = io.BytesIO()
    Image.new("RGB", (10, 10), color="red").save(dummy_img, format="JPEG")
    dummy_img.seek(0)
    resp = client.post("/ui/predict-htmx", files={"file": ("test.jpg", dummy_img, "image/jpeg")})
    assert resp.status_code == 200
    assert "Classification Result" in resp.text


def test_explain_endpoint() -> None:
    """Verify CAM visual explanation endpoint returns generated heatmap status without file."""
    resp = client.post("/api/v1/explain")
    assert resp.status_code == 200
    assert resp.json()["heatmap_status"] == "generated"


def test_explain_endpoint_with_image() -> None:
    """Verify CAM visual explanation endpoint accepts uploaded image and computes spatial CAM."""
    buf = io.BytesIO()
    Image.new("RGB", (32, 32), color="green").save(buf, format="JPEG")
    buf.seek(0)
    resp = client.post("/api/v1/explain", files={"file": ("leaf.jpg", buf, "image/jpeg")})
    assert resp.status_code == 200
    data = resp.json()
    assert data["heatmap_status"] == "generated"
    assert data["heatmap_shape"] == [32, 32]
    assert data["latency_ms"] >= 0.0
    assert data["cam_heatmap_base64"] is not None


def test_explain_endpoint_htmx() -> None:
    """Verify CAM explanation endpoint renders HTML fragment when requested via HTMX."""
    buf = io.BytesIO()
    Image.new("RGB", (32, 32), color="green").save(buf, format="JPEG")
    buf.seek(0)
    resp = client.post(
        "/api/v1/explain?target_class_index=0",
        files={"file": ("leaf.jpg", buf, "image/jpeg")},
        headers={"HX-Request": "true"},
    )
    assert resp.status_code == 200
    assert "cam-overlay" in resp.text
    assert "data:image/png;base64," in resp.text
    assert "<img" in resp.text


def test_explain_endpoint_invalid_image() -> None:
    """Verify CAM visual explanation endpoint rejects invalid image bytes with HTTP 400."""
    dummy_img = io.BytesIO(b"corrupt image bytes")
    resp = client.post("/api/v1/explain", files={"file": ("corrupt.jpg", dummy_img, "image/jpeg")})
    assert resp.status_code == 400
    assert "Cannot decode upload as valid image for CAM" in resp.json()["detail"]


def test_metrics_endpoint() -> None:
    """Verify Prometheus exposition endpoint serves scraped metric samples."""
    PREDICTION_COUNTER.labels(status="test").inc()
    resp = client.get("/metrics")
    assert resp.status_code == 200
    assert "taxon_predictions_total" in resp.text


def test_explain_endpoint_saliency_extraction_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify CAM visual explanation endpoint translates SaliencyExtractionError into HTTP 422."""
    import taxon_vision.service.routes.explain as explain_mod
    from taxon_vision.inference.cam import SaliencyExtractionError

    def mock_generate_heatmap(*args: object, **kwargs: object) -> None:
        raise SaliencyExtractionError("Backbone failed to emit spatial activations")

    monkeypatch.setattr(explain_mod, "generate_heatmap", mock_generate_heatmap)
    buf = io.BytesIO()
    Image.new("RGB", (32, 32), color="green").save(buf, format="JPEG")
    buf.seek(0)
    resp = client.post("/api/v1/explain", files={"file": ("leaf.jpg", buf, "image/jpeg")})
    assert resp.status_code == 422
    assert "Backbone failed to emit spatial activations" in resp.json()["detail"]


def test_ui_predict_htmx_saliency_extraction_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify UI predict HTMX route sets heatmap_status='unavailable' on SaliencyExtractionError."""
    import taxon_vision.service.routes.views as views_mod
    from taxon_vision.inference.cam import SaliencyExtractionError

    def mock_generate_heatmap(*args: object, **kwargs: object) -> None:
        raise SaliencyExtractionError("Spatial map unavailable for model")

    monkeypatch.setattr(views_mod, "generate_heatmap", mock_generate_heatmap)
    dummy_img = io.BytesIO()
    Image.new("RGB", (10, 10), color="blue").save(dummy_img, format="JPEG")
    dummy_img.seek(0)
    resp = client.post("/ui/predict-htmx", files={"file": ("test.jpg", dummy_img, "image/jpeg")})
    assert resp.status_code == 200
    assert "Visual attribution (saliency heatmap) is unavailable" in resp.text
