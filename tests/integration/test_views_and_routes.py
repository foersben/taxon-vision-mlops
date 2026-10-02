"""Test views and routes.py.

This module provides functionality related to test_views_and_routes.
"""

import io

from fastapi.testclient import TestClient

from taxon_vision.monitoring.telemetry import PREDICTION_COUNTER
from taxon_vision.service.api import app

client = TestClient(app)


def test_index_view() -> None:
    """Test index view."""
    resp = client.get("/")
    assert resp.status_code == 200
    assert "TaxonVision" in resp.text


def test_predict_htmx_view() -> None:
    """Test predict htmx view."""
    dummy_img = io.BytesIO(b"fake image bytes")
    resp = client.post("/ui/predict-htmx", files={"file": ("test.jpg", dummy_img, "image/jpeg")})
    assert resp.status_code == 200
    assert "Monarch Butterfly" in resp.text


def test_explain_endpoint() -> None:
    """Test explain endpoint."""
    resp = client.post("/api/v1/explain")
    assert resp.status_code == 200
    assert resp.json()["heatmap_status"] == "generated"


def test_metrics_endpoint() -> None:
    """Test metrics endpoint."""
    PREDICTION_COUNTER.labels(status="test").inc()
    resp = client.get("/metrics")
    assert resp.status_code == 200
    assert "taxon_predictions_total" in resp.text
