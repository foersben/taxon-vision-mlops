import io

from fastapi.testclient import TestClient
from PIL import Image

from taxon_vision.monitoring.telemetry import PREDICTION_COUNTER
from taxon_vision.service.api import app

client = TestClient(app)


def test_index_view() -> None:
    resp = client.get("/")
    assert resp.status_code == 200
    assert "TaxonVision" in resp.text


def test_predict_htmx_view() -> None:
    dummy_img = io.BytesIO()
    Image.new("RGB", (10, 10), color="red").save(dummy_img, format="JPEG")
    dummy_img.seek(0)
    resp = client.post("/ui/predict-htmx", files={"file": ("test.jpg", dummy_img, "image/jpeg")})
    assert resp.status_code == 200
    assert "Classification Result" in resp.text


def test_explain_endpoint() -> None:
    resp = client.post("/api/v1/explain")
    assert resp.status_code == 200
    assert resp.json()["heatmap_status"] == "generated"


def test_metrics_endpoint() -> None:
    PREDICTION_COUNTER.labels(status="test").inc()
    resp = client.get("/metrics")
    assert resp.status_code == 200
    assert "taxon_predictions_total" in resp.text
