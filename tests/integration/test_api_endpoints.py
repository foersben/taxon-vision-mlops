import io

from fastapi.testclient import TestClient

from taxon_vision.service.api import app

client = TestClient(app)


def test_health_endpoint() -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "healthy"


def test_predict_endpoint() -> None:
    dummy_img = io.BytesIO(b"fake image bytes")
    resp = client.post("/api/v1/predict", files={"file": ("test.jpg", dummy_img, "image/jpeg")})
    assert resp.status_code == 200
    data = resp.json()
    assert "top_prediction" in data
    assert data["top_prediction"]["scientific_name"] == "Danaus plexippus"
    assert "conformal_prediction_set" in data
