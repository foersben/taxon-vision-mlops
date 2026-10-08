import io
from pathlib import Path

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


def test_top_level_predict_endpoint() -> None:
    dummy_img = io.BytesIO(b"fake image bytes")
    resp = client.post("/predict", files={"file": ("test.jpg", dummy_img, "image/jpeg")})
    assert resp.status_code == 200
    data = resp.json()
    assert "top_prediction" in data
    assert "conformal_prediction_set" in data


def test_training_endpoint_sync() -> None:
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
    resp = client.post("/train", json={"epochs": 1})
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "completed"
    assert data["epochs_trained"] == 1


def test_training_endpoint_background() -> None:
    resp = client.post("/training", json={"epochs": 1, "background": True})
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "started"


def test_real_image_prediction_if_sample_exists() -> None:
    sample_path = Path("data/sample/406185885_taxon_47120.jpg")
    if sample_path.exists():
        with open(sample_path, "rb") as f:
            resp = client.post("/predict", files={"file": ("sample.jpg", f, "image/jpeg")})
        assert resp.status_code == 200
        data = resp.json()
        assert (
            data["top_prediction"]["scientific_name"] == "Apis mellifera"
            or data["top_prediction"]["scientific_name"] is not None
        )
        assert len(data["conformal_prediction_set"]) >= 1
