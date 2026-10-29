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


def _jpeg_buffer(color: str = "red", size: int = 10) -> io.BytesIO:
    """Build an in-memory JPEG used as an upload payload."""
    buf = io.BytesIO()
    Image.new("RGB", (size, size), color=color).save(buf, format="JPEG")
    buf.seek(0)
    return buf


def test_predict_htmx_view() -> None:
    """Verify HTMX prediction route returns rendered partial card."""
    dummy_img = io.BytesIO()
    Image.new("RGB", (10, 10), color="red").save(dummy_img, format="JPEG")
    dummy_img.seek(0)
    resp = client.post("/ui/predict-htmx", files={"file": ("test.jpg", dummy_img, "image/jpeg")})
    assert resp.status_code == 200
    assert "Classification Result" in resp.text


def test_predict_htmx_renders_drawers_and_cached_image_hash() -> None:
    """The card exposes progressive-disclosure drawers and a CAM button bound to the cached image hash."""
    from taxon_vision.service.storage import EphemeralImageCache, get_image_cache

    payload = _jpeg_buffer().getvalue()
    resp = client.post("/ui/predict-htmx", files={"file": ("test.jpg", payload, "image/jpeg")})
    assert resp.status_code == 200
    expected_hash = EphemeralImageCache.hash_bytes(payload)
    assert resp.text.count("<details") == 3
    assert f"image_hash={expected_hash}" in resp.text
    assert 'id="btn-inspect-cam"' in resp.text
    assert get_image_cache().get(expected_hash) == payload


def test_predict_htmx_does_not_run_cam_synchronously(monkeypatch: pytest.MonkeyPatch) -> None:
    """Uploading must not trigger a CAM forward pass; attribution is requested on demand."""
    import taxon_vision.inference.cam as cam_mod

    def fail(*args: object, **kwargs: object) -> None:
        raise AssertionError("CAM must not be computed during /ui/predict-htmx")

    monkeypatch.setattr(cam_mod, "generate_heatmap", fail)
    resp = client.post("/ui/predict-htmx", files={"file": ("test.jpg", _jpeg_buffer(), "image/jpeg")})
    assert resp.status_code == 200


def test_explain_by_image_hash_without_file() -> None:
    """Follow-up CAM request resolves the image from the cache and returns an HTML overlay."""
    from taxon_vision.service.storage import get_image_cache

    image_hash = get_image_cache().store(_jpeg_buffer("green", 32).getvalue())
    resp = client.post(
        f"/api/v1/explain?image_hash={image_hash}&target_class_index=0",
        headers={"HX-Request": "true"},
    )
    assert resp.status_code == 200
    assert "cam-overlay" in resp.text
    assert "data:image/png;base64," in resp.text


def test_explain_by_image_hash_json_client() -> None:
    """JSON clients may also reference a cached image and receive the standard response model."""
    from taxon_vision.service.storage import get_image_cache

    image_hash = get_image_cache().store(_jpeg_buffer("green", 32).getvalue())
    resp = client.post(f"/api/v1/explain?image_hash={image_hash}")
    assert resp.status_code == 200
    assert resp.json()["heatmap_shape"] == [32, 32]


def test_explain_unknown_hash_returns_422_for_json_clients() -> None:
    """An expired or unknown hash is a client error (HTTP 422) for non-HTMX callers."""
    resp = client.post("/api/v1/explain?image_hash=" + "0" * 64)
    assert resp.status_code == 422
    assert "expired or unknown" in resp.json()["detail"]


def test_explain_unknown_hash_renders_alert_fragment_for_htmx() -> None:
    """HTMX does not swap 4xx bodies, so the failure is rendered inline as an alert fragment."""
    resp = client.post("/api/v1/explain?image_hash=" + "0" * 64, headers={"HX-Request": "true"})
    assert resp.status_code == 200
    assert "cam-error" in resp.text
    assert "expired or unknown" in resp.text


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


def test_ui_predict_htmx_saliency_failure_surfaces_on_demand(monkeypatch: pytest.MonkeyPatch) -> None:
    """A CAM failure does not break the upload; it is reported inline when attribution is requested."""
    import taxon_vision.service.routes.explain as explain_mod
    from taxon_vision.inference.cam import SaliencyExtractionError
    from taxon_vision.service.storage import get_image_cache

    def mock_generate_heatmap(*args: object, **kwargs: object) -> None:
        raise SaliencyExtractionError("Spatial map unavailable for model")

    monkeypatch.setattr(explain_mod, "generate_heatmap", mock_generate_heatmap)
    resp = client.post("/ui/predict-htmx", files={"file": ("test.jpg", _jpeg_buffer("blue"), "image/jpeg")})
    assert resp.status_code == 200

    image_hash = get_image_cache().store(_jpeg_buffer("blue").getvalue())
    explain = client.post(f"/api/v1/explain?image_hash={image_hash}", headers={"HX-Request": "true"})
    assert explain.status_code == 200
    assert "Spatial map unavailable for model" in explain.text


def test_multi_pod_split_brain_resolution(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify multi-pod split-brain resolution with shared Redis cache backend.

    Why:
        In a multi-replica Kubernetes deployment, an upload request to `/ui/predict-htmx`
        lands on Pod 1, whereas a subsequent HTMX CAM request (`/api/v1/explain`) may be routed
        to Pod 2. Without a shared distributed cache, Pod 2 returns HTTP 422.

    How:
        Simulate Pod 1 and Pod 2 as separate cache instances backed by a shared FakeServer.
        Pod 1 stores the upload; Pod 2 successfully resolves the image and computes CAM.
    """
    import fakeredis

    from taxon_vision.service.storage import RedisImageCache

    server = fakeredis.FakeServer()
    pod1_client = fakeredis.FakeRedis(server=server)
    pod2_client = fakeredis.FakeRedis(server=server)

    pod1_cache = RedisImageCache("redis://fake:6379/0", ttl_seconds=60.0, redis_client=pod1_client)
    pod2_cache = RedisImageCache("redis://fake:6379/0", ttl_seconds=60.0, redis_client=pod2_client)

    # 1. Pod 1 handles the upload
    monkeypatch.setattr("taxon_vision.service.routes.views.get_image_cache", lambda: pod1_cache)
    resp_upload = client.post(
        "/ui/predict-htmx",
        files={"file": ("leaf.jpg", _jpeg_buffer("green", 32), "image/jpeg")},
    )
    assert resp_upload.status_code == 200

    image_bytes = _jpeg_buffer("green", 32).getvalue()
    image_hash = pod1_cache.hash_bytes(image_bytes)
    assert pod1_cache.get(image_hash) is not None
    # Verify Pod 2 sees the key through the shared backend
    assert pod2_cache.get(image_hash) is not None

    # 2. Pod 2 handles the CAM request
    monkeypatch.setattr("taxon_vision.service.routes.explain.get_image_cache", lambda: pod2_cache)
    resp_explain = client.post(
        f"/api/v1/explain?image_hash={image_hash}&target_class_index=0",
        headers={"HX-Request": "true"},
    )
    assert resp_explain.status_code == 200
    assert "cam-overlay" in resp_explain.text
