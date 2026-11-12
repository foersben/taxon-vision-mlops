# SPDX-FileCopyrightText: 2026 Benjamin Förster
# SPDX-License-Identifier: MIT
"""Unit tests for active learning query strategies and triage review queue."""

from __future__ import annotations

import numpy as np

from taxon_vision.active_learning.query import margin_sampling
from taxon_vision.active_learning.triage_queue import TriageItem, TriageQueue


def test_margin_sampling() -> None:
    """Verify margin sampling correctly ranks ambiguous probability distributions higher."""
    probs = np.array([[0.6, 0.3, 0.1], [0.34, 0.33, 0.33]])
    scores = margin_sampling(probs)
    # The second row is more ambiguous, so uncertainty score should be higher
    assert scores[1] > scores[0]


def test_triage_queue() -> None:
    """Verify priority queue orders items by uncertainty descending and handles popping."""
    queue = TriageQueue()
    queue.push(TriageItem(observation_id="obs_1", uncertainty_score=0.4, conformal_set_size=1, image_url="url1"))
    queue.push(TriageItem(observation_id="obs_2", uncertainty_score=0.9, conformal_set_size=3, image_url="url2"))
    assert len(queue) == 2
    assert len(queue.queue) == 2

    # Highest uncertainty first
    item = queue.pop()
    assert item is not None
    assert item.observation_id == "obs_2"

    item2 = queue.pop()
    assert item2 is not None
    assert item2.observation_id == "obs_1"

    assert queue.pop() is None
    assert len(queue) == 0


def test_dagshub_label_studio_client_consent_guard_and_export(tmp_path, monkeypatch) -> None:
    """Verify DagsHub Label Studio client rejects unconsented items, uploads consented crops via SigV4 PUT without boto3, and exports tasks."""
    import json
    import urllib.request

    from taxon_vision.active_learning.label_studio_sync import DagsHubLabelStudioClient, DagsHubRepoConfig

    crop = tmp_path / "obs-1_crop.jpg"
    crop.write_bytes(b"\xff\xd8\xff\xe0dummy")

    unconsented_meta = tmp_path / "obs-unconsented_meta.json"
    unconsented_meta.write_text(
        json.dumps({"observation_id": "obs-u", "license_code": "", "photographer_name": ""}), encoding="utf-8"
    )

    consented_meta = tmp_path / "obs-1_meta.json"
    consented_meta.write_text(
        json.dumps(
            {
                "observation_id": "obs-1",
                "license_code": "CC-BY",
                "photographer_name": "Jane Doe",
                "uncertainty_score": 0.72,
            }
        ),
        encoding="utf-8",
    )

    s3_puts: list[tuple[str, str, str, bytes]] = []
    api_calls: list[tuple[str, str, object]] = []

    class _FakeResp:
        def __enter__(self) -> _FakeResp:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

    def _fake_urlopen(req: urllib.request.Request, timeout: float = 0.0) -> _FakeResp:
        s3_puts.append(
            (
                req.get_method(),
                req.full_url,
                str(req.get_header("Authorization") or ""),
                bytes(req.data or b""),
            )
        )
        return _FakeResp()

    monkeypatch.setattr(urllib.request, "urlopen", _fake_urlopen)

    client = DagsHubLabelStudioClient(DagsHubRepoConfig(token="pat-test", project_id=7))

    def _fake_api(method: str, path: str, payload: object | None = None) -> object:
        api_calls.append((method, path, payload))
        if path.endswith("/export?exportType=JSON"):
            return [{"id": 1, "annotations": [{"result": []}]}, {"id": 2, "annotations": []}]
        return {}

    monkeypatch.setattr(client, "_api_request", _fake_api)

    assert client.sync_flagged_observation(unconsented_meta, crop) is False
    assert client.sync_flagged_observation(consented_meta, crop) is True
    assert len(s3_puts) == 1
    method, url, auth, data = s3_puts[0]
    assert method == "PUT"
    assert url == "https://dagshub.com/api/v1/repo-buckets/s3/foersben/taxon-vision-mlops/triage-queue/obs-1_crop.jpg"
    assert auth.startswith("AWS4-HMAC-SHA256 Credential=pat-test/")
    assert data == b"\xff\xd8\xff\xe0dummy"
    assert json.loads(consented_meta.read_text(encoding="utf-8"))["synced_to_label_studio"] is True

    export_file = tmp_path / "export.json"
    count = client.export_completed_annotations(export_file)
    assert count == 1
    assert len(json.loads(export_file.read_text(encoding="utf-8"))) == 1


def test_persist_flagged_observation_computes_margin_and_gates_consent(tmp_path) -> None:
    """Compute margin-sampling uncertainty and persist CC metadata only upon explicit valid consent."""
    import json

    from taxon_vision.service.storage import ContributorConsent, persist_flagged_observation

    pred = {
        "scientific_name": "Apis mellifera",
        "common_name": "Western Honey Bee",
        "confidence": 0.55,
        "probabilities": [0.55, 0.40, 0.05],
        "is_ood": False,
        "requires_human_review": True,
    }
    meta_no_consent = persist_flagged_observation("obs-noc", pred, storage_dir=tmp_path, image_bytes=b"abc")
    data_no = json.loads(meta_no_consent.read_text(encoding="utf-8"))
    assert data_no["contribute_open_license"] is False
    assert abs(data_no["uncertainty_score"] - 0.85) < 1e-4

    meta_yes = persist_flagged_observation(
        "obs-yes",
        pred,
        storage_dir=tmp_path,
        image_bytes=b"abc",
        consent=ContributorConsent(contribute_open_license=True, photographer_name="Alex Smith", license_code="cc-by"),
    )
    data_yes = json.loads(meta_yes.read_text(encoding="utf-8"))
    assert data_yes["contribute_open_license"] is True
    assert data_yes["license_code"] == "CC-BY"
    assert data_yes["photographer_name"] == "Alex Smith"


def test_s3_streamer_routes_dagshub_triage_uri_through_boto3(monkeypatch) -> None:
    """Verify DagsHub repository S3 URIs resolve with DAGSHUB_TOKEN SigV4 credentials in s3_streamer."""
    import builtins
    import io
    from types import SimpleNamespace

    from taxon_vision.data.s3_streamer import create_streaming_http_client, fetch_image_with_retry

    captured: dict[str, object] = {}

    class _FakeS3:
        def get_object(self, **kwargs: object) -> dict[str, object]:
            captured["bucket"] = kwargs["Bucket"]
            captured["key"] = kwargs["Key"]
            return {"Body": io.BytesIO(b"jpeg-payload")}

    def _make_client(*_args: object, **kwargs: object) -> _FakeS3:
        captured["kwargs"] = kwargs
        return _FakeS3()

    fake_boto3 = SimpleNamespace(client=_make_client)
    orig_import = builtins.__import__

    def _patched_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "boto3":
            return fake_boto3
        return orig_import(name, *args, **kwargs)

    monkeypatch.setenv("DAGSHUB_TOKEN", "pat-s3-123")
    monkeypatch.setattr(builtins, "__import__", _patched_import)

    with create_streaming_http_client() as client:
        data = fetch_image_with_retry("s3://taxon-vision-mlops/triage-queue/sample_crop.jpg", client=client)

    assert data == b"jpeg-payload"
    assert captured["bucket"] == "taxon-vision-mlops"
    assert captured["key"] == "triage-queue/sample_crop.jpg"
    kwargs_dict = captured["kwargs"]
    assert isinstance(kwargs_dict, dict)
    assert kwargs_dict["endpoint_url"] == "https://dagshub.com/api/v1/repo-buckets/s3/foersben"
    assert kwargs_dict["aws_access_key_id"] == "pat-s3-123"
