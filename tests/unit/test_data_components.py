# SPDX-FileCopyrightText: 2026 Benjamin Förster
# SPDX-License-Identifier: MIT
"""Unit tests for data streaming, API client, and domain models."""

from __future__ import annotations

import io
import sys
from pathlib import Path
from typing import Any, cast

import polars as pl
import pytest
from PIL import Image

from taxon_vision.data.inat_client import INatAPIClient
from taxon_vision.data.s3_streamer import S3ImageStreamer, _fetch_record_image
from taxon_vision.data.s3_streamer import main as streamer_main
from taxon_vision.domain.license import AttributionRecord, OpenLicense
from taxon_vision.domain.observation import ObservationRecord
from taxon_vision.domain.taxonomy import TaxonNode


def test_s3_image_streamer(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Verify S3ImageStreamer reads parquet manifest and yields images."""
    manifest_file = tmp_path / "manifest.parquet"
    df = pl.DataFrame(
        {
            "observation_uuid": ["uuid-1", "uuid-2"],
            "taxon_id": [1, 2],
            "image_url": ["https://mock.s3/img1.jpg", "https://mock.s3/img2.jpg"],
        }
    )
    df.write_parquet(manifest_file)

    img_bytes = io.BytesIO()
    Image.new("RGB", (32, 32), color="green").save(img_bytes, format="JPEG")
    fake_content = img_bytes.getvalue()

    class MockResponse:
        def __init__(self, content: bytes) -> None:
            self.content = content

        def raise_for_status(self) -> None:
            pass

    class MockClient:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def __enter__(self) -> MockClient:
            return self

        def __exit__(self, *args: Any) -> None:
            pass

        def get(self, _url: str) -> MockResponse:
            return MockResponse(fake_content)

    monkeypatch.setattr("httpx.Client", MockClient)

    streamer = S3ImageStreamer(manifest_path=manifest_file)
    items = list(streamer.stream_images(limit=2))
    assert len(items) == 2
    assert items[0][0] == "uuid-1"
    assert items[0][1] == 1
    assert isinstance(items[0][2], Image.Image)


@pytest.mark.asyncio
async def test_inat_api_client(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify INatAPIClient asynchronously fetches observations."""

    class MockAsyncResponse:
        def __init__(self, json_data: dict[str, Any]) -> None:
            self._json = json_data

        def raise_for_status(self) -> None:
            pass

        def json(self) -> dict[str, Any]:
            return self._json

    class MockAsyncClient:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        async def __aenter__(self) -> MockAsyncClient:
            return self

        async def __aexit__(self, *args: Any) -> None:
            pass

        async def get(self, _url: str, **_kwargs: Any) -> MockAsyncResponse:
            return MockAsyncResponse(
                {
                    "results": [
                        {
                            "id": "obs_1",
                            "photos": [{"license_code": "cc-by", "url": "square_url1", "attribution": "attr1"}],
                        },
                        {
                            "id": "obs_2",
                            "photos": [{"license_code": "cc-by", "url": "square_url2", "attribution": "attr2"}],
                        },
                    ]
                }
            )

    monkeypatch.setattr("httpx.AsyncClient", MockAsyncClient)

    client = INatAPIClient(max_requests_per_minute=600)
    data = await client.fetch_observations(taxon_id=1, per_page=2)
    assert len(data) == 2
    assert "observation_id" in data[0]


def test_observation_record() -> None:
    """Verify ObservationRecord domain entity instantiation and properties."""
    taxon = TaxonNode(taxon_id=1, scientific_name="Danaus plexippus")
    attribution = AttributionRecord(
        photographer_name="Jane Doe",
        license_code=OpenLicense.CC_BY,
        license_url="http://example.com",
        observation_uuid="uuid-123",
    )
    obs = ObservationRecord(
        observation_id="obs_100", taxon=taxon, attribution=attribution, image_url="http://example.com/img.jpg"
    )
    assert obs.observation_id == "obs_100"


def test_s3_image_streamer_balanced_and_concurrent(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Verify S3ImageStreamer handles balanced taxon sampling and multithreaded fetching."""
    manifest_file = tmp_path / "balanced_manifest.parquet"
    df = pl.DataFrame(
        {
            "observation_uuid": [f"uuid-{i}" for i in range(6)],
            "taxon_id": [1, 1, 1, 2, 2, 2],
            "image_url": [f"https://mock.s3/img_{i}.jpg" for i in range(6)],
        }
    )
    df.write_parquet(manifest_file)

    img_bytes = io.BytesIO()
    Image.new("RGB", (32, 32), color="blue").save(img_bytes, format="JPEG")
    fake_content = img_bytes.getvalue()

    class MockResponse:
        def __init__(self, content: bytes) -> None:
            self.content = content

        def raise_for_status(self) -> None:
            pass

    class MockClient:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def __enter__(self) -> MockClient:
            return self

        def __exit__(self, *args: Any) -> None:
            pass

        def get(self, _url: str) -> MockResponse:
            return MockResponse(fake_content)

    monkeypatch.setattr("httpx.Client", MockClient)

    streamer = S3ImageStreamer(manifest_path=manifest_file)
    items = list(streamer.stream_images(limit=4, max_workers=2, balanced=True, shuffle=True))
    assert len(items) == 4

    # Test _fetch_record_image fallback with client=None
    res = _fetch_record_image(
        {"observation_uuid": "test-uuid", "taxon_id": 1, "image_url": "https://mock.s3/test.jpg"},
        client=None,
    )
    assert res is not None
    assert res[0] == "test-uuid"

    # Test _fetch_record_image on exception returns None
    class FailingClient:
        def get(self, _url: str) -> Any:
            raise RuntimeError("Connection dropped")

    fail_res = _fetch_record_image(
        {"observation_uuid": "test-uuid", "taxon_id": 1, "image_url": "https://mock.s3/fail.jpg"},
        client=cast(Any, FailingClient()),
    )
    assert fail_res is None


def test_s3_streamer_main_cli(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Verify S3 streamer CLI main() executes and writes downloaded images to output directory."""
    manifest_file = tmp_path / "cli_manifest.parquet"
    df = pl.DataFrame(
        {
            "observation_uuid": ["obs-a", "obs-b"],
            "taxon_id": [10, 20],
            "image_url": ["https://mock.s3/a.jpg", "https://mock.s3/b.jpg"],
        }
    )
    df.write_parquet(manifest_file)
    out_dir = tmp_path / "streamed_images"

    img_bytes = io.BytesIO()
    Image.new("RGB", (16, 16), color="yellow").save(img_bytes, format="JPEG")
    fake_content = img_bytes.getvalue()

    class MockResponse:
        def __init__(self, content: bytes) -> None:
            self.content = content

        def raise_for_status(self) -> None:
            pass

    class MockClient:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def __enter__(self) -> MockClient:
            return self

        def __exit__(self, *args: Any) -> None:
            pass

        def get(self, _url: str) -> MockResponse:
            return MockResponse(fake_content)

    monkeypatch.setattr("httpx.Client", MockClient)
    monkeypatch.setattr(
        sys,
        "argv",
        ["s3_streamer", "--manifest", str(manifest_file), "--limit", "2", "--out", str(out_dir)],
    )

    streamer_main()
    assert out_dir.exists()
    saved_files = list(out_dir.glob("*.jpg"))
    assert len(saved_files) == 2


def test_fetch_image_with_retry_success_and_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify fetch_image_with_retry retries transient errors and succeeds when response recovers.

    Why:
        Confirms network resilience against transient socket drops and rate limiting during bulk streaming.

    How:
        Simulates an initial RequestError followed by HTTP 503, finally recovering with 200 OK using zero delay backoff.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
    """
    import httpx

    from taxon_vision.data.s3_streamer import create_streaming_http_client, fetch_image_with_retry

    attempts = 0

    class TransientClient:
        def get(self, _url: str) -> Any:
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise httpx.ConnectError("DNS resolution failed")
            if attempts == 2:

                class ErrorResponse:
                    status_code = 503
                    content = b"Service Unavailable"

                return ErrorResponse()

            class SuccessResponse:
                status_code = 200
                content = b"valid_image_bytes"

            return SuccessResponse()

    client = cast(httpx.Client, TransientClient())
    data = fetch_image_with_retry("https://mock.s3/img.jpg", client=client, max_retries=3, backoff_factor=0.0)
    assert data == b"valid_image_bytes"
    assert attempts == 3

    failing_attempts = 0

    class AlwaysFailingClient:
        def get(self, _url: str) -> Any:
            nonlocal failing_attempts
            failing_attempts += 1
            raise httpx.ReadTimeout("Timeout reading stream")

    fail_client = cast(httpx.Client, AlwaysFailingClient())
    failed_data = fetch_image_with_retry(
        "https://mock.s3/fail.jpg", client=fail_client, max_retries=2, backoff_factor=0.0
    )
    assert failed_data is None
    assert failing_attempts == 3

    real_or_mock_client = create_streaming_http_client()
    assert real_or_mock_client is not None
