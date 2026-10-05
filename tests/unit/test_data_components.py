# SPDX-FileCopyrightText: 2026 Benjamin Förster
# SPDX-License-Identifier: MIT
"""Unit tests for data streaming, API client, and domain models."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import polars as pl
import pytest
from PIL import Image

from taxon_vision.data.inat_client import INatAPIClient
from taxon_vision.data.s3_streamer import S3ImageStreamer
from taxon_vision.domain.license import AttributionRecord, OpenLicense
from taxon_vision.domain.observation import ObservationRecord
from taxon_vision.domain.taxonomy import TaxonNode


def test_s3_image_streamer(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
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
async def test_inat_api_client() -> None:
    client = INatAPIClient(max_requests_per_minute=600)
    data = await client.fetch_observations(taxon_id=1, per_page=2)
    assert len(data) == 2
    assert "observation_id" in data[0]


def test_observation_record() -> None:
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
