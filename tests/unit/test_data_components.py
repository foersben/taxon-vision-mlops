from pathlib import Path

import pytest

from taxon_vision.data.dvc_pipeline import compute_dataset_hash
from taxon_vision.data.inat_client import INatAPIClient
from taxon_vision.data.s3_streamer import S3ImageStreamer
from taxon_vision.domain.license import AttributionRecord, OpenLicense
from taxon_vision.domain.observation import ObservationRecord
from taxon_vision.domain.taxonomy import TaxonNode


def test_s3_image_streamer():
    streamer = S3ImageStreamer()
    items = list(streamer.stream_sample_images(limit=3))
    assert len(items) == 3
    assert items[0][0] == "obs_0000"
    assert items[0][1].size == (224, 224)


@pytest.mark.asyncio
async def test_inat_api_client():
    client = INatAPIClient(max_requests_per_minute=600)
    data = await client.fetch_observations(taxon_id=1, per_page=2)
    assert len(data) == 2
    assert "observation_id" in data[0]


def test_compute_dataset_hash(tmp_path: Path):
    file1 = tmp_path / "test.txt"
    file1.write_text("sample content")
    hash_val = compute_dataset_hash(tmp_path)
    assert len(hash_val) == 64


def test_observation_record():
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
