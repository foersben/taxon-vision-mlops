# SPDX-FileCopyrightText: 2026 Benjamin Förster
# SPDX-License-Identifier: MIT
"""Streaming reader for AWS Open Data iNaturalist S3 bucket."""

from __future__ import annotations

import io
from collections.abc import Generator
from pathlib import Path

import httpx
import polars as pl
from PIL import Image


class S3ImageStreamer:
    """Stream images directly from S3 open data bucket without local storage explosion."""

    def __init__(self, manifest_path: str | Path = "data/manifests/dataset_manifest.parquet") -> None:
        """Initialize the S3 image streaming interface.

        Args:
            manifest_path: Path to the Parquet manifest file containing image URLs.
        """
        self.manifest_path = Path(manifest_path)
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found at {self.manifest_path}. Run ingest_data.py first.")

        self.df = pl.read_parquet(self.manifest_path)

    def stream_images(self, limit: int | None = None) -> Generator[tuple[str, int, Image.Image], None, None]:
        """Generator yielding streamed PIL images directly from the S3 URLs.

        This method is intended to be used in training pipelines to avoid downloading the entire dataset to disk.
        Furthermore the streaming approach is also used to avoid RAM overloads on smaller devices.

        Args:
            limit: Optional limit on the number of images to stream.

        Yields:
            A generator yielding tuples of (observation_uuid, taxon_id, PIL Image).
        """
        records = self.df.to_dicts()
        if limit is not None:
            records = records[:limit]

        with httpx.Client(timeout=15.0) as client:
            for record in records:
                obs_uuid = str(record["observation_uuid"])
                taxon_id = int(record["taxon_id"])
                image_url = str(record["image_url"])

                try:
                    response = client.get(image_url)
                    response.raise_for_status()
                    img: Image.Image = Image.open(io.BytesIO(response.content))
                    # Convert to RGB to handle grayscale/RGBA inconsistencies
                    if img.mode != "RGB":
                        img = img.convert("RGB")
                    yield obs_uuid, taxon_id, img
                except Exception as e:
                    print(f"Failed to stream {obs_uuid} from {image_url}: {e}")
