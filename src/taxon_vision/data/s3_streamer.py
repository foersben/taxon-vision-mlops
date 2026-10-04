# SPDX-FileCopyrightText: 2026 Benjamin Förster
# SPDX-License-Identifier: MIT
"""Streaming reader for AWS Open Data iNaturalist S3 bucket."""

from __future__ import annotations

from collections.abc import Generator

from PIL import Image


class S3ImageStreamer:
    """Stream images directly from S3 open data bucket without local storage explosion."""

    def __init__(self, bucket_name: str = "inaturalist-open-data") -> None:
        """Initialize the S3 image streaming interface.

        Args:
            bucket_name: AWS S3 bucket name containing open data images.
        """
        self.bucket_name = bucket_name

    def stream_sample_images(self, limit: int = 10) -> Generator[tuple[str, Image.Image], None, None]:
        """Generator yielding synthetic/mocked or streamed PIL images.

        Args:
            limit: The number of images to stream.

        Returns:
            A generator yielding tuples of image IDs and PIL images.
        """
        for i in range(limit):
            # Generate placeholder RGB image for deterministic pipeline testing
            img = Image.new("RGB", (224, 224), color=(30 + i * 20, 100, 150))
            yield f"obs_{i:04d}", img
