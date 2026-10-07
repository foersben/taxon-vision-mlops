# SPDX-FileCopyrightText: 2026 Benjamin Förster
# SPDX-License-Identifier: MIT
"""Streaming reader for AWS Open Data iNaturalist S3 bucket."""

from __future__ import annotations

import io
import logging
from collections.abc import Generator
from pathlib import Path

import httpx
import polars as pl
from PIL import Image

logger = logging.getLogger(__name__)


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
                    logger.warning("Failed to stream %s from %s: %s", obs_uuid, image_url, e)


def main() -> None:
    """CLI entrypoint to test and stream images from the Parquet manifest."""
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="Stream images from Parquet manifest directly from S3 open data without disk bloat."
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default="data/manifests/dataset_manifest.parquet",
        help="Path to dataset Parquet manifest",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of images to stream (default: 5)",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Optional directory to save streamed images for local inspection",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )

    streamer = S3ImageStreamer(manifest_path=args.manifest)
    out_dir = Path(args.out) if args.out else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loaded manifest '%s' with %d records.", args.manifest, len(streamer.df))
    logger.info("Streaming up to %d images in-memory...", args.limit)

    count = 0
    for obs_uuid, taxon_id, img in streamer.stream_images(limit=args.limit):
        count += 1
        logger.info(
            "[%d/%d] Streamed observation %s (taxon=%d): %s %s",
            count,
            args.limit,
            obs_uuid,
            taxon_id,
            img.size,
            img.mode,
        )
        if out_dir:
            save_path = out_dir / f"{obs_uuid}_taxon_{taxon_id}.jpg"
            img.save(save_path)
            logger.info("  -> Saved to %s", save_path)

    logger.info("Successfully streamed %d images.", count)


if __name__ == "__main__":
    main()
