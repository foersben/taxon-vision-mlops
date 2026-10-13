# SPDX-FileCopyrightText: 2026 Benjamin Förster
# SPDX-License-Identifier: MIT
"""Streaming reader for AWS Open Data iNaturalist S3 bucket."""

from __future__ import annotations

import concurrent.futures
import io
import logging
from collections.abc import Generator
from pathlib import Path
from typing import Any

import httpx
import polars as pl
from PIL import Image

logger = logging.getLogger(__name__)


def _prepare_balanced_manifest(
    df: pl.DataFrame,
    limit: int | None,
    shuffle: bool,
    seed: int,
) -> list[dict[str, Any]]:
    """Sample records evenly across unique taxa in the manifest.

    Args:
        df: Polars DataFrame holding observation records.
        limit: Optional upper bound on records to yield.
        shuffle: Whether to pseudorandomly shuffle records.
        seed: PRNG seed for reproducible shuffling.

    Returns:
        List of observation record dictionaries.
    """
    taxa = df["taxon_id"].unique().to_list()
    per_taxon = max(1, (limit + len(taxa) - 1) // len(taxa)) if limit is not None else len(df)
    dfs = []
    for tid in taxa:
        sub = df.filter(pl.col("taxon_id") == tid)
        if shuffle:
            sub = sub.sample(fraction=1.0, shuffle=True, seed=seed)
        dfs.append(sub.head(per_taxon))
    combined = pl.concat(dfs)
    if shuffle:
        combined = combined.sample(fraction=1.0, shuffle=True, seed=seed)
    return combined.head(limit).to_dicts() if limit is not None else combined.to_dicts()


def _prepare_stream_records(
    df: pl.DataFrame,
    limit: int | None,
    shuffle: bool,
    balanced: bool,
    seed: int,
) -> list[dict[str, Any]]:
    """Select, balance, and shuffle observation records for streaming.

    Args:
        df: Polars DataFrame holding observation records.
        limit: Optional upper bound on records to yield.
        shuffle: Whether to pseudorandomly shuffle records.
        balanced: Whether to balance sample counts across unique taxon categories.
        seed: PRNG seed for reproducible shuffling.

    Returns:
        List of selected observation record dictionaries.
    """
    if balanced and "taxon_id" in df.columns:
        return _prepare_balanced_manifest(df, limit, shuffle, seed)
    if shuffle:
        shuffled = df.sample(fraction=1.0, shuffle=True, seed=seed)
        return shuffled.head(limit).to_dicts() if limit is not None else shuffled.to_dicts()
    return df.head(limit).to_dicts() if limit is not None else df.to_dicts()


def _fetch_record_image(
    record: dict[str, Any],
    client: httpx.Client | None = None,
) -> tuple[str, int, Image.Image] | None:
    """Fetch and decode an image from a single observation record.

    Args:
        record: Observation record dictionary.
        client: Optional existing httpx.Client to reuse.

    Returns:
        Tuple of (observation_uuid, taxon_id, Image) or None if retrieval failed.
    """
    obs_uuid = str(record["observation_uuid"])
    taxon_id = int(record["taxon_id"])
    image_url = str(record["image_url"])
    try:
        if client is not None:
            response = client.get(image_url)
        else:
            with httpx.Client(timeout=15.0) as local_client:
                response = local_client.get(image_url)
        response.raise_for_status()
        raw_img = Image.open(io.BytesIO(response.content))
        img: Image.Image = raw_img.convert("RGB") if raw_img.mode != "RGB" else raw_img
        return obs_uuid, taxon_id, img
    except Exception as e:
        logger.warning("Failed to stream %s from %s: %s", obs_uuid, image_url, e)
        return None


def _stream_sequential(
    records: list[dict[str, Any]],
) -> Generator[tuple[str, int, Image.Image], None, None]:
    """Stream records sequentially using a persistent HTTP client session.

    Args:
        records: Observation record dictionaries to stream.

    Yields:
        Tuple of (observation_uuid, taxon_id, Image).
    """
    with httpx.Client(timeout=15.0) as client:
        for record in records:
            res = _fetch_record_image(record, client=client)
            if res is not None:
                yield res


def _stream_concurrent(
    records: list[dict[str, Any]],
    max_workers: int,
) -> Generator[tuple[str, int, Image.Image], None, None]:
    """Stream records concurrently using a ThreadPoolExecutor.

    Args:
        records: Observation record dictionaries to stream.
        max_workers: Thread worker pool concurrency limit.

    Yields:
        Tuple of (observation_uuid, taxon_id, Image).
    """
    chunk_size = max(1, min(32, len(records)))
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        for i in range(0, len(records), chunk_size):
            chunk = records[i : i + chunk_size]
            for result in executor.map(_fetch_record_image, chunk):
                if result is not None:
                    yield result


class S3ImageStreamer:
    """Stream imagery directly from open cloud buckets without local disk explosion.

    Why:
        Global biodiversity datasets (e.g. iNaturalist on AWS Open Data) exceed hundreds of gigabytes or terabytes in raw image storage. Downloading entire datasets locally before training causes storage exhaustion, slow provisioning cycles, and high disk I/O. Streaming images in memory directly from remote S3 presigned URLs via HTTP connection pooling bounds local disk footprint to zero while avoiding RAM exhaustion through lazy generator evaluation.

    How:
        Loads the tabular Parquet manifest into memory using Apache Arrow / Polars, extracts observation UUIDs, taxon IDs, and image URLs, and lazily streams byte streams using persistent HTTP connections via `httpx.Client`. Converts inbound buffers into standardized RGB PIL Image instances on-the-fly.

    Attributes:
        manifest_path: Filesystem path to the Parquet manifest containing image URLs.
        df: Polars DataFrame holding observation records.
    """

    def __init__(self, manifest_path: str | Path = "data/manifests/dataset_manifest.parquet") -> None:
        """Initialize the S3 image streaming interface from a Parquet manifest.

        Why:
            Reads metadata using columnar Parquet format to enable fast projected queries without parsing uncompressed text or CSV files.

        How:
            Resolves manifest path, asserts file existence, and loads columnar tables via `polars.read_parquet`.

        Args:
            manifest_path: Path to the Parquet manifest file containing image URLs.

        Raises:
            FileNotFoundError: If the manifest file cannot be located on disk.
        """
        self.manifest_path = Path(manifest_path)
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found at {self.manifest_path}. Run ingest_data.py first.")

        self.df = pl.read_parquet(self.manifest_path)

    def stream_images(
        self,
        limit: int | None = None,
        shuffle: bool = False,
        balanced: bool = False,
        seed: int = 42,
        max_workers: int = 8,
    ) -> Generator[tuple[str, int, Image.Image], None, None]:
        """Lazily yield streamed PIL images directly from remote S3 URLs.

        Why:
            Executing deep learning training or representation extraction without downloading the entire dataset prevents disk saturation and enables training on edge or memory-constrained workstations. Yielding a lazy generator ensures only one image buffer exists in RAM at any given instant per worker.

        How:
            Prepares selected, balanced, and shuffled records from the manifest, then streams images sequentially or concurrently via connection-pooled HTTP workers.

        Args:
            limit: Optional upper bound on the number of images to yield from the manifest.
            shuffle: Whether to pseudorandomly shuffle observation records prior to streaming.
            balanced: Whether to balance sample counts across unique taxon categories.
            seed: PRNG seed for reproducible manifest shuffling.
            max_workers: Concurrency level for parallel HTTP GET streaming requests.

        Yields:
            Tuple of (observation_uuid, taxon_id, PIL.Image.Image) for each streamed record.
        """
        records = _prepare_stream_records(self.df, limit, shuffle, balanced, seed)
        if max_workers <= 1 or len(records) <= 2:
            yield from _stream_sequential(records)
        else:
            yield from _stream_concurrent(records, max_workers)


def main() -> None:
    """CLI entrypoint to test and stream images from the Parquet manifest.

    Why:
        Provides developers with an interactive diagnostic utility to verify S3 network connectivity, inspect remote image dimensions, and save small test batches locally.

    How:
        Parses `--manifest`, `--limit`, and optional `--out` directory arguments, initializes `S3ImageStreamer`, and iterates through the generator logging observation metadata.
    """
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
