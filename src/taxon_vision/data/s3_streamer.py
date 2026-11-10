# SPDX-FileCopyrightText: 2026 Benjamin Förster
# SPDX-License-Identifier: MIT
"""Streaming reader for AWS Open Data iNaturalist S3 bucket."""

from __future__ import annotations

import concurrent.futures
import io
import logging
import threading
import time
from collections.abc import Generator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple, Protocol, runtime_checkable

import httpx
import polars as pl
from PIL import Image

from taxon_vision.config import get_settings

logger = logging.getLogger(__name__)


class StreamedObservation(NamedTuple):
    """Typed container for a single streamed observation record and decoded RGB image."""

    observation_uuid: str
    taxon_id: int
    image: Image.Image


class _FetchAttemptOutcome(NamedTuple):
    """Outcome of a single HTTP GET attempt."""

    content: bytes | None
    should_retry: bool


@dataclass(frozen=True)
class StreamSelectionConfig:
    """Configuration parameters governing manifest record selection and shuffling."""

    limit: int | None = None
    shuffle: bool = False
    balanced: bool = False
    seed: int = 42


@runtime_checkable
class HttpResponseWithStatus(Protocol):
    """Protocol for HTTP response objects carrying a status code."""

    status_code: int


@runtime_checkable
class ClosableResource(Protocol):
    """Protocol for resources exposing a close method."""

    def close(self) -> None:
        """Close open socket and underlying transport resources."""
        ...


def _prepare_balanced_manifest(
    df: pl.DataFrame,
    config: StreamSelectionConfig,
) -> list[dict[str, Any]]:
    """Sample records evenly across unique taxa in the manifest.

    Args:
        df: Polars DataFrame holding observation records.
        config: Selection configuration specifying limit, shuffle, and seed.

    Returns:
        List of observation record dictionaries.
    """
    taxa = df["taxon_id"].unique().to_list()
    per_taxon = max(1, (config.limit + len(taxa) - 1) // len(taxa)) if config.limit is not None else len(df)
    dfs = []
    for tid in taxa:
        sub = df.filter(pl.col("taxon_id") == tid)
        if config.shuffle:
            sub = sub.sample(fraction=1.0, shuffle=True, seed=config.seed)
        dfs.append(sub.head(per_taxon))
    combined = pl.concat(dfs)
    if config.shuffle:
        combined = combined.sample(fraction=1.0, shuffle=True, seed=config.seed)
    return combined.head(config.limit).to_dicts() if config.limit is not None else combined.to_dicts()


def _prepare_stream_records(
    df: pl.DataFrame,
    config: StreamSelectionConfig,
) -> list[dict[str, Any]]:
    """Select, balance, and shuffle observation records for streaming.

    Args:
        df: Polars DataFrame holding observation records.
        config: Selection configuration governing limit, shuffle, balance, and seed.

    Returns:
        List of selected observation record dictionaries.
    """
    if config.balanced and "taxon_id" in df.columns:
        return _prepare_balanced_manifest(df, config)
    if config.shuffle:
        shuffled = df.sample(fraction=1.0, shuffle=True, seed=config.seed)
        return shuffled.head(config.limit).to_dicts() if config.limit is not None else shuffled.to_dicts()
    return df.head(config.limit).to_dicts() if config.limit is not None else df.to_dicts()


def create_streaming_http_client(
    max_connections: int | None = None,
    max_keepalive_connections: int | None = None,
    retries: int | None = None,
    timeout_seconds: float | None = None,
) -> httpx.Client:
    """Instantiate a persistent HTTP client configured with connection pooling and transport retries.

    Why:
        Streaming high-volume imagery over remote HTTP endpoints saturates local sockets and triggers DNS lookup exhaustion unless connections are pooled and reused across worker threads.

    How:
        Loads pooling limits, transport retry bounds, and request timeouts from centralized settings (or arguments) and constructs a thread-safe httpx.Client.

    Args:
        max_connections: Maximum socket pool connection capacity.
        max_keepalive_connections: Maximum idle keep-alive sockets to preserve.
        retries: Transport-level connection retry count.
        timeout_seconds: Maximum request timeout in seconds.

    Returns:
        Configured httpx.Client instance.
    """
    settings = get_settings()
    cfg = settings.streaming
    limits = httpx.Limits(
        max_connections=max_connections if max_connections is not None else cfg.http_max_connections,
        max_keepalive_connections=max_keepalive_connections
        if max_keepalive_connections is not None
        else cfg.http_keepalive_connections,
    )
    transport = httpx.HTTPTransport(retries=retries if retries is not None else cfg.http_max_retries)
    timeout = httpx.Timeout(timeout_seconds if timeout_seconds is not None else cfg.http_timeout_seconds)
    return httpx.Client(transport=transport, limits=limits, timeout=timeout)


def _is_transient_http_error(status_code: int) -> bool:
    """Determine whether an HTTP status code represents a transient, retryable failure.

    Why:
        Rate limiting (429) and server-side errors (5xx) are transient conditions that should be retried with exponential backoff.

    How:
        Returns True if status_code is 429 or between 500 and 599, otherwise False.

    Args:
        status_code: HTTP response status integer.

    Returns:
        Boolean flag indicating retry eligibility.
    """
    return status_code == 429 or (500 <= status_code < 600)


def _attempt_http_get(url: str, client: httpx.Client, attempt: int, retries: int) -> _FetchAttemptOutcome:
    """Execute a single HTTP GET request and classify retry eligibility.

    Args:
        url: Remote image URL to retrieve.
        client: HTTP client instance.
        attempt: Current zero-based attempt index.
        retries: Maximum configured retry attempts.

    Returns:
        _FetchAttemptOutcome containing content bytes and retry indicator.
    """
    try:
        response = client.get(url)
        status_code = response.status_code if isinstance(response, HttpResponseWithStatus) else 200
        if status_code == 200:
            return _FetchAttemptOutcome(content=response.content, should_retry=False)
        if _is_transient_http_error(status_code) and attempt < retries:
            return _FetchAttemptOutcome(content=None, should_retry=True)
        logger.warning("HTTP %d when fetching %s (attempt %d/%d)", status_code, url, attempt + 1, retries + 1)
        return _FetchAttemptOutcome(content=None, should_retry=False)
    except (httpx.RequestError, OSError) as exc:
        if attempt < retries:
            return _FetchAttemptOutcome(content=None, should_retry=True)
        logger.warning("Request failed for %s after %d retries: %s", url, retries, exc)
        return _FetchAttemptOutcome(content=None, should_retry=False)


def fetch_image_with_retry(
    url: str,
    client: httpx.Client,
    max_retries: int | None = None,
    backoff_factor: float | None = None,
) -> bytes | None:
    """Fetch raw image byte content from a remote URL with exponential backoff on transient errors.

    Why:
        Remote object store endpoints can encounter DNS lookup throttling, socket exhaustion, or rate-limiting during high-concurrency ingestion. Retrying with exponential backoff prevents transient dropouts from corrupting training datasets.

    How:
        Executes HTTP GET requests up to max_retries times via `_attempt_http_get`. Sleeps for backoff_factor * (2 ** attempt) seconds before retrying transient HTTP 429/5xx or transport errors. Returns content bytes on HTTP 200, or None after exhaustion.

    Args:
        url: Remote image URL to retrieve.
        client: Persistent httpx.Client to reuse for requests.
        max_retries: Optional maximum retry attempts. If None, derived from centralized settings.
        backoff_factor: Optional exponential backoff multiplier. If None, derived from centralized settings.

    Returns:
        Decoded image bytes if retrieved successfully, or None if retrieval failed.
    """
    settings = get_settings()
    retries = max_retries if max_retries is not None else settings.streaming.http_max_retries
    factor = backoff_factor if backoff_factor is not None else settings.streaming.http_backoff_factor

    for attempt in range(retries + 1):
        outcome = _attempt_http_get(url, client, attempt, retries)
        if outcome.content is not None:
            return outcome.content
        if not outcome.should_retry:
            return None
        time.sleep(factor * (2**attempt))
    return None


def _fetch_record_image(
    record: dict[str, Any],
    client: httpx.Client | None = None,
) -> StreamedObservation | None:
    """Fetch and decode an image from a single observation record.

    Why:
        Centralizes the decoding pipeline from observation metadata into in-memory PIL RGB representations.

    How:
        Extracts observation identifiers and URL, fetches raw bytes via fetch_image_with_retry using a persistent client session, and decodes byte buffers into standardized RGB format.

    Args:
        record: Observation record dictionary containing UUID, taxon ID, and image URL.
        client: Optional existing httpx.Client to reuse across requests.

    Returns:
        StreamedObservation tuple of (observation_uuid, taxon_id, Image) or None if retrieval or decoding failed.
    """
    obs_uuid = str(record["observation_uuid"])
    taxon_id = int(record["taxon_id"])
    image_url = str(record["image_url"])
    try:
        content: bytes | None
        if client is not None:
            content = fetch_image_with_retry(image_url, client=client)
        else:
            with create_streaming_http_client() as local_client:
                content = fetch_image_with_retry(image_url, client=local_client)

        if content is None:
            return None

        raw_img = Image.open(io.BytesIO(content))
        img: Image.Image = raw_img.convert("RGB") if raw_img.mode != "RGB" else raw_img
        return StreamedObservation(obs_uuid, taxon_id, img)
    except Exception as e:
        logger.warning("Failed to stream %s from %s: %s", obs_uuid, image_url, e)
        return None


def _stream_sequential(
    records: list[dict[str, Any]],
    client: httpx.Client | None = None,
) -> Generator[StreamedObservation, None, None]:
    """Stream records sequentially using a persistent HTTP client session.

    Args:
        records: Observation record dictionaries to stream.
        client: Optional persistent httpx.Client session.

    Yields:
        StreamedObservation of (observation_uuid, taxon_id, Image).
    """
    if client is not None:
        for record in records:
            res = _fetch_record_image(record, client=client)
            if res is not None:
                yield res
    else:
        with create_streaming_http_client() as new_client:
            for record in records:
                res = _fetch_record_image(record, client=new_client)
                if res is not None:
                    yield res


class _WorkerTLS(threading.local):
    """Thread-local storage for dedicated worker HTTP clients."""

    client: httpx.Client | None = None


_tls = _WorkerTLS()


def _stream_concurrent(
    records: list[dict[str, Any]],
    max_workers: int,
    client: httpx.Client | None = None,
) -> Generator[StreamedObservation, None, None]:
    """Stream records concurrently using a ThreadPoolExecutor with per-worker HTTP clients.

    Args:
        records: Observation record dictionaries to stream.
        max_workers: Thread worker pool concurrency limit.
        client: Optional explicit httpx.Client override (used for mock injection in tests).

    Yields:
        StreamedObservation of (observation_uuid, taxon_id, Image).
    """
    chunk_size = max(1, min(32, len(records)))
    active_clients: list[httpx.Client] = []
    pool_lock = threading.Lock()

    def _init_thread_worker() -> None:
        if client is None:
            c = create_streaming_http_client()
            _tls.client = c
            with pool_lock:
                active_clients.append(c)

    def _fetch_worker(record: dict[str, Any]) -> StreamedObservation | None:
        active_client = client if client is not None else _tls.client
        return _fetch_record_image(record, client=active_client)

    try:
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=max_workers, initializer=_init_thread_worker
        ) as executor:
            for i in range(0, len(records), chunk_size):
                chunk = records[i : i + chunk_size]
                for result in executor.map(_fetch_worker, chunk):
                    if result is not None:
                        yield result
    finally:
        for c in active_clients:
            if isinstance(c, ClosableResource):
                c.close()


class S3ImageStreamer:
    """Stream imagery directly from open cloud buckets without local disk explosion.

    Why:
        Global biodiversity datasets (e.g. iNaturalist on AWS Open Data) exceed hundreds of gigabytes or terabytes in raw image storage. Downloading entire datasets locally before training causes storage exhaustion, slow provisioning cycles, and high disk I/O. Streaming images in memory directly from remote S3 presigned URLs via HTTP connection pooling bounds local disk footprint to zero while avoiding RAM exhaustion through lazy generator evaluation.

    How:
        Loads the tabular Parquet manifest into memory using Apache Arrow / Polars, extracts observation UUIDs, taxon IDs, and image URLs, and lazily streams byte streams using persistent HTTP connections via `httpx.Client`. Converts inbound buffers into standardized RGB PIL Image instances on-the-fly.

    Attributes:
        manifest_path: Filesystem path to the Parquet manifest containing image URLs.
        df: Polars DataFrame holding observation records.
        client: Optional shared httpx.Client instance.
    """

    def __init__(
        self,
        manifest_path: str | Path = "data/manifests/dataset_manifest.parquet",
        client: httpx.Client | None = None,
    ) -> None:
        """Initialize the S3 image streaming interface from a Parquet manifest.

        Why:
            Reads metadata using columnar Parquet format to enable fast projected queries without parsing uncompressed text or CSV files.

        How:
            Resolves manifest path, asserts file existence, loads columnar tables via polars.read_parquet, and optionally registers a persistent HTTP client.

        Args:
            manifest_path: Path to the Parquet manifest file containing image URLs.
            client: Optional pre-configured httpx.Client for HTTP pooling reuse.

        Raises:
            FileNotFoundError: If the manifest file cannot be located on disk.
        """
        self.manifest_path = Path(manifest_path)
        if not self.manifest_path.exists():
            raise FileNotFoundError(
                f"Manifest not found at {self.manifest_path}. Run 'python -m taxon_vision.pipelines.ingest' first."
            )

        self.df = pl.read_parquet(self.manifest_path)
        self._external_client = client is not None
        self._client = client

    def __enter__(self) -> S3ImageStreamer:
        """Enter context manager, initializing persistent HTTP client if none exists."""
        if self._client is None:
            self._client = create_streaming_http_client()
        return self

    def __exit__(self, *args: Any) -> None:
        """Exit context manager, closing owned HTTP client resources."""
        if self._client is not None and not self._external_client:
            if isinstance(self._client, ClosableResource):
                self._client.close()
            self._client = None

    def stream_images(
        self,
        limit: int | None = None,
        shuffle: bool = False,
        balanced: bool = False,
        seed: int = 42,
        max_workers: int = 8,
    ) -> Generator[StreamedObservation, None, None]:
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
            StreamedObservation of (observation_uuid, taxon_id, PIL.Image.Image) for each streamed record.
        """
        selection = StreamSelectionConfig(limit=limit, shuffle=shuffle, balanced=balanced, seed=seed)
        records = _prepare_stream_records(self.df, selection)
        if max_workers <= 1 or len(records) <= 2:
            yield from _stream_sequential(records, client=self._client)
        else:
            yield from _stream_concurrent(records, max_workers, client=self._client)


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
