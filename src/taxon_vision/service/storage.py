# SPDX-FileCopyrightText: 2026 Benjamin Förster
# SPDX-License-Identifier: MIT
"""Storage service for persisting flagged observations and caching uploads ephemerally."""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt
import redis

from taxon_vision.config import get_settings

logger = logging.getLogger(__name__)


@runtime_checkable
class ImageCacheProtocol(Protocol):
    """Protocol defining the interface for temporary observation image caches.

    Why:
        Decouples the HTTP presentation and CAM explanation layers from physical caching
        mechanisms, enabling transparent switching between in-process memory and distributed
        Redis storage without code alterations.

    How:
        Requires implementations of content-addressed byte storage (`store`), retrieval (`get`),
        cache clearing (`clear`), and item counting (`__len__`).
    """

    def store(self, image_bytes: bytes) -> str:
        """Insert an image and return its content hash."""
        ...

    def get(self, image_hash: str) -> bytes | None:
        """Look up a cached image by hash."""
        ...

    def clear(self) -> None:
        """Remove all entries from the cache."""
        ...

    def __len__(self) -> int:
        """Return the count of cached entries."""
        ...


class MemoryImageCache:
    """Thread-safe, bounded in-memory cache of uploaded observation images.

    Why:
        HTMX is stateless: the multipart upload stream is consumed by `/ui/predict-htmx`, so a follow-up request (such as the CAM button) cannot re-submit the file and would otherwise fail with HTTP 422. Persisting temporary uploads to disk would cause avoidable SSD write amplification and leak potentially sensitive imagery, so the bytes are held in RAM only and addressed by their SHA-256 content hash.

    How:
        Entries live in an `OrderedDict` ordered by recency of use. On insertion, expired entries (older than `ttl_seconds`, measured with the monotonic clock) are pruned first and the least recently used entry is evicted if `max_items` would be exceeded. A `threading.Lock` guards all mutations because FastAPI executes synchronous work in a worker thread pool.

    Complexity:
        `store` and `get` are O(1) amortised; pruning in `store` is O(n) in the number of entries, which is bounded by `max_items`.

    Attributes:
        max_items: Maximum number of images retained simultaneously.
        ttl_seconds: Lifetime of an entry in seconds, refreshed on every insertion of the same image.
    """

    def __init__(self, max_items: int, ttl_seconds: float) -> None:
        """Initialise an empty cache.

        Args:
            max_items: Maximum number of images to retain. Must be positive.
            ttl_seconds: Time-to-live of each entry in seconds. Must be positive.

        Raises:
            ValueError: If `max_items` or `ttl_seconds` is not strictly positive.
        """
        if max_items <= 0 or ttl_seconds <= 0:
            raise ValueError("max_items and ttl_seconds must be strictly positive.")
        self.max_items = max_items
        self.ttl_seconds = ttl_seconds
        self._lock = threading.Lock()
        self._entries: OrderedDict[str, tuple[bytes, float]] = OrderedDict()

    @staticmethod
    def hash_bytes(image_bytes: bytes) -> str:
        """Compute the content address of an image.

        Args:
            image_bytes: Raw encoded image bytes.

        Returns:
            Lower-case hexadecimal SHA-256 digest.
        """
        return hashlib.sha256(image_bytes).hexdigest()

    def store(self, image_bytes: bytes) -> str:
        """Insert an image and return its content hash.

        Re-storing identical bytes refreshes the entry's timestamp and recency instead of duplicating it.

        Args:
            image_bytes: Raw encoded image bytes.

        Returns:
            SHA-256 hex digest usable as a lookup key in `get`.
        """
        image_hash = self.hash_bytes(image_bytes)
        now = time.monotonic()
        with self._lock:
            self._prune_expired(now)
            if image_hash not in self._entries:
                self._evict_to_capacity()
            self._entries[image_hash] = (image_bytes, now)
            self._entries.move_to_end(image_hash)
        return image_hash

    def get(self, image_hash: str) -> bytes | None:
        """Look up a cached image and mark it as recently used.

        Args:
            image_hash: SHA-256 hex digest returned by `store`.

        Returns:
            The original bytes, or None if the hash is unknown or the entry has expired.
        """
        now = time.monotonic()
        with self._lock:
            entry = self._entries.get(image_hash)
            if entry is None:
                return None
            image_bytes, stored_at = entry
            if now - stored_at > self.ttl_seconds:
                del self._entries[image_hash]
                return None
            self._entries.move_to_end(image_hash)
            return image_bytes

    def clear(self) -> None:
        """Remove all entries (used for test isolation and explicit memory release)."""
        with self._lock:
            self._entries.clear()

    def __len__(self) -> int:
        """Return the number of currently retained entries (expired ones included until pruned)."""
        with self._lock:
            return len(self._entries)

    def _prune_expired(self, now: float) -> None:
        """Drop every entry older than the TTL. Caller must hold the lock.

        Args:
            now: Current monotonic timestamp in seconds.
        """
        expired = [key for key, (_, stored_at) in self._entries.items() if now - stored_at > self.ttl_seconds]
        for key in expired:
            del self._entries[key]

    def _evict_to_capacity(self) -> None:
        """Evict least-recently-used entries so one new entry fits. Caller must hold the lock."""
        while len(self._entries) >= self.max_items:
            self._entries.popitem(last=False)


EphemeralImageCache = MemoryImageCache
"""Alias of MemoryImageCache to preserve backward compatibility for callers and tests."""


class RedisImageCache:
    """Distributed Redis-backed cache for temporary observation images.

    Why:
        In multi-replica Kubernetes deployments, user uploads to `/ui/predict-htmx` land on
        arbitrary pods. Follow-up requests (such as CAM attribution via `/api/v1/explain`) are routed
        round-robin across pods. A centralized in-memory store prevents HTTP 422 cache misses without
        requiring sticky sessions or local disk write amplification.

    How:
        Images are keyed by their SHA-256 digest with prefix `taxon:img:`. Images are stored
        using `SETEX` with a configurable TTL. Lookups fetch bytes and refresh TTL via `EXPIRE`.
        Connection pool handles connection reuse and timeouts.

    Complexity:
        `store` and `get` are O(1) network roundtrips bounded by socket timeouts. Space complexity
        in Redis is bounded by TTL and the Redis `maxmemory` eviction policy (`allkeys-lru`).

    Attributes:
        ttl_seconds: Lifetime of an entry in seconds.
        key_prefix: Redis key namespace prefix to prevent collision.
    """

    def __init__(
        self,
        url: str,
        ttl_seconds: float,
        *,
        key_prefix: str = "taxon:img:",
        socket_timeout: float = 2.0,
        redis_client: Any | None = None,
    ) -> None:
        """Initialise the distributed Redis image cache.

        Args:
            url: Redis connection URL (e.g. `redis://localhost:6379/0`).
            ttl_seconds: Time-to-live of each entry in seconds. Must be strictly positive.
            key_prefix: Redis key namespace prefix for cached images.
            socket_timeout: Socket timeout for Redis operations in seconds.
            redis_client: Optional pre-configured Redis client instance (e.g. for testing with fakeredis).

        Raises:
            ValueError: If `ttl_seconds` is not strictly positive.
        """
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be strictly positive.")
        self.ttl_seconds = ttl_seconds
        self.key_prefix = key_prefix
        if redis_client is not None:
            self._redis = redis_client
        else:
            pool = redis.ConnectionPool.from_url(
                url,
                socket_timeout=socket_timeout,
                socket_connect_timeout=socket_timeout,
            )
            self._redis = redis.Redis(connection_pool=pool)

    @staticmethod
    def hash_bytes(image_bytes: bytes) -> str:
        """Compute the content address of an image.

        Args:
            image_bytes: Raw encoded image bytes.

        Returns:
            Lower-case hexadecimal SHA-256 digest.
        """
        return hashlib.sha256(image_bytes).hexdigest()

    def store(self, image_bytes: bytes) -> str:
        """Insert an image into Redis with TTL and return its SHA-256 hash.

        Args:
            image_bytes: Raw encoded image bytes.

        Returns:
            SHA-256 hex digest usable as a lookup key in `get`.
        """
        image_hash = self.hash_bytes(image_bytes)
        key = f"{self.key_prefix}{image_hash}"
        ttl = max(1, int(self.ttl_seconds))
        self._redis.setex(key, ttl, image_bytes)
        return image_hash

    def get(self, image_hash: str) -> bytes | None:
        """Look up a cached image in Redis and refresh its TTL.

        Args:
            image_hash: SHA-256 hex digest returned by `store`.

        Returns:
            The original bytes, or None if the hash is unknown or expired.
        """
        key = f"{self.key_prefix}{image_hash}"
        data = self._redis.get(key)
        if data is None:
            return None
        ttl = max(1, int(self.ttl_seconds))
        self._redis.expire(key, ttl)
        if isinstance(data, (bytes, bytearray)):
            return bytes(data)
        return str(data).encode("latin1")

    def clear(self) -> None:
        """Remove all keys with the cache prefix."""
        pattern = f"{self.key_prefix}*"
        cursor: Any = 0
        while True:
            cursor, keys = self._redis.scan(cursor=cursor, match=pattern, count=100)
            if keys:
                self._redis.delete(*keys)
            if cursor in (0, "0", b"0"):
                break

    def __len__(self) -> int:
        """Return the count of cached keys matching the prefix."""
        pattern = f"{self.key_prefix}*"
        cursor: Any = 0
        total = 0
        while True:
            cursor, keys = self._redis.scan(cursor=cursor, match=pattern, count=100)
            total += len(keys)
            if cursor in (0, "0", b"0"):
                break
        return total

    def ping(self) -> bool:
        """Verify Redis connectivity.

        Returns:
            True if Redis ping succeeds, False otherwise.
        """
        try:
            return bool(self._redis.ping())
        except (redis.ConnectionError, redis.TimeoutError, OSError) as exc:
            logger.warning("Redis ping failed: %s", exc)
            return False


@lru_cache(maxsize=1)
def get_image_cache() -> ImageCacheProtocol:
    """Return the process-wide ephemeral image cache, configured from application settings.

    Why:
        A lazily constructed singleton avoids reading settings at import time and can be reset in tests via `get_image_cache.cache_clear()`.

    Returns:
        The shared `ImageCacheProtocol` instance (RedisImageCache if configured and reachable, else MemoryImageCache).
    """
    service = get_settings().service
    if service.redis_url:
        try:
            cache = RedisImageCache(
                url=service.redis_url,
                ttl_seconds=service.image_cache_ttl_seconds,
            )
            if cache.ping():
                logger.info("Initialized distributed RedisImageCache connecting to %s", service.redis_url)
                return cache
            logger.warning(
                "Redis connection failed for %s; falling back to local MemoryImageCache",
                service.redis_url,
            )
        except Exception as exc:
            logger.warning(
                "Failed to initialize RedisImageCache (%s); falling back to local MemoryImageCache",
                exc,
            )
    return MemoryImageCache(
        max_items=service.image_cache_max_items,
        ttl_seconds=service.image_cache_ttl_seconds,
    )


@dataclass(frozen=True)
class FlaggedObservationPayload:
    """Consolidated payload for an observation flagged during inference."""

    observation_id: str
    prediction_result: dict[str, Any]
    embedding: npt.NDArray[np.float32] | None = None
    image_bytes: bytes | None = None
    storage_dir: Path | str | None = None


def persist_flagged_observation(
    observation_id: str,
    prediction_result: dict[str, Any],
    embedding: npt.NDArray[np.float32] | None = None,
    storage_dir: Path | str | None = None,
    *,
    image_bytes: bytes | None = None,
) -> Path:
    """Atomically persist metadata, image crop, and embeddings for observations flagged during inference.

    Why:
        Observations identified as Out-of-Distribution (OOD) or exhibiting high conformal ambiguity require human expert review and provide the highest information gain for active learning retrain loops. Staging the tabular decision metadata, the dense visual feature embedding, and the raw visual image crop allows subsequent ingestion services (e.g. SeaweedFS, LanceDB, or DagsHub Label Studio triage queues) to consume high-priority edge cases without recomputing vision foundation forward passes.

    How:
        1. Resolves destination storage directory from arguments or application configuration.
        2. Compiles a consolidated record containing prediction metrics and timestamp.
        3. If an embedding vector is provided, serializes it to disk as an optimized `.npy` file.
        4. If raw image bytes are provided, saves the observation image crop as `{observation_id}_crop.jpg`.
        5. Serializes the metadata record as an individual JSON artifact and appends a line to `manifest.jsonl`.

    Complexity:
        O(D + B) where D is feature dimension and B is raw image byte length.

    Args:
        observation_id: Unique identifier for the observation capture.
        prediction_result: Dictionary returned by the prediction pipeline containing classification results.
        embedding: Optional dense feature representation vector of shape (D,).
        storage_dir: Optional filesystem path override for the staging directory.
        image_bytes: Optional binary payload of the raw uploaded image crop.

    Returns:
        Path to the written metadata JSON artifact.

    Raises:
        OSError: If destination directories cannot be created or file writes fail.
    """
    settings = get_settings()
    target_dir = Path(storage_dir or settings.model.flagged_storage_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    embedding_rel_path: str | None = None
    if embedding is not None:
        emb_filename = f"{observation_id}_embedding.npy"
        emb_path = target_dir / emb_filename
        np.save(str(emb_path), embedding)
        embedding_rel_path = emb_filename

    image_rel_path: str | None = None
    if image_bytes is not None:
        img_filename = f"{observation_id}_crop.jpg"
        img_path = target_dir / img_filename
        img_path.write_bytes(image_bytes)
        image_rel_path = img_filename

    meta_filename = f"{observation_id}_meta.json"
    meta_path = target_dir / meta_filename

    top_pred = {
        "scientific_name": str(prediction_result.get("scientific_name", "")),
        "common_name": str(prediction_result.get("common_name", "")),
        "confidence": float(prediction_result.get("confidence", 0.0)),
    }

    record: dict[str, Any] = {
        "observation_id": observation_id,
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "ood_energy_score": float(prediction_result.get("ood_energy_score", 0.0)),
        "is_ood": bool(prediction_result.get("is_ood", False)),
        "requires_human_review": bool(prediction_result.get("requires_human_review", False)),
        "conformal_set": list(prediction_result.get("conformal_set", [])),
        "top_prediction": top_pred,
        "probabilities": list(prediction_result.get("probabilities", [])),
        "embedding_path": embedding_rel_path,
        "image_path": image_rel_path,
    }

    meta_path.write_text(json.dumps(record, indent=2), encoding="utf-8")

    manifest_path = target_dir / "manifest.jsonl"
    with manifest_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

    logger.info("Persisted flagged observation %s (crop=%s) to %s", observation_id, bool(image_bytes), target_dir)
    return meta_path
