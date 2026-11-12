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
from functools import lru_cache, partial
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import anyio
import numpy as np
import numpy.typing as npt
import redis

from taxon_vision.active_learning.query import margin_sampling
from taxon_vision.active_learning.triage_queue import TriageItem, TriageQueue
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


_OPEN_LICENSE_URLS: dict[str, str] = {
    "cc0": "https://creativecommons.org/publicdomain/zero/1.0/",
    "cc-by": "https://creativecommons.org/licenses/by/4.0/",
    "cc-by-nc": "https://creativecommons.org/licenses/by-nc/4.0/",
}

_TRIAGE_QUEUE = TriageQueue()


def get_triage_queue() -> TriageQueue:
    """Return the process-wide priority review queue ranking flagged observations by margin uncertainty.

    Why:
        Provides an in-memory O(log N) binary max-heap over flagged observations so active-learning consumers can inspect or dequeue highest-uncertainty specimens deterministically.

    How:
        Returns the module-level TriageQueue singleton populated during `persist_flagged_observation`.

    Returns:
        Active TriageQueue instance.
    """
    return _TRIAGE_QUEUE


@dataclass(frozen=True)
class ContributorConsent:
    """Explicit open-license opt-in and attribution metadata supplied by a citizen-science contributor.

    Why:
        Replaces loose boolean/string tuples at the FastAPI API/HTML submission boundary with an explicit domain value object so Rule 05 licensing invariants are checked uniformly.

    How:
        Constructed by `/predict` and `/views/predict` handlers from multipart form inputs and evaluated via `is_valid_contributor_consent()`.

    Attributes:
        contribute_open_license: Whether the user explicitly checked the open-license donation box.
        photographer_name: Attributable contributor credit (minimum 3 characters when opted in).
        license_code: Chosen Creative Commons license (`cc0`, `cc-by`, or `cc-by-nc`).
        license_url: Canonical license deed URL resolved automatically if omitted.
    """

    contribute_open_license: bool = False
    photographer_name: str = ""
    license_code: str = ""
    license_url: str = ""


@dataclass(frozen=True)
class FlaggedObservationPayload:
    """Consolidated payload for an observation flagged during inference.

    Why:
        Prevents parameter arity creep (`> 4` positional args under Rule 08) across storage and background triage queueing call sites.

    How:
        Bundles the UUID, prediction dictionary, optional embedding vector, raw image bytes, target directory, and contributor open-license consent.

    Attributes:
        observation_id: Unique prediction UUID.
        prediction_result: Serialised inference prediction payload.
        embedding: Optional L2-normalised feature embedding vector.
        image_bytes: Optional raw uploaded image crop bytes.
        storage_dir: Optional filesystem staging directory override.
        consent: Optional open-license opt-in consent object.
    """

    observation_id: str
    prediction_result: dict[str, Any]
    embedding: npt.NDArray[np.float32] | None = None
    image_bytes: bytes | None = None
    storage_dir: Path | str | None = None
    consent: ContributorConsent | None = None


def is_valid_contributor_consent(consent: ContributorConsent | None) -> bool:
    """Return True when contributor explicitly opted into an allowed Creative Commons license with attribution.

    Why:
        Enforces Rule 05 Open License Compliance at the serving boundary: citizen-science uploads without affirmative Creative Commons licensing (`CC0`, `CC-BY`, `CC-BY-NC`) and non-trivial photographer attribution (`>= 3` characters) must never enter external annotation buckets or training manifests.

    How:
        Verifies `contribute_open_license` is True, normalises `license_code` to lowercase hyphenated form against `_OPEN_LICENSE_URLS`, and checks trimmed `photographer_name` length.

    Args:
        consent: Optional ContributorConsent submitted with the prediction request.

    Returns:
        True if open-license and attribution requirements are satisfied, otherwise False.
    """
    if consent is None or not consent.contribute_open_license:
        return False
    license_norm = consent.license_code.strip().lower().replace("_", "-")
    return license_norm in _OPEN_LICENSE_URLS and len(consent.photographer_name.strip()) >= 3


def _compute_margin_uncertainty(probabilities: list[float]) -> float:
    """Compute bounded `[0, 1]` active-learning margin sampling priority score from class probabilities.

    Why:
        Ranking flagged captures inside `TriageQueue` requires a monotonic uncertainty score (`1.0 - (p_1 - p_2)`) so near-tie classifications reach human taxonomists ahead of trivial cases.

    How:
        Delegates multi-class probability vectors to `margin_sampling()`, handles degenerate single-class probabilities via `1.0 - p_0`, and returns `0.0` when probabilities are empty.

    Args:
        probabilities: Softmax class probability distribution for the observation.

    Returns:
        Margin uncertainty scalar bounded in `[0.0, 1.0]`.
    """
    if len(probabilities) >= 2:
        prob_arr = np.asarray(probabilities, dtype=np.float32).reshape(1, -1)
        return float(margin_sampling(prob_arr)[0])
    if len(probabilities) == 1:
        return float(max(0.0, 1.0 - float(probabilities[0])))
    return 0.0


def _resolve_consent_record(consent: ContributorConsent | None) -> dict[str, Any]:
    """Build explicit licensing and attribution fields only when affirmative open-license consent passes validation.

    Why:
        Prevents accidental persistence of unverified photographer strings or disallowed licenses (`all-rights-reserved`) inside `{obs_id}_meta.json`.

    How:
        Evaluates `is_valid_contributor_consent(consent)`; returns `{"contribute_open_license": False}` on failure or normalised uppercase Creative Commons metadata and attribution on success.

    Args:
        consent: Optional `ContributorConsent` provided on upload.

    Returns:
        Dictionary fragment merged into the observation's `_meta.json` record.
    """
    if not is_valid_contributor_consent(consent) or consent is None:
        return {"contribute_open_license": False}
    license_norm = consent.license_code.strip().lower().replace("_", "-")
    return {
        "contribute_open_license": True,
        "license_code": license_norm.upper(),
        "photographer_name": consent.photographer_name.strip(),
        "license_url": consent.license_url.strip() or _OPEN_LICENSE_URLS[license_norm],
    }


def persist_flagged_observation(
    observation_id: str,
    prediction_result: dict[str, Any],
    embedding: npt.NDArray[np.float32] | None = None,
    storage_dir: Path | str | None = None,
    *,
    image_bytes: bytes | None = None,
    consent: ContributorConsent | None = None,
) -> Path:
    """Atomically persist metadata, image crop, and embeddings for observations flagged during inference.

    Why:
        Observations identified as Out-of-Distribution (OOD) or exhibiting high conformal ambiguity require human expert review and provide the highest information gain for active learning retrain loops. Staging the tabular decision metadata, dense visual feature embedding, margin uncertainty score, open-license opt-in consent, and raw visual image crop allows downstream active-learning workers (DagsHub Label Studio sync and dataset curation) to prioritise and ingest high-value edge cases whilst preserving strict copyright compliance.

    How:
        1. Resolves destination storage directory from arguments or application configuration.
        2. Computes margin-sampling uncertainty via `margin_sampling()` and validates optional contributor open-license metadata.
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
        consent: Optional ContributorConsent specifying affirmative Creative Commons opt-in and attribution.

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
    probabilities = [float(p) for p in prediction_result.get("probabilities", [])]
    uncertainty_score = _compute_margin_uncertainty(probabilities)

    record: dict[str, Any] = {
        "observation_id": observation_id,
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "ood_energy_score": float(prediction_result.get("ood_energy_score", 0.0)),
        "is_ood": bool(prediction_result.get("is_ood", False)),
        "requires_human_review": bool(prediction_result.get("requires_human_review", False)),
        "conformal_set": list(prediction_result.get("conformal_set", [])),
        "top_prediction": top_pred,
        "probabilities": probabilities,
        "uncertainty_score": round(uncertainty_score, 6),
        "embedding_path": embedding_rel_path,
        "image_path": image_rel_path,
        "synced_to_label_studio": False,
        **_resolve_consent_record(consent),
    }

    meta_path.write_text(json.dumps(record, indent=2), encoding="utf-8")

    manifest_path = target_dir / "manifest.jsonl"
    with manifest_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

    _TRIAGE_QUEUE.push(
        TriageItem(
            observation_id=observation_id,
            uncertainty_score=round(uncertainty_score, 6),
            conformal_set_size=len(record["conformal_set"]),
            image_url=image_rel_path or "",
        )
    )

    logger.info("Persisted flagged observation %s (crop=%s) to %s", observation_id, bool(image_bytes), target_dir)
    return meta_path


def queue_consented_triage_sync(
    background_tasks: Any,
    observation_id: str,
    meta_path: Path,
    consent: ContributorConsent | None,
) -> bool:
    """Schedule non-blocking DagsHub Label Studio sync when valid open-license consent exists.

    Why:
        Decouples external S3 and Label Studio REST uploads from the synchronous `/predict` execution path while ensuring unconsented captures remain strictly local.

    How:
        Checks `is_valid_contributor_consent(consent)` and local crop existence before registering `background_sync_flagged_observation` on the FastAPI `BackgroundTasks` queue.

    Args:
        background_tasks: FastAPI BackgroundTasks container or None.
        observation_id: Unique observation UUID string.
        meta_path: Path to the persisted `{observation_id}_meta.json` file.
        consent: Optional ContributorConsent provided by the uploader.

    Returns:
        True when a background synchronisation task was scheduled, False otherwise.
    """
    if background_tasks is None or not is_valid_contributor_consent(consent):
        return False
    crop_path = meta_path.parent / f"{observation_id}_crop.jpg"
    if not crop_path.is_file():
        return False
    from taxon_vision.active_learning.label_studio_sync import background_sync_flagged_observation

    background_tasks.add_task(background_sync_flagged_observation, meta_path, crop_path)
    return True


async def handle_flagged_prediction_if_needed(
    payload: FlaggedObservationPayload,
    background_tasks: Any = None,
) -> Path | None:
    """Persist flagged OOD or ambiguous captures asynchronously and schedule consented Label Studio upload.

    Why:
        Consolidates the asynchronous threadpool disk offload and conditional DagsHub Label Studio background queueing shared by REST `/predict` and HTMX `/ui/predict-htmx` routes, protecting the event loop under Rule 08.

    How:
        Verifies whether `is_ood` or `requires_human_review` is set, offloads `persist_flagged_observation` to a worker thread via `anyio.to_thread.run_sync`, and invokes `queue_consented_triage_sync`.

    Args:
        payload: FlaggedObservationPayload containing prediction results, image bytes, and optional consent.
        background_tasks: Optional FastAPI BackgroundTasks instance for non-blocking remote sync.

    Returns:
        Path to the written metadata JSON file when flagged, or None when no review was required.
    """
    if not (payload.prediction_result.get("is_ood") or payload.prediction_result.get("requires_human_review")):
        return None
    meta_path = await anyio.to_thread.run_sync(
        partial(
            persist_flagged_observation,
            payload.observation_id,
            payload.prediction_result,
            payload.embedding,
            payload.storage_dir,
            image_bytes=payload.image_bytes,
            consent=payload.consent,
        )
    )
    queue_consented_triage_sync(background_tasks, payload.observation_id, meta_path, payload.consent)
    return meta_path
