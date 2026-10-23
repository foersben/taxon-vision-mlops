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
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from taxon_vision.config import get_settings

logger = logging.getLogger(__name__)


class EphemeralImageCache:
    """Thread-safe, bounded in-memory cache of uploaded observation images.

    Why:
        HTMX is stateless: the multipart upload stream is consumed by `/ui/predict-htmx`, so a follow-up
        request (such as the CAM button) cannot re-submit the file and would otherwise fail with
        HTTP 422. Persisting temporary uploads to disk would cause avoidable SSD write amplification
        and leak potentially sensitive imagery, so the bytes are held in RAM only and addressed by
        their SHA-256 content hash.

    How:
        Entries live in an `OrderedDict` ordered by recency of use. On insertion, expired entries
        (older than `ttl_seconds`, measured with the monotonic clock) are pruned first and the least
        recently used entry is evicted if `max_items` would be exceeded. A `threading.Lock` guards all
        mutations because FastAPI executes synchronous work in a worker thread pool.

    Complexity:
        `store` and `get` are O(1) amortised; pruning in `store` is O(n) in the number of entries,
        which is bounded by `max_items`.

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


@lru_cache(maxsize=1)
def get_image_cache() -> EphemeralImageCache:
    """Return the process-wide ephemeral image cache, configured from application settings.

    Why:
        A lazily constructed singleton avoids reading settings at import time and can be reset in
        tests via `get_image_cache.cache_clear()`.

    Returns:
        The shared `EphemeralImageCache` instance.
    """
    service = get_settings().service
    return EphemeralImageCache(
        max_items=service.image_cache_max_items,
        ttl_seconds=service.image_cache_ttl_seconds,
    )


def persist_flagged_observation(
    observation_id: str,
    prediction_result: dict[str, Any],
    embedding: npt.NDArray[np.float32] | None = None,
    storage_dir: Path | str | None = None,
) -> Path:
    """Atomically persist metadata and embeddings for observations flagged during inference.

    Why:
        Observations identified as Out-of-Distribution (OOD) or exhibiting high conformal ambiguity require human expert review and provide the highest information gain for active learning retrain loops. Staging both the tabular decision metadata and the dense visual feature embedding allows subsequent ingestion services (e.g. SeaweedFS, LanceDB, or triage queues) to consume high-priority edge cases without recomputing vision foundation forward passes.

    How:
        1. Resolves destination storage directory from arguments or application configuration.
        2. Compiles a consolidated record containing prediction metrics and timestamp.
        3. If an embedding vector is provided, serializes it to disk as an optimized `.npy` file.
        4. Serializes the metadata record as an individual JSON artifact and appends a line to `manifest.jsonl`.

    Complexity:
        O(D) where D is the dimensionality of the feature embedding vector.

    Args:
        observation_id: Unique identifier for the observation capture.
        prediction_result: Dictionary returned by the prediction pipeline containing classification results.
        embedding: Optional dense feature representation vector of shape (D,).
        storage_dir: Optional filesystem path override for the staging directory.

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
    }

    meta_path.write_text(json.dumps(record, indent=2), encoding="utf-8")

    manifest_path = target_dir / "manifest.jsonl"
    with manifest_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

    logger.info("Persisted flagged observation %s to %s", observation_id, target_dir)
    return meta_path
