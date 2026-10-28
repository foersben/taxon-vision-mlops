# SPDX-FileCopyrightText: 2026 Benjamin Förster
# SPDX-License-Identifier: MIT
"""Dataset and feature embedding split loading, disk caching, and synthetic fallbacks."""

from __future__ import annotations

import logging
from pathlib import Path

import torch
import yaml

from taxon_vision.config import get_settings
from taxon_vision.models.factory import create_feature_extractor, get_feature_dimension
from taxon_vision.models.training.types import EmbeddingSplit

logger = logging.getLogger(__name__)


def get_dvc_dataset_hash(lock_path: Path | str = "dvc.lock") -> str | None:
    """Parse dvc.lock to extract the dataset hash from the ingestion stage.

    Why:
        In auditable MLOps pipelines, every trained checkpoint must be cryptographically linked to the exact immutable dataset snapshot used during training. Extracting the DVC manifest hash provides end-to-end lineage tracking from raw observations to deployed models.

    How:
        Reads `dvc.lock`, parses the YAML stage graph, extracts the output MD5 checksum associated with `data/manifests`, and returns the hex digest string.

    Args:
        lock_path: Path to dvc.lock file (default: "dvc.lock").

    Returns:
        The MD5 hash of the data/manifests directory, or None if unavailable.
    """
    try:
        dvc_lock_path = Path(lock_path)
        if not dvc_lock_path.exists():
            logger.debug("dvc.lock not found at '%s'; skipping DVC hash logging.", dvc_lock_path)
            return None

        with dvc_lock_path.open("r", encoding="utf-8") as f:
            dvc_data = yaml.safe_load(f)

        if not isinstance(dvc_data, dict):
            return None

        stages = dvc_data.get("stages", {})
        if not isinstance(stages, dict):
            return None

        ingestion = stages.get("ingestion", {})
        if not isinstance(ingestion, dict):
            return None

        outs = ingestion.get("outs", [])
        if not isinstance(outs, list):
            return None

        for out in outs:
            if isinstance(out, dict) and out.get("path") == "data/manifests":
                md5_val = out.get("md5")
                return str(md5_val) if md5_val is not None else None

        logger.debug("data/manifests output not found in dvc.lock")
    except Exception as e:
        logger.debug("Could not parse dvc.lock for dataset hashing: %s", e)
    return None


def resolve_feature_dim(extractor: str) -> int:
    """Resolve embedding dimension for the specified backbone extractor.

    Args:
        extractor: Name of the vision foundation backbone.

    Returns:
        Integer dimensionality of the extracted representation vectors.
    """
    try:
        backbone = create_feature_extractor(extractor, pretrained=False, allow_fallback=True)
        return get_feature_dimension(backbone)
    except Exception as err:
        logger.error("Could not resolve backbone feature dimension for '%s': %s", extractor, err)
        raise RuntimeError(f"Could not determine feature dimension for backbone '{extractor}': {err}") from err


def _try_load_cached_split(candidate: Path, dim: int) -> EmbeddingSplit | None:
    """Attempt to deserialize and validate an embedding split from a candidate filesystem path.

    Args:
        candidate: File path pointing to a potential serialized `.pt` embedding cache.
        dim: Expected feature projection dimension.

    Returns:
        Validated EmbeddingSplit instance if compatible, otherwise None.
    """
    if not candidate.exists():
        return None

    try:
        split = EmbeddingSplit.from_cache(candidate)
        if split.feature_dim == dim:
            logger.info(
                "Loaded pre-computed feature embeddings from '%s' (%d train, %d val samples, dim=%d)",
                candidate,
                split.train_embeddings.shape[0],
                split.val_embeddings.shape[0],
                split.feature_dim,
            )
            return split
        logger.warning(
            "Candidate cache '%s' has incompatible feature dimension (expected %d, got %d); skipping",
            candidate,
            dim,
            split.feature_dim,
        )
    except Exception as err:
        logger.warning("Could not deserialize cache '%s' (%s); falling back", candidate, err)

    return None


def _generate_synthetic_embedding_split(dim: int, num_classes: int) -> EmbeddingSplit:
    """Generate deterministic synthetic Gaussian embeddings for rapid offline testing.

    Args:
        dim: Dimensionality of feature vectors.
        num_classes: Total number of classification categories.

    Returns:
        Deterministic EmbeddingSplit seeded with PRNG seed from configuration.
    """
    seed = get_settings().training.random_seed
    logger.info("Initializing deterministic synthetic embedding split (seed=%d, %d dim)", seed, dim)
    torch.manual_seed(seed)
    num_train = 120
    num_val = 30
    num_cal = 20
    num_test = 20
    train_z = torch.randn(num_train, dim)
    train_y = torch.randint(0, num_classes, (num_train,))
    val_z = torch.randn(num_val, dim)
    val_y = torch.randint(0, num_classes, (num_val,))
    cal_z = torch.randn(num_cal, dim)
    cal_y = torch.randint(0, num_classes, (num_cal,))
    test_z = torch.randn(num_test, dim)
    test_y = torch.randint(0, num_classes, (num_test,))
    samples_per_class = [max(1, int((train_y == c).sum().item())) for c in range(num_classes)]

    return EmbeddingSplit(
        train_embeddings=train_z,
        train_labels=train_y,
        val_embeddings=val_z,
        val_labels=val_y,
        cal_embeddings=cal_z,
        cal_labels=cal_y,
        test_embeddings=test_z,
        test_labels=test_y,
        samples_per_class=samples_per_class,
    )


def _try_load_split_from_pair(train_path: Path, val_path: Path, dim: int) -> EmbeddingSplit | None:
    """Load train and validation partitions directly from discrete .pt tensor files.

    Args:
        train_path: Path to train_embeddings.pt.
        val_path: Path to val_embeddings.pt.
        dim: Expected feature dimension.

    Returns:
        Instantiated EmbeddingSplit or None if missing or incompatible.
    """
    if not train_path.exists() or not val_path.exists():
        return None
    try:
        train_payload = torch.load(train_path, map_location="cpu", weights_only=False)
        val_payload = torch.load(val_path, map_location="cpu", weights_only=False)

        train_z = train_payload["embeddings"] if isinstance(train_payload, dict) else train_payload
        train_y = train_payload["labels"] if isinstance(train_payload, dict) else None
        samples_per_class = train_payload.get("samples_per_class") if isinstance(train_payload, dict) else None
        feat_dim = (
            train_payload.get("feature_dim", train_z.shape[1]) if isinstance(train_payload, dict) else train_z.shape[1]
        )

        val_z = val_payload["embeddings"] if isinstance(val_payload, dict) else val_payload
        val_y = val_payload["labels"] if isinstance(val_payload, dict) else None

        if not isinstance(train_z, torch.Tensor) or not isinstance(train_y, torch.Tensor):
            return None
        if not isinstance(val_z, torch.Tensor) or not isinstance(val_y, torch.Tensor):
            return None
        if feat_dim != dim:
            logger.warning("Split embeddings at %s have dimension %d, expected %d", train_path, feat_dim, dim)
            return None

        split = EmbeddingSplit(
            train_embeddings=train_z,
            train_labels=train_y,
            val_embeddings=val_z,
            val_labels=val_y,
            samples_per_class=samples_per_class,
        )
        logger.info(
            "Loaded isolated train (%d) and val (%d) embeddings from %s, %s (dim=%d)",
            len(train_z),
            len(val_z),
            train_path,
            val_path,
            dim,
        )
        return split
    except Exception as err:
        logger.warning("Failed loading split tensors from %s / %s: %s", train_path, val_path, err)
        return None


def resolve_synthetic_permission(
    require_cache: bool,
    allow_synthetic: bool | None,
    default_fallback: bool,
) -> bool:
    """Determine whether synthetic embedding generation is permitted.

    Args:
        require_cache: Strict requirement for real embedding cache.
        allow_synthetic: Optional explicit user flag.
        default_fallback: Global fallback setting.

    Returns:
        True if synthetic generation is authorized; otherwise False.
    """
    if require_cache:
        return False
    if allow_synthetic is not None:
        return allow_synthetic
    return default_fallback


def load_or_create_training_split(
    dim: int,
    num_classes: int,
    cache_path: Path | str | None = None,
    train_embeddings_path: Path | str | None = None,
    val_embeddings_path: Path | str | None = None,
    allow_synthetic: bool = False,
) -> EmbeddingSplit:
    """Construct or restore deterministic embedding splits for rapid head training.

    Why:
        Stage 3 (`extract_features.py`) materializes split-isolated tensor partitions (`train_embeddings.pt`, `val_embeddings.pt`) in production while unit and smoke tests may pass composite caches or rely on deterministic synthetic Gaussian splits. Centralizing split resolution enforces feature dimensionality checks and prevents silent fallback to synthetic random noise unless `allow_synthetic=True` is explicitly authorized.

    How:
        1. Probes explicit `train_embeddings_path` / `val_embeddings_path` pair or canonical `data/embeddings/{train,val}_embeddings.pt` files.
        2. Falls back to composite `cache_path` (or `data/embeddings/cached_features.pt`) via `_try_load_cached_split`.
        3. Raises `FileNotFoundError` when no valid cache matches `dim` and `allow_synthetic` is False; otherwise generates a seeded Gaussian split.

    Complexity:
        O(N * D) disk deserialization and memory allocation for N total samples of dimension D.

    Args:
        dim: Dimensionality of feature vectors.
        num_classes: Total number of classification categories.
        cache_path: Optional custom filesystem path to a serialized `.pt` embedding cache.
        train_embeddings_path: Optional custom path to train_embeddings.pt.
        val_embeddings_path: Optional custom path to val_embeddings.pt.
        allow_synthetic: Whether to fall back to synthetic Gaussian embeddings if cache is missing.

    Returns:
        Pre-allocated and validated EmbeddingSplit data container.

    Raises:
        FileNotFoundError: If cache is absent and allow_synthetic is False.
    """
    if train_embeddings_path is not None and val_embeddings_path is not None:
        pair_loaded = _try_load_split_from_pair(Path(train_embeddings_path), Path(val_embeddings_path), dim)
        if pair_loaded is not None:
            return pair_loaded

    default_train = Path("data/embeddings/train_embeddings.pt")
    default_val = Path("data/embeddings/val_embeddings.pt")
    if cache_path is None and default_train.exists() and default_val.exists():
        pair_loaded = _try_load_split_from_pair(default_train, default_val, dim)
        if pair_loaded is not None:
            return pair_loaded

    candidates: list[Path] = (
        [Path(cache_path)] if cache_path is not None else [Path("data/embeddings/cached_features.pt")]
    )

    for candidate in candidates:
        loaded = _try_load_cached_split(candidate, dim)
        if loaded is not None:
            return loaded

    if not allow_synthetic:
        logger.error(
            "Embedding cache not found at candidates (%s). Synthetic fallback is disabled.",
            candidates,
        )
        raise FileNotFoundError(
            f"Embedding cache not found at candidates ({candidates}). Synthetic fallback is disabled via --require-cache."
        )

    logger.warning("Embedding cache not found; generating synthetic Gaussian split for testing.")
    return _generate_synthetic_embedding_split(dim, num_classes)
