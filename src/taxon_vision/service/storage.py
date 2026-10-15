# SPDX-FileCopyrightText: 2026 Benjamin Förster
# SPDX-License-Identifier: MIT
"""Storage service for persisting out-of-distribution and ambiguous observations."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from taxon_vision.config import get_settings

logger = logging.getLogger(__name__)


def persist_flagged_observation(
    observation_id: str,
    prediction_result: dict[str, Any],
    embedding: npt.NDArray[np.float32] | None = None,
    storage_dir: Path | str | None = None,
) -> Path:
    """Atomically persist metadata and embeddings for observations flagged during inference.

    Why:
        Observations identified as Out-of-Distribution (OOD) or exhibiting high conformal ambiguity require human expert review and provide the highest information gain for active learning retrain loops. Staging both the tabular decision metadata and the dense visual feature embedding allows subsequent ingestion services (e.g. MinIO, LanceDB, or triage queues) to consume high-priority edge cases without recomputing vision foundation forward passes.

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
