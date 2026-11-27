# SPDX-FileCopyrightText: 2026 Benjamin Förster
# SPDX-License-Identifier: MIT
"""Dataset cryptographic lineage and training metric trajectory loggers for MLflow.

Summary:
    Records cryptographic SHA-256 dataset digests, Pandas open-manifest lineage, NumPy split tensor inputs, resolved hyperparameters, conformal energy calibration thresholds, and per-epoch training trajectories into active MLflow runs.

Why (Architectural Rationale & Trade-offs):
    Decoupling dataset schema binding (`mlflow.data`) and training trajectory serialisation from RAII run lifecycle management keeps each registry sibling compact (< 250 LOC) and allows independent verification of lineage bindings.

How (Algorithmic Mechanism & Invariants):
    Streams Parquet manifest bytes in 64 KiB chunks to compute SHA-256 digests in O(1) memory, registers split tensors via `mlflow.data.from_numpy`, and emits epoch-indexed metrics alongside MLflow 3 `model_id` / `dataset` evaluation bindings.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any

import mlflow
import mlflow.data
import pandas as pd

from taxon_vision.models.training.types import EmbeddingSplit
from taxon_vision.registry.types import LoggedDatasets, safe_mlflow_call

logger = logging.getLogger(__name__)


def _compute_file_sha256(path: Path) -> str:
    """Compute the cryptographic SHA-256 hex digest of a file on disk using bounded chunk streaming."""
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def _log_manifest_dataset(manifest_path: Path | str | None) -> Any | None:
    """Register the Parquet open-access manifest with SHA-256 and licence policy tags in MLflow."""
    if manifest_path is None:
        return None
    path = Path(manifest_path)
    if not path.is_file():
        return None
    try:
        df = pd.read_parquet(path)
        sha256 = _compute_file_sha256(path)
        target_col = "taxon_id" if "taxon_id" in df.columns else None
        manifest_ds = mlflow.data.from_pandas(  # type: ignore[attr-defined]
            df,
            source=str(path),
            name="inaturalist_open_manifest",
            targets=target_col,
        )
        mlflow.log_input(
            manifest_ds,
            context="manifest",
            tags={
                "sha256": sha256,
                "license.policy": "CC0,CC-BY,CC-BY-NC",
                "records_count": str(len(df)),
            },
        )
        return manifest_ds
    except Exception as err:
        logger.warning("Failed to log manifest lineage to MLflow: %s", err)
        return None


def _log_numpy_dataset(embeddings: Any, labels: Any, name: str, context: str) -> Any | None:
    """Register a feature and label partition tensor as an MLflow input dataset."""
    if embeddings is None or labels is None:
        return None
    dataset = mlflow.data.from_numpy(  # type: ignore[attr-defined]
        features=embeddings.detach().cpu().numpy(),
        targets=labels.detach().cpu().numpy(),
        name=name,
    )
    mlflow.log_input(dataset, context=context)
    return dataset


def log_dataset_lineage(
    data: EmbeddingSplit,
    manifest_path: Path | str | None = Path("data/manifests/dataset_manifest.parquet"),
) -> LoggedDatasets:
    """Log dataset input lineage metadata (Parquet manifest and split tensors) into the active MLflow run.

    Why:
        Preserves cryptographic data lineage and open-licence attribution metadata on the MLflow Model Card while returning dataset handles for MLflow 3 evaluation binding.

    How:
        1. Registers `dataset_manifest.parquet` via `mlflow.data.from_pandas` with SHA-256 digest and CC licence tags.
        2. Registers training and validation embeddings via `mlflow.data.from_numpy`.
        3. Registers optional calibration and holdout test splits when available.
        4. Returns an immutable `LoggedDatasets` container.

    Args:
        data: Pre-loaded `EmbeddingSplit` containing training, validation, calibration, and test tensors.
        manifest_path: Optional filesystem path to `dataset_manifest.parquet`.

    Returns:
        `LoggedDatasets` container holding registered MLflow dataset handles.
    """
    if not mlflow.active_run():
        return LoggedDatasets(train=None, val=None)

    manifest_ds = _log_manifest_dataset(manifest_path)
    train_ds = None
    val_ds = None
    cal_ds = None
    test_ds = None

    try:
        train_ds = _log_numpy_dataset(data.train_embeddings, data.train_labels, "train_embeddings", "training")
        val_ds = _log_numpy_dataset(data.val_embeddings, data.val_labels, "val_embeddings", "validation")
        cal_ds = _log_numpy_dataset(data.cal_embeddings, data.cal_labels, "cal_embeddings", "calibration")
        test_ds = _log_numpy_dataset(data.test_embeddings, data.test_labels, "test_embeddings", "test")
    except Exception as err:
        logger.warning("Failed to log tensor split lineage to MLflow: %s", err)

    return LoggedDatasets(
        train=train_ds,
        val=val_ds,
        manifest=manifest_ds,
        cal=cal_ds,
        test=test_ds,
    )


def _log_epoch_trajectory(history: dict[str, list[float]]) -> int:
    """Emit per-epoch training metric trajectories to the active MLflow run and return completed epoch count."""
    epochs = max((len(vals) for vals in history.values()), default=0)
    for epoch in range(epochs):
        step_metrics = {k: v[epoch] for k, v in history.items() if epoch < len(v)}
        mlflow.log_metrics(step_metrics, step=epoch + 1)
    return epochs


def _log_bound_evaluation_metrics(
    history: dict[str, list[float]],
    epochs: int,
    model_id: str | None,
    dataset: Any | None,
) -> None:
    """Emit terminal summary metrics bound to MLflow 3 `model_id` and validation `dataset` lineage."""
    if not model_id and not dataset:
        return
    final_metrics = {f"final_{k}": v[-1] for k, v in history.items() if v}
    extra_kwargs: dict[str, Any] = {}
    if model_id:
        extra_kwargs["model_id"] = model_id
    if dataset:
        extra_kwargs["dataset"] = dataset
    safe_mlflow_call(
        "log_bound_evaluation_metrics",
        lambda: mlflow.log_metrics(final_metrics, step=epochs, **extra_kwargs),
    )


def log_mlflow_metrics(
    history: dict[str, list[float]],
    duration: float,
    *,
    model_id: str | None = None,
    dataset: Any | None = None,
) -> None:
    """Log training metric trajectories, wall-clock duration, and bound final metrics to the active run.

    Args:
        history: Mapping from metric keys to lists of per-epoch float measurements.
        duration: Wall-clock duration of training in seconds.
        model_id: Optional MLflow 3 LoggedModel ID bound to terminal metrics.
        dataset: Optional MLflow validation dataset handle bound to terminal metrics.
    """

    def _action() -> None:
        epochs = _log_epoch_trajectory(history)
        mlflow.log_metric("duration_seconds", duration)
        _log_bound_evaluation_metrics(history, epochs, model_id, dataset)

    safe_mlflow_call("log_mlflow_metrics", _action)


def log_training_hyperparameters(params: dict[str, Any]) -> None:
    """Record resolved training hyperparameters and provenance tag to the active MLflow run.

    Args:
        params: Resolved hyperparameter mapping from training setup.
    """

    def _action() -> None:
        mlflow.log_params(params)
        if "hyperparameter_source" in params:
            mlflow.set_tag("hyperparameter_source", str(params["hyperparameter_source"]))

    safe_mlflow_call("log_training_hyperparameters", _action)


def log_calibrated_energy_threshold(threshold: float) -> None:
    """Record the calibrated OOD Helmholtz free energy threshold to the active MLflow run.

    Args:
        threshold: Calibrated scalar threshold value.
    """
    safe_mlflow_call(
        "log_calibrated_energy_threshold",
        lambda: mlflow.log_metric("calibrated_ood_energy_threshold", threshold),
    )
