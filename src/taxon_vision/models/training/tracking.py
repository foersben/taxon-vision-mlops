# SPDX-FileCopyrightText: 2026 Benjamin Förster
# SPDX-License-Identifier: MIT
"""MLOps tracking lifecycle and promotion utilities."""

from __future__ import annotations

import contextlib
import copy
import hashlib
import json
import logging
import os
import subprocess
from collections.abc import Callable, Generator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import onnx
import pandas as pd
from mlflow.models import ModelSignature
from mlflow.models.model import ModelInfo
from mlflow.tracking import MlflowClient
from mlflow.types.schema import Schema, TensorSpec
from torch import nn

from taxon_vision.config import Settings, get_settings
from taxon_vision.models.training.dataset import get_dvc_dataset_hash
from taxon_vision.models.training.promotion import (
    ModelDimensions,
    _apply_model_version_metadata as _apply_model_version_metadata,
    _determine_promotion_alias as _determine_promotion_alias,
    _fetch_run_safely as _fetch_run_safely,
    _get_alias_version_safe as _get_alias_version_safe,
    _get_incumbent_score as _get_incumbent_score,
    _log_model_artifact as _log_model_artifact,
    _prepare_model_signature_and_example as _prepare_model_signature_and_example,
    _promote_model_if_better as _promote_model_if_better,
)
from taxon_vision.models.training.types import EmbeddingSplit
from taxon_vision.models.training.visualizations import (
    _plot_confusion_matrix as _plot_confusion_matrix,
    _plot_energy_distribution as _vis_plot_energy_distribution,
    _plot_precision_recall_curves as _plot_precision_recall_curves,
    log_training_visualizations as log_training_visualizations,
)

logger = logging.getLogger(__name__)

__all__ = [
    "LoggedDatasets",
    "MLflowTracker",
    "ParentRunContext",
    "mlflow_run_scope",
    "resume_parent_mlflow_run",
]


@dataclass(frozen=True)
class ParentRunContext:
    """Immutable context bundle extracted from reports/train_metrics.json for cross-stage MLflow run resumption.

    Why:
        Encapsulates the parent MLflow run and model version identifiers to prevent tuple blindness when downstream DVC pipeline stages (ONNX export, conformal calibration, evaluation) re-enter the parent training run context.

    How:
        Provides typed fields for run_id, model_id, extractor name, and status read directly from Stage 4 serialized JSON metrics.

    Attributes:
        run_id: The MLflow run ID of the parent training stage.
        model_id: The MLflow model version ID.
        extractor: The extractor name.
        status: The status of the training stage.
    """

    run_id: str
    model_id: str | None
    extractor: str
    status: str


@dataclass(frozen=True)
class LoggedDatasets:
    """Container of datasets registered with MLflow during training lineage logging.

    Attributes:
        train: Logged MLflow dataset instance for the training partition.
        val: Logged MLflow dataset instance for the validation partition.
        manifest: Logged MLflow dataset instance for the open manifest, or None.
        cal: Optional logged MLflow dataset instance for calibration partition.
        test: Optional logged MLflow dataset instance for holdout test partition.
    """

    train: Any | None
    val: Any | None
    manifest: Any | None = None
    cal: Any | None = None
    test: Any | None = None


def _plot_energy_distribution(
    id_logits: np.ndarray,
    energy_threshold: float,
) -> Any:
    """Plot histogram and KDE distribution of negative energy scores.

    Backward compatibility delegation function for OKF claims and tests.
    """
    return _vis_plot_energy_distribution(id_logits, energy_threshold)


def _safe_mlflow_call[T](action_name: str, fn: Callable[[], T], default: T | None = None) -> T | None:
    """Execute a non-critical MLflow action, suppressing transient network or API errors.

    Args:
        action_name: Short human-readable identifier for the operation being performed.
        fn: Zero-argument callable executing the MLflow operation.
        default: Fallback return value if an exception occurs.

    Returns:
        The result of fn(), or default if an exception was raised.
    """
    try:
        return fn()
    except Exception as err:
        logger.debug("Non-fatal MLflow error during %s: %s", action_name, err)
        return default


def _end_mlflow_run_safely(status: str) -> None:
    """End the MLflow run safely, logging any errors to debug.

    Args:
        status: The status to set the run to (e.g., "FAILED", "FINISHED").
    """
    _safe_mlflow_call(f"end_run({status})", lambda: mlflow.end_run(status=status))


def _attach_mlflow_artifact_safely(log_file: Path | str | None) -> None:
    """Attach a log file as an MLflow artifact safely.

    Args:
        log_file: Path to the log file to attach.
    """
    if not log_file or not Path(log_file).exists():
        return
    _safe_mlflow_call("log_artifact", lambda: mlflow.log_artifact(str(log_file)))


def _inject_lineage_tags() -> None:
    """Inject Git commit lineage and CI/CD orchestration tags into the active MLflow run."""
    try:
        git_sha = os.environ.get("GIT_COMMIT")
        if not git_sha:
            git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        git_branch = os.environ.get("GIT_BRANCH")
        if not git_branch:
            git_branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], text=True).strip()
        mlflow.set_tag("git.commit_sha", git_sha)
        mlflow.set_tag("git.branch", git_branch)

        ci_mappings = (
            ("BUILD_TAG", "ci.jenkins_build_tag"),
            ("BUILD_URL", "ci.jenkins_build_url"),
            ("GITHUB_RUN_ID", "ci.github_run_id"),
            ("GITHUB_WORKFLOW", "ci.github_workflow"),
        )
        for env_var, tag_key in ci_mappings:
            val = os.environ.get(env_var)
            if val:
                mlflow.set_tag(tag_key, val)
    except Exception as git_err:
        logger.debug("Could not resolve git lineage tags: %s", git_err)


def _ensure_dagshub_auth(uri: str) -> bool:
    """Populate MLflow Basic Auth environment credentials for DagsHub tracking URIs when a token is available.

    Args:
        uri: Candidate MLflow tracking URI.

    Returns:
        True if a non-empty DagsHub token is configured, False otherwise.
    """
    dagshub_token = os.getenv("DAGSHUB_TOKEN") or os.getenv("MLFLOW_TRACKING_PASSWORD")
    if not dagshub_token:
        return False
    if "dagshub.com" in uri:
        os.environ.setdefault("MLFLOW_TRACKING_USERNAME", "foersben")
        os.environ.setdefault("MLFLOW_TRACKING_PASSWORD", dagshub_token)
    return True


def _resolve_reachable_tracking_uri(settings: Settings | None = None) -> str:
    """Resolve the active MLflow tracking URI targeting DagsHub or an explicit environment override.

    Summary:
        Determines the active MLflow tracking URI from `MLFLOW_TRACKING_URI` or `cfg.mlflow.tracking_uri` and skips cleanly without creating local SQLite databases or `mlruns` directories when credentials are absent.

    Why:
        Experiment tracking is consolidated exclusively onto the managed DagsHub MLflow server (`https://dagshub.com/foersben/taxon-vision-mlops.mlflow`) to eliminate duplicated in-cluster database state and cross-server run mirroring. Unauthenticated offline environments or dry-runs must skip tracking gracefully rather than writing orphaned local SQLite artefacts.

    How:
        1. Checks `MLFLOW_TRACKING_URI` if explicitly set in the environment: an empty string disables tracking immediately; non-empty URIs inject DagsHub Basic Auth credentials when targeting `dagshub.com` and return `""` if unauthenticated.
        2. Otherwise inspects `cfg.mlflow.tracking_uri`: validates DagsHub credentials via `_ensure_dagshub_auth` when targeting `dagshub.com`, returning the URI if authenticated or logging a debug message and returning `""` when unauthenticated.

    Args:
        settings: Application settings instance. If None, resolved via get_settings().

    Returns:
        Configured MLflow tracking URI string, or an empty string when tracking is disabled or unauthenticated.
    """
    env_uri = os.getenv("MLFLOW_TRACKING_URI")
    if env_uri is not None:
        cleaned_env = env_uri.strip()
        if not cleaned_env:
            return ""
        if "dagshub.com" in cleaned_env and not _ensure_dagshub_auth(cleaned_env):
            return ""
        return cleaned_env

    cfg = settings or get_settings()
    configured_uri = (cfg.mlflow.tracking_uri or "").strip()
    if not configured_uri:
        return ""
    if "dagshub.com" in configured_uri and not _ensure_dagshub_auth(configured_uri):
        logger.debug("Skipping DagsHub MLflow tracking (%s): credentials unavailable.", configured_uri)
        return ""
    return configured_uri


def _open_mlflow_run(
    run_name: str,
    params: dict[str, Any] | None,
    nested: bool,
    experiment_name: str | None,
) -> Any:
    """Resolve DagsHub tracking URI, configure experiment, start run, and log initial metadata."""
    settings = get_settings()
    tracking_uri = _resolve_reachable_tracking_uri(settings)
    if not tracking_uri:
        return None
    mlflow.set_tracking_uri(tracking_uri)
    target_exp = experiment_name or settings.mlflow.experiment_name
    if target_exp:
        mlflow.set_experiment(target_exp)
    active_run = mlflow.start_run(run_name=run_name, nested=True) if nested else mlflow.start_run(run_name=run_name)
    if params:
        mlflow.log_params(params)
    dataset_hash = get_dvc_dataset_hash()
    if dataset_hash:
        mlflow.log_param("dvc_dataset_hash", dataset_hash)
    _inject_lineage_tags()
    return active_run


@contextmanager
def mlflow_run_scope(
    run_name: str,
    params: dict[str, Any] | None = None,
    *,
    log_file: Path | str | None = None,
    nested: bool = False,
    experiment_name: str | None = None,
) -> Generator[mlflow.ActiveRun | None, None, None]:
    """Provide RAII (Resource Acquisition Is Initialization) lifecycle management for MLflow experiment tracking.

    RAII is a programming paradigm in which the lifetime of a resource is tied to the lifetime of a scope. In this case, the resource is an MLflow run, and the scope is the context manager.

    Why:
        Orphaned experiment runs left in 'RUNNING' status pollute MLOps registries when training processes abort, crash, or hit out-of-memory errors. Passing boolean flags (`is_active: bool`) across subroutines causes tramp data code smells and fragile error handling. A context manager guarantees deterministic lifecycle boundaries: setting up tracking, logging hyperparameter metadata, persisting run artifacts, and safely closing the run - marking failures as FAILED and successful completions as FINISHED - even during unhandled exceptions.

    How:
        1. Configures MLflow tracking URI via 3-tier reachable resolution and experiment name from project settings or caller argument.
        2. Starts an active MLflow run with the given `run_name` (optionally nested).
        3. Logs initial training parameters and DVC dataset manifest hashes.
        4. Injects cryptographic Git lineage tags and CI/CD orchestration tags (Jenkins/GitHub Actions).
        5. Yields the active run object to the caller.
        6. If an unhandled exception occurs inside the block, marks the run status as FAILED via `mlflow.end_run(status="FAILED")` and re-raises.
        7. On normal exit, logs execution log file artifacts (if present) and marks the run as FINISHED via `mlflow.end_run(status="FINISHED")`.
        8. If MLflow tracking server is unreachable, catches connection errors gracefully, logs a debug notice, and yields None to allow offline training to proceed unimpeded.

    Args:
        run_name: Human-readable run name for identification in the MLflow UI.
        params: Optional dictionary of hyperparameter key-value pairs to log.
        log_file: Optional path to an execution log file to attach as an MLflow artifact.
        nested: Whether to start a nested child run within an existing active run scope.
        experiment_name: Optional target experiment name. If None, uses settings.mlflow.experiment_name.

    Yields:
        Active MLflow run instance if tracking succeeded, or None if unavailable.
    """
    active_run = None
    try:
        active_run = _open_mlflow_run(run_name, params, nested, experiment_name)
    except Exception as err:
        logger.debug("MLflow tracking setup skipped or unavailable: %s", err)
        active_run = None

    try:
        yield active_run
    except Exception:
        if active_run:
            _end_mlflow_run_safely("FAILED")
        raise
    else:
        if active_run:
            _attach_mlflow_artifact_safely(log_file)
            _end_mlflow_run_safely("FINISHED")


@contextmanager
def resume_parent_mlflow_run(
    train_metrics_path: Path | str = "reports/train_metrics.json",
) -> Generator[ParentRunContext | None, None, None]:
    """Provide RAII lifecycle context manager to re-enter a parent MLflow run across multi-process DVC stages.

    Why:
        DVC executes each pipeline stage in dvc.yaml as an isolated operating system process. When Stage 4 (head training) finishes, its Python interpreter exits and its MLflow run is closed. Downstream stages (Stage 5 export_onnx, Stage 6 calibrate, Stage 7 evaluate) need to attach artifacts (INT8 ONNX model, conformal parameters, holdout metrics) to the exact same parent MLflow run and LoggedModel card without scattering disconnected runs across the experiment tracker.

    How:
        1. Reads reports/train_metrics.json to recover mlflow_run_id and mlflow_model_id.
        2. Resolves the active DagsHub or environment-overridden tracking URI and skips cleanly when unauthenticated.
        3. Invokes mlflow.start_run(run_id=run_id) to reopen the parent run.
        4. Yields a typed ParentRunContext dataclass to the caller.
        5. Automatically closes the reopened run on block exit without altering its terminal FINISHED status.
        6. Safely yields None if train_metrics.json is missing, unparsable, or contains no run_id (allowing dry-runs and offline operations to continue).

    Args:
        train_metrics_path: File path to reports/train_metrics.json holding the parent run ID.

    Yields:
        ParentRunContext instance if run was successfully resumed, or None otherwise.
    """
    metrics_file = Path(train_metrics_path)
    if not metrics_file.exists():
        logger.debug("Parent metrics file %s does not exist; skipping MLflow run resumption.", metrics_file)
        yield None
        return

    try:
        data = json.loads(metrics_file.read_text(encoding="utf-8"))
    except Exception as read_err:
        logger.warning("Could not parse %s for MLflow run resumption: %s", metrics_file, read_err)
        yield None
        return

    run_id = data.get("mlflow_run_id")
    if not run_id:
        logger.debug("No mlflow_run_id found in %s; skipping run resumption.", metrics_file)
        yield None
        return

    ctx = ParentRunContext(
        run_id=str(run_id),
        model_id=data.get("mlflow_model_id"),
        extractor=data.get("extractor", "unknown"),
        status=data.get("status", "unknown"),
    )

    settings = get_settings()
    tracking_uri = _resolve_reachable_tracking_uri(settings)
    if not tracking_uri:
        logger.debug("MLflow tracking URI unavailable or unauthenticated; skipping run resumption.")
        yield None
        return
    mlflow.set_tracking_uri(tracking_uri)

    active_run = None
    try:
        active_run = mlflow.start_run(run_id=ctx.run_id)
        if ctx.model_id:
            with contextlib.suppress(Exception):
                mlflow.set_active_model(model_id=ctx.model_id)
        logger.info("Resumed parent MLflow run %s for stage continuity", ctx.run_id)
    except Exception as start_err:
        logger.warning("Failed to resume parent MLflow run %s: %s", ctx.run_id, start_err)
        yield None
        return

    try:
        yield ctx
    finally:
        if active_run is not None:
            _end_mlflow_run_safely("FINISHED")


def _log_epoch_trajectory(history: dict[str, list[float]]) -> int:
    """Log per-epoch training metrics to active MLflow run and return total completed epochs.

    Args:
        history: Metric history dictionary.

    Returns:
        Total number of epochs completed.
    """
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
    """Log final summary metrics bound to MLflow 3 model_id and dataset evaluation split.

    Args:
        history: Metric history dictionary.
        epochs: Total completed epochs.
        model_id: Optional model version ID string.
        dataset: Optional evaluation dataset split.
    """
    if not model_id and not dataset:
        return
    final_metrics = {f"final_{k}": v[-1] for k, v in history.items() if v}
    extra_kwargs: dict[str, Any] = {}
    if model_id:
        extra_kwargs["model_id"] = model_id
    if dataset:
        extra_kwargs["dataset"] = dataset
    with contextlib.suppress(Exception):
        mlflow.log_metrics(final_metrics, step=epochs, **extra_kwargs)


def _log_mlflow_metrics(
    history: dict[str, list[float]],
    duration: float,
    *,
    model_id: str | None = None,
    dataset: Any | None = None,
) -> None:
    """Log training metric trajectories and duration to the active MLflow run.

    Why:
        Tracking per-epoch convergence curves (training loss, validation loss, validation accuracy, F1, and PR-AUC) in MLflow enables visualization of overfitting, learning rate decay effectiveness, and historical comparisons across training runs.

    How:
        Emits per-epoch trajectory metrics via _log_epoch_trajectory, logs total wall-clock duration, and binds final epoch metrics to the LoggedModel and validation dataset lineage.

    Args:
        history: Training history mapping metric keys to per-epoch scalar floats.
        duration: Total head training time in seconds.
        model_id: Optional MLflow 3 LoggedModel ID to bind metrics directly to the model card.
        dataset: Optional MLflow Dataset evaluation split to bind metrics to dataset lineage.
    """

    def _action() -> None:
        epochs = _log_epoch_trajectory(history)
        mlflow.log_metric("duration_seconds", duration)
        _log_bound_evaluation_metrics(history, epochs, model_id, dataset)

    _safe_mlflow_call("log_mlflow_metrics", _action)


def _compute_file_sha256(path: Path) -> str:
    """Compute the cryptographic SHA256 hex digest of a file on disk.

    Args:
        path: Path to the file to compute the digest for.

    Returns:
        SHA256 hex digest of the file.
    """
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def _log_manifest_dataset(manifest_path: Path | str | None) -> Any | None:
    """Register the Parquet manifest dataset with SHA256 lineage metadata into the active MLflow run."""
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
        logger.debug("Failed to log manifest lineage to MLflow: %s", err)
        return None


def _log_numpy_dataset(embeddings: Any, labels: Any, name: str, context: str) -> Any | None:
    """Register a feature and label split tensor as an MLflow input dataset."""
    if embeddings is None or labels is None:
        return None
    dataset = mlflow.data.from_numpy(  # type: ignore[attr-defined]
        features=embeddings.detach().cpu().numpy(),
        targets=labels.detach().cpu().numpy(),
        name=name,
    )
    mlflow.log_input(dataset, context=context)
    return dataset


def _log_dataset_lineage(
    data: EmbeddingSplit,
    manifest_path: Path | str | None = Path("data/manifests/dataset_manifest.parquet"),
) -> LoggedDatasets:
    """Log dataset input lineage metadata (Parquet manifest and split tensors) into the active MLflow run.

    Why:
        Preserves cryptographic data lineage and open-access license attribution metadata on the MLflow Model Card, while returning dataset references for MLflow 3 metric binding.

    How:
        1. Reads dataset_manifest.parquet and registers it via mlflow.data.from_pandas with SHA256 digest tag.
        2. Registers train_embeddings and val_embeddings via mlflow.data.from_numpy.
        3. Registers optional calibration and test splits if available.
        4. Returns an immutable LoggedDatasets container.

    Args:
        data: Pre-loaded EmbeddingSplit containing training and validation tensors.
        manifest_path: Optional path to dataset_manifest.parquet.

    Returns:
        LoggedDatasets container holding registered MLflow dataset objects.
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
        logger.debug("Failed to log tensor split lineage to MLflow: %s", err)

    return LoggedDatasets(
        train=train_ds,
        val=val_ds,
        manifest=manifest_ds,
        cal=cal_ds,
        test=test_ds,
    )


def _log_training_hyperparameters(params: dict[str, Any]) -> None:
    """Log resolved training hyperparameters and source tags to the active MLflow run.

    Args:
        params: Dictionary of resolved hyperparameters and execution parameters.
    """

    def _action() -> None:
        mlflow.log_params(params)
        if "hyperparameter_source" in params:
            mlflow.set_tag("hyperparameter_source", str(params["hyperparameter_source"]))

    _safe_mlflow_call("log_training_hyperparameters", _action)


def _log_calibrated_energy_threshold(threshold: float) -> None:
    """Log the calibrated 95th percentile OOD energy threshold to the active MLflow run.

    Args:
        threshold: Calibrated scalar Helmholtz free energy threshold.
    """
    _safe_mlflow_call(
        "log_calibrated_energy_threshold",
        lambda: mlflow.log_metric("calibrated_ood_energy_threshold", threshold),
    )


def _resolve_model_dims(
    model: nn.Module,
    input_dim: int | None,
    num_classes: int | None,
) -> ModelDimensions:
    """Resolve embedding dimension and class count from linear layer geometry.

    Args:
        model: PyTorch model module to inspect.
        input_dim: Optional explicit input dimension.
        num_classes: Optional explicit number of output classes.

    Returns:
        ModelDimensions of (input_dim, num_classes).
    """
    resolved_dim = input_dim or 1280
    resolved_classes = num_classes or 10
    linear_layers = [m for m in model.modules() if isinstance(m, nn.Linear)]
    if not linear_layers:
        return ModelDimensions(input_dim=resolved_dim, num_classes=resolved_classes)
    if input_dim is None:
        resolved_dim = linear_layers[0].in_features
    if num_classes is None:
        resolved_classes = linear_layers[-1].out_features
    return ModelDimensions(input_dim=resolved_dim, num_classes=resolved_classes)


def _log_pytorch_model_with_fallback(
    model: nn.Module,
    reg_name: str,
    signature: ModelSignature,
    input_example: np.ndarray,
    pip_reqs: list[str],
    *,
    params: dict[str, Any] | None = None,
    model_type: str = "classifier",
) -> ModelInfo | None:
    """Attempt default logging with fallback to pickle serialization.

    Why:
        MLflow 3 introduces `extra_kwargs` (passed via `**kwargs`) to `log_model` for binding metadata to model artifacts and datasets. This function implements a fallback strategy to accommodate legacy MLflow versions that reject unknown keyword arguments, ensuring robust model registration across environments.

    How:
        - Attempts default `mlflow.pytorch.log_model` with `extra_kwargs`.
        - On `TypeError` (unknown keyword arguments), falls back to `log_model` without `extra_kwargs`.
        - On other exceptions, logs debug message and retries with `serialization_format="pickle"` and `extra_kwargs`.

    Args:
        model: PyTorch model module on CPU.
        reg_name: Target registered model name.
        signature: ModelSignature instance.
        input_example: Input array example.
        pip_reqs: Required pip dependencies.
        params: Optional MLflow params dictionary.
        model_type: Optional model type string.

    Returns:
        ModelInfo instance on success, or None on failure.
    """
    if not isinstance(signature, ModelSignature):
        raise ValueError(
            f"Strict contract violation: signature must be an instance of ModelSignature, got {type(signature).__name__}"
        )
    extra_kwargs: dict[str, Any] = {}
    if params:
        extra_kwargs["params"] = params
    if model_type:
        extra_kwargs["model_type"] = model_type
    try:
        return mlflow.pytorch.log_model(
            model,
            name="model",
            registered_model_name=reg_name,
            signature=signature,
            input_example=input_example,
            pip_requirements=pip_reqs,
            **extra_kwargs,
        )
    except TypeError:
        # Fallback if older MLflow mock does not accept MLflow 3 extra_kwargs
        try:
            return mlflow.pytorch.log_model(
                model,
                name="model",
                registered_model_name=reg_name,
                signature=signature,
                input_example=input_example,
                pip_requirements=pip_reqs,
            )
        except Exception:
            pass
    except Exception as default_err:
        logger.debug("Default MLflow PyTorch model serialization failed; retrying with pickle: %s", default_err)
    try:
        return mlflow.pytorch.log_model(
            model,
            name="model",
            registered_model_name=reg_name,
            signature=signature,
            input_example=input_example,
            serialization_format="pickle",
            pip_requirements=pip_reqs,
            **extra_kwargs,
        )
    except Exception as err:
        logger.warning("Failed to log model to registry: %s", err)
        return None


class MLflowTracker:
    """MLOps experiment tracking and model registration orchestrator."""

    run_scope = staticmethod(mlflow_run_scope)
    log_metrics = staticmethod(_log_mlflow_metrics)
    log_visualizations = staticmethod(log_training_visualizations)

    @staticmethod
    def build_embedding_signature(input_dim: int, num_classes: int) -> ModelSignature:
        """Construct explicit MLflow ModelSignature for backbone embeddings -> taxonomic logits.

        Why:
            Populates explicit named inputs and outputs in MLflow Model Registry metadata, preventing unpopulated schema definitions (`Inputs (0) / Outputs (0)`).

        Args:
            input_dim: Dimensionality of feature embeddings.
            num_classes: Number of target taxonomic classes.

        Returns:
            ModelSignature with backbone_embeddings input and taxonomic_logits output TensorSpecs.
        """
        input_schema = Schema([TensorSpec(np.dtype(np.float32), (-1, input_dim), name="backbone_embeddings")])
        output_schema = Schema([TensorSpec(np.dtype(np.float32), (-1, num_classes), name="taxonomic_logits")])
        return ModelSignature(inputs=input_schema, outputs=output_schema)

    @staticmethod
    def build_onnx_signature(num_classes: int, image_size: int = 224) -> ModelSignature:
        """Construct explicit MLflow ModelSignature for raw image tensor -> (logits, embeddings).

        Why:
            Provides end-to-end tensor contracts for deployed ONNX Runtime graphs. Tensor names must match the exported dual-output graph (`input`, `logits`, `embeddings`) so MLflow serving-input validation feeds the expected input name to ONNX Runtime.

        Args:
            num_classes: Number of target taxonomic classes.
            image_size: Spatial resolution of input images in pixels.

        Returns:
            ModelSignature with `input` tensor and dual `logits` / `embeddings` output TensorSpecs.
        """
        input_schema = Schema([TensorSpec(np.dtype(np.float32), (-1, 3, image_size, image_size), name="input")])
        output_schema = Schema(
            [
                TensorSpec(np.dtype(np.float32), (-1, num_classes), name="logits"),
                TensorSpec(np.dtype(np.float32), (-1, -1), name="embeddings"),
            ]
        )
        return ModelSignature(inputs=input_schema, outputs=output_schema)

    @classmethod
    def log_model(
        cls,
        model: nn.Module,
        model_name: str = "taxon_vision_classifier",
        input_dim: int | None = None,
        num_classes: int | None = None,
        signature: ModelSignature | None = None,
        input_example: np.ndarray | None = None,
        registered_model_name: str | None = None,
        pip_requirements: list[str] | None = None,
        params: dict[str, Any] | None = None,
        model_type: str = "classifier",
    ) -> Any:
        """Log PyTorch classifier to MLflow registry with strongly-typed signature and input example.

        Args:
            model: PyTorch model module to serialize.
            model_name: Target artifact or registry name.
            input_dim: Optional dimensionality of embedding features.
            num_classes: Optional number of taxonomic output categories.
            signature: Optional pre-constructed ModelSignature.
            input_example: Optional real or zero-padded input tensor.
            registered_model_name: Optional explicit registry model name.
            pip_requirements: Optional list of package dependencies.
            params: Optional dictionary of hyperparameter metadata to attach to the LoggedModel.
            model_type: Model classification type string (default: "classifier").

        Returns:
            ModelInfo instance on success, or None on failure.
        """
        cpu_model = copy.deepcopy(model).cpu().eval()
        resolved_dim, resolved_classes = _resolve_model_dims(cpu_model, input_dim, num_classes)

        if signature is None:
            signature = cls.build_embedding_signature(resolved_dim, resolved_classes)

        if not isinstance(signature, ModelSignature):
            raise ValueError(
                f"Strict contract violation: signature must be an instance of ModelSignature, got {type(signature).__name__}"
            )

        if input_example is None:
            input_example = np.zeros((1, resolved_dim), dtype=np.float32)

        reg_name = registered_model_name or model_name
        pip_reqs = pip_requirements or ["torch", "numpy", "cloudpickle"]
        return _log_pytorch_model_with_fallback(
            cpu_model,
            reg_name,
            signature,
            input_example,
            pip_reqs,
            params=params,
            model_type=model_type,
        )

    @classmethod
    def log_onnx_model(
        cls,
        onnx_model: Any,
        artifact_path: str = "model",
        num_classes: int = 10,
        image_size: int = 224,
        signature: ModelSignature | None = None,
        input_example: np.ndarray | None = None,
        registered_model_name: str | None = None,
        **kwargs: Any,
    ) -> Any:
        """Log ONNX model to MLflow with explicit input/output tensor signature.

        Args:
            onnx_model: ONNX ModelProto or path to .onnx artifact.
            artifact_path: Artifact directory destination in MLflow run.
            num_classes: Number of target taxonomic classes.
            image_size: Input spatial image dimension in pixels.
            signature: Optional pre-constructed ModelSignature.
            input_example: Optional real or zero-padded input tensor.
            registered_model_name: Optional model registry name.
            **kwargs: Extra arguments forwarded to mlflow.onnx.log_model.

        Returns:
            ModelInfo instance on success, or None on failure.
        """
        if signature is None:
            signature = cls.build_onnx_signature(num_classes=num_classes, image_size=image_size)

        if not isinstance(signature, ModelSignature):
            raise ValueError(
                f"Strict contract violation: signature must be an instance of ModelSignature, got {type(signature).__name__}"
            )

        if input_example is None:
            input_example = np.zeros((1, 3, image_size, image_size), dtype=np.float32)

        model_proto = onnx_model
        if isinstance(onnx_model, (str, Path)):
            model_proto = onnx.load(str(onnx_model))

        try:
            return mlflow.onnx.log_model(
                onnx_model=model_proto,
                artifact_path=artifact_path,
                registered_model_name=registered_model_name,
                signature=signature,
                input_example=input_example,
                **kwargs,
            )
        except Exception as err:
            logger.warning("Failed to log ONNX model to MLflow: %s", err)
            return None


def log_stage_metrics_to_parent_run(
    metrics: dict[str, Any],
    train_metrics_path: Path | str = "reports/train_metrics.json",
    prefix: str = "",
) -> None:
    """Log downstream pipeline stage metrics into the parent training MLflow run.

    Args:
        metrics: Dictionary of metric values to record.
        train_metrics_path: Path to reports/train_metrics.json holding the parent run ID.
        prefix: Optional prefix to namespace metric keys (e.g. "calibrate." or "eval.").
    """
    path = Path(train_metrics_path)
    if not path.is_file():
        return
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        run_id = data.get("mlflow_run_id")
        if not run_id:
            return
        settings = get_settings()
        tracking_uri = _resolve_reachable_tracking_uri(settings)
        if not tracking_uri:
            return
        mlflow.set_tracking_uri(tracking_uri)
        client = MlflowClient(tracking_uri=tracking_uri)
        for key, val in metrics.items():
            if isinstance(val, (int, float)) and not isinstance(val, bool):
                metric_name = f"{prefix}{key}" if prefix else key
                client.log_metric(run_id=run_id, key=metric_name, value=float(val))
    except Exception as err:
        logger.debug("Failed to log stage metrics to MLflow parent run %s: %s", path, err)
