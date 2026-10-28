# SPDX-FileCopyrightText: 2026 Benjamin Förster
# SPDX-License-Identifier: MIT
"""MLOps tracking lifecycle and promotion utilities."""

from __future__ import annotations

import copy
import json
import logging
import os
import subprocess
from collections.abc import Callable, Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import onnx
from mlflow.models import ModelSignature
from mlflow.types.schema import Schema, TensorSpec
from torch import nn

from taxon_vision.config import get_settings
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
from taxon_vision.models.training.types import PromotionAlias as PromotionAlias
from taxon_vision.models.training.visualizations import (
    _plot_confusion_matrix as _plot_confusion_matrix,
    _plot_energy_distribution as _vis_plot_energy_distribution,
    _plot_precision_recall_curves as _plot_precision_recall_curves,
    log_training_visualizations as log_training_visualizations,
)

logger = logging.getLogger(__name__)


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


@contextmanager
def mlflow_run_scope(
    run_name: str,
    params: dict[str, Any] | None = None,
    log_file: Path | str | None = None,
    nested: bool = False,
) -> Generator[mlflow.ActiveRun | None, None, None]:
    """Provide RAII (Resource Acquisition Is Initialization) lifecycle management for MLflow experiment tracking.

    RAII is a programming paradigm in which the lifetime of a resource is tied to the lifetime of a scope. In this case, the resource is an MLflow run, and the scope is the context manager.

    Why:
        Orphaned experiment runs left in 'RUNNING' status pollute MLOps registries when training processes abort, crash, or hit out-of-memory errors. Passing boolean flags (`is_active: bool`) across subroutines causes tramp data code smells and fragile error handling. A context manager guarantees deterministic lifecycle boundaries: setting up tracking, logging hyperparameter metadata, persisting run artifacts, and safely closing the run - marking failures as FAILED and successful completions as FINISHED - even during unhandled exceptions.

    How:
        1. Configures MLflow tracking URI and experiment name from project settings.
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

    Yields:
        Active MLflow run instance if tracking succeeded, or None if unavailable.
    """
    active_run = None
    try:
        settings = get_settings()
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI") or settings.mlflow.tracking_uri
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        if settings.mlflow.experiment_name:
            mlflow.set_experiment(settings.mlflow.experiment_name)

        active_run = mlflow.start_run(run_name=run_name, nested=True) if nested else mlflow.start_run(run_name=run_name)

        if params:
            mlflow.log_params(params)

        dataset_hash = get_dvc_dataset_hash()
        if dataset_hash:
            mlflow.log_param("dvc_dataset_hash", dataset_hash)

        _inject_lineage_tags()
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


def _log_mlflow_metrics(history: dict[str, list[float]], duration: float) -> None:
    """Log training metric trajectories and duration to the active MLflow run.

    Why:
        Tracking per-epoch convergence curves (training loss, validation loss, validation accuracy, F1, and PR-AUC) in MLflow enables visualization of overfitting, learning rate decay effectiveness, and historical comparisons across training runs.

    How:
        Iterates over recorded epochs in the history dictionary, emitting `mlflow.log_metrics` at each step index, followed by the total wall-clock training duration.

    Args:
        history: Metric history dictionary containing epoch evaluation metrics.
        duration: Total elapsed training duration in seconds.
    """
    try:
        n_epochs = len(history.get("train_loss", []))
        for i in range(n_epochs):
            metrics = {
                "train_loss": history.get("train_loss", [0.0])[i],
                "val_loss": history.get("val_loss", [0.0])[i],
                "val_accuracy": history.get("val_accuracy", [0.0])[i],
                "val_f1": history.get("val_f1", [0.0])[i],
                "val_pr_auc": history.get("val_pr_auc", [0.0])[i],
            }
            if "learning_rate" in history and i < len(history["learning_rate"]):
                metrics["learning_rate"] = history["learning_rate"][i]
            mlflow.log_metrics(metrics, step=i + 1)
        mlflow.log_metric("duration_seconds", duration)
    except Exception as err:
        logger.debug("Failed to log MLflow metrics: %s", err)


def _log_dataset_lineage(train_emb: np.ndarray, train_lbl: np.ndarray) -> None:
    """Log dataset input lineage metadata into the active MLflow run.

    Args:
        train_emb: Training embeddings feature array.
        train_lbl: Training integer targets array.
    """
    if not mlflow.active_run():
        return

    def _log() -> None:
        dataset = mlflow.data.from_numpy(  # type: ignore[attr-defined]
            features=train_emb,
            targets=train_lbl,
            name="train_embeddings",
        )
        mlflow.log_input(dataset, context="training")

    _safe_mlflow_call("log_dataset_lineage", _log)


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
) -> Any:
    """Attempt default logging with fallback to pickle serialization.

    Args:
        model: PyTorch model module on CPU.
        reg_name: Target registered model name.
        signature: ModelSignature instance.
        input_example: Input array example.
        pip_reqs: Required pip dependencies.

    Returns:
        ModelInfo instance on success, or None on failure.
    """
    if not isinstance(signature, ModelSignature):
        raise ValueError(
            f"Strict contract violation: signature must be an instance of ModelSignature, got {type(signature).__name__}"
        )

    try:
        return mlflow.pytorch.log_model(
            model,
            name="model",
            registered_model_name=reg_name,
            signature=signature,
            input_example=input_example,
            pip_requirements=pip_reqs,
        )
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
        """Construct explicit MLflow ModelSignature for raw image tensor -> taxonomic logits.

        Why:
            Provides end-to-end tensor contracts for deployed ONNX Runtime graphs.

        Args:
            num_classes: Number of target taxonomic classes.
            image_size: Spatial resolution of input images in pixels.

        Returns:
            ModelSignature with input_images tensor and taxonomic_logits output TensorSpecs.
        """
        input_schema = Schema([TensorSpec(np.dtype(np.float32), (-1, 3, image_size, image_size), name="input_images")])
        output_schema = Schema([TensorSpec(np.dtype(np.float32), (-1, num_classes), name="taxonomic_logits")])
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
        return _log_pytorch_model_with_fallback(cpu_model, reg_name, signature, input_example, pip_reqs)

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
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI") or settings.mlflow.tracking_uri
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)
        for key, val in metrics.items():
            if isinstance(val, (int, float)) and not isinstance(val, bool):
                metric_name = f"{prefix}{key}" if prefix else key
                client.log_metric(run_id=run_id, key=metric_name, value=float(val))
    except Exception as err:
        logger.debug("Failed to log stage metrics to MLflow parent run %s: %s", path, err)
