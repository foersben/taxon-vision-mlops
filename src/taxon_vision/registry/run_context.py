# SPDX-FileCopyrightText: 2026 Benjamin Förster
# SPDX-License-Identifier: MIT
"""RAII experiment run lifecycle managers, tracking URI resolution, and MLflow model loggers.

Summary:
    Provides deterministic RAII context managers (`mlflow_run_scope`, `resume_parent_mlflow_run`) for experiment tracking lifecycle across multi-process DVC stages alongside strongly typed MLflow 3 model logging via `MLflowTracker`.

Why (Architectural Rationale & Trade-offs):
    Separating MLflow SDK lifecycle management and direct MLflow 3 model logging from training optimisation decouples registry persistence cleanly, retires `SH-11` (`_log_pytorch_model_with_fallback`), and satisfies Rule 08 parameter arity bounds via typed request dataclasses.

How (Algorithmic Mechanism & Invariants):
    Resolves DagsHub Basic Auth or `MLFLOW_TRACKING_URI` overrides without creating local SQLite/mlruns clutter when unauthenticated; transitions runs cleanly to `FINISHED` or `FAILED` on scope exit; enforces strict `ModelSignature` contracts prior to registration.
"""

from __future__ import annotations

import copy
import json
import logging
import os
import subprocess
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import mlflow
import mlflow.onnx
import mlflow.pytorch
import numpy as np
import onnx
from mlflow.models import ModelSignature

from taxon_vision.config.loader import get_settings
from taxon_vision.config.schemas import Settings
from taxon_vision.models.training.dataset import get_dvc_dataset_hash
from taxon_vision.registry.types import (
    ONNXModelLogSpec,
    ParentRunContext,
    PyTorchModelLogSpec,
    build_embedding_signature,
    build_onnx_signature,
    resolve_model_dims,
    safe_mlflow_call,
)

logger = logging.getLogger(__name__)


def end_mlflow_run_safely(status: str) -> None:
    """Terminate the active MLflow run safely, logging warnings if the tracking endpoint fails.

    Args:
        status: Terminal MLflow run state (`"FINISHED"` or `"FAILED"`).
    """
    safe_mlflow_call(f"end_run({status})", lambda: mlflow.end_run(status=status))


def attach_mlflow_artifact_safely(log_file: Path | str | None) -> None:
    """Upload a local log artefact into the active MLflow run if present on disk.

    Args:
        log_file: Optional path to a training or evaluation log file.
    """
    if not log_file or not Path(log_file).exists():
        return
    safe_mlflow_call("log_artifact", lambda: mlflow.log_artifact(str(log_file)))


def inject_lineage_tags() -> None:
    """Attach Git revision SHA, branch name, and CI orchestration tags to the active MLflow run."""
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


def ensure_dagshub_auth(uri: str) -> bool:
    """Populate MLflow Basic Auth environment credentials for DagsHub endpoints when a token exists.

    Args:
        uri: Candidate MLflow tracking URI.

    Returns:
        True if a non-empty DagsHub credential token is available, False otherwise.
    """
    dagshub_token = os.getenv("DAGSHUB_TOKEN") or os.getenv("MLFLOW_TRACKING_PASSWORD")
    if not dagshub_token:
        return False
    if "dagshub.com" in uri:
        os.environ.setdefault("MLFLOW_TRACKING_USERNAME", "foersben")
        os.environ.setdefault("MLFLOW_TRACKING_PASSWORD", dagshub_token)
    return True


def resolve_reachable_tracking_uri(settings: Settings | None = None) -> str:
    """Resolve the active MLflow tracking URI targeting DagsHub or an explicit environment override.

    Why:
        Centralises MLflow state on DagsHub (`https://dagshub.com/foersben/taxon-vision-mlops.mlflow`) to prevent fragmented SQLite / `mlruns` artefacts in offline test runs or unauthenticated dry-runs.

    How:
        1. Inspects `MLFLOW_TRACKING_URI`: empty disables tracking; DagsHub endpoints verify `ensure_dagshub_auth`.
        2. Otherwise inspects `cfg.mlflow.tracking_uri` using the identical authentication check.

    Args:
        settings: Optional pre-loaded `Settings` instance.

    Returns:
        Configured MLflow tracking URI string, or `""` when disabled or unauthenticated.
    """
    env_uri = os.getenv("MLFLOW_TRACKING_URI")
    if env_uri is not None:
        cleaned_env = env_uri.strip()
        if not cleaned_env:
            return ""
        if "dagshub.com" in cleaned_env and not ensure_dagshub_auth(cleaned_env):
            return ""
        return cleaned_env

    cfg = settings or get_settings()
    configured_uri = (cfg.mlflow.tracking_uri or "").strip()
    if not configured_uri:
        return ""
    if "dagshub.com" in configured_uri and not ensure_dagshub_auth(configured_uri):
        logger.debug("Skipping DagsHub MLflow tracking (%s): credentials unavailable.", configured_uri)
        return ""
    return configured_uri


def _open_mlflow_run(
    run_name: str,
    params: dict[str, Any] | None,
    nested: bool,
    experiment_name: str | None,
) -> Any:
    """Resolve tracking URI, configure experiment, launch active run, and attach lineage metadata."""
    settings = get_settings()
    tracking_uri = resolve_reachable_tracking_uri(settings)
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
    inject_lineage_tags()
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
    """Provide deterministic RAII lifecycle management for MLflow experiment tracking runs.

    Why:
        Orphaned runs left in `'RUNNING'` state pollute registries after OOM or runtime crashes; context-managed RAII guarantees exact `FINISHED`/`FAILED` status transitions on scope exit.

    How:
        Starts an active run via `_open_mlflow_run`, yields `active_run` (or `None` when unreachable), marks `FAILED` and re-raises on unhandled exceptions, or attaches `log_file` and marks `FINISHED` on clean completion.

    Args:
        run_name: Human-readable run display name.
        params: Optional initial hyperparameter dictionary.
        log_file: Optional execution log file path uploaded on clean completion.
        nested: Whether to create a child run nested under an existing active run.
        experiment_name: Optional explicit experiment name override.

    Yields:
        Active MLflow run handle if tracking is reachable, otherwise `None`.
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
            end_mlflow_run_safely("FAILED")
        raise
    else:
        if active_run:
            attach_mlflow_artifact_safely(log_file)
            end_mlflow_run_safely("FINISHED")


@contextmanager
def resume_parent_mlflow_run(
    train_metrics_path: Path | str = "reports/train_metrics.json",
) -> Generator[ParentRunContext | None, None, None]:
    """Provide RAII context manager re-entering the Stage 4 parent MLflow run across DVC subprocesses.

    Why:
        Downstream DVC pipeline stages (`export`, `calibrate`, `evaluate`) execute in separate OS processes and must bind ONNX artefacts, conformal parameters, and holdout metrics onto the originating training run and `LoggedModel` card.

    How:
        Parses `mlflow_run_id` and `mlflow_model_id` from `train_metrics_path`, verifies reachable authentication via `resolve_reachable_tracking_uri`, reopens the parent run via `mlflow.start_run(run_id=...)`, yields `ParentRunContext`, and closes safely on exit.

    Args:
        train_metrics_path: Filesystem path to Stage 4 `reports/train_metrics.json`.

    Yields:
        `ParentRunContext` if resumed cleanly, otherwise `None`.
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
    tracking_uri = resolve_reachable_tracking_uri(settings)
    if not tracking_uri:
        logger.debug("MLflow tracking URI unavailable or unauthenticated; skipping run resumption.")
        yield None
        return
    mlflow.set_tracking_uri(tracking_uri)

    active_run = None
    try:
        active_run = mlflow.start_run(run_id=ctx.run_id)
        if ctx.model_id:
            safe_mlflow_call("set_active_model", lambda: mlflow.set_active_model(model_id=ctx.model_id))
        logger.info("Resumed parent MLflow run %s for stage continuity", ctx.run_id)
    except Exception as start_err:
        logger.warning("Failed to resume parent MLflow run %s: %s", ctx.run_id, start_err)
        yield None
        return

    try:
        yield ctx
    finally:
        if active_run is not None:
            end_mlflow_run_safely("FINISHED")


class MLflowTracker:
    """MLOps experiment tracking signature builders and typed MLflow 3 model loggers."""

    build_embedding_signature = staticmethod(build_embedding_signature)
    build_onnx_signature = staticmethod(build_onnx_signature)

    @classmethod
    def log_model(cls, spec: PyTorchModelLogSpec) -> Any:
        """Register a PyTorch classification head directly to MLflow 3 with explicit schema validation.

        Args:
            spec: Strongly typed `PyTorchModelLogSpec` bundling model weights, names, schema, and parameters.

        Returns:
            MLflow `ModelInfo` instance on success, or `None` when serialisation fails.
        """
        cpu_model = copy.deepcopy(spec.model).cpu().eval()
        resolved_dim, resolved_classes = resolve_model_dims(cpu_model, spec.input_dim, spec.num_classes)
        signature = spec.signature or build_embedding_signature(resolved_dim, resolved_classes)

        if not isinstance(signature, ModelSignature):
            raise ValueError(
                f"Strict contract violation: signature must be an instance of ModelSignature, got {type(signature).__name__}"
            )

        input_example = (
            spec.input_example if spec.input_example is not None else np.zeros((1, resolved_dim), dtype=np.float32)
        )
        reg_name = spec.registered_model_name or spec.model_name
        pip_reqs = spec.pip_requirements or ["torch", "numpy", "cloudpickle"]

        extra_kwargs: dict[str, Any] = {}
        if spec.params:
            extra_kwargs["params"] = spec.params
        if spec.model_type:
            extra_kwargs["model_type"] = spec.model_type

        try:
            return mlflow.pytorch.log_model(
                cpu_model,
                name="model",
                registered_model_name=reg_name,
                signature=signature,
                input_example=input_example,
                pip_requirements=pip_reqs,
                **extra_kwargs,
            )
        except Exception as err:
            logger.warning("Failed to log PyTorch model to MLflow registry: %s", err)
            return None

    @classmethod
    def log_onnx_model(cls, spec: ONNXModelLogSpec) -> Any:
        """Register a fused ONNX model graph with explicit NCHW tensor signature into MLflow.

        Args:
            spec: Strongly typed `ONNXModelLogSpec` bundling graph/path, dimensions, and signature metadata.

        Returns:
            MLflow `ModelInfo` descriptor on success, or `None` when logging fails.
        """
        settings = get_settings()
        resolved_size = spec.image_size if spec.image_size is not None else settings.model.image_size
        if spec.num_classes is None:
            raise ValueError("ONNXModelLogSpec.num_classes must be supplied when logging ONNX models.")
        resolved_classes = spec.num_classes
        signature = spec.signature or build_onnx_signature(num_classes=resolved_classes, image_size=resolved_size)

        if not isinstance(signature, ModelSignature):
            raise ValueError(
                f"Strict contract violation: signature must be an instance of ModelSignature, got {type(signature).__name__}"
            )

        input_example = (
            spec.input_example
            if spec.input_example is not None
            else np.zeros((1, 3, resolved_size, resolved_size), dtype=np.float32)
        )

        model_proto = onnx.load(str(spec.onnx_model)) if isinstance(spec.onnx_model, (str, Path)) else spec.onnx_model

        try:
            return mlflow.onnx.log_model(
                onnx_model=model_proto,
                artifact_path=spec.artifact_path,
                registered_model_name=spec.registered_model_name,
                signature=signature,
                input_example=input_example,
                **spec.extra_kwargs,
            )
        except Exception as err:
            logger.warning("Failed to log ONNX model to MLflow: %s", err)
            return None
