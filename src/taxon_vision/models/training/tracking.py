# SPDX-FileCopyrightText: 2026 Benjamin Förster
# SPDX-License-Identifier: MIT
"""MLOps tracking lifecycle and promotion utilities."""

from __future__ import annotations

import logging
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import mlflow
from torch import nn

from taxon_vision.config import get_settings

logger = logging.getLogger(__name__)


def _get_dvc_dataset_hash() -> str | None:
    """Parse dvc.lock to extract the dataset hash from the ingestion stage.

    Why:
        In auditable MLOps pipelines, every trained checkpoint must be cryptographically linked to the exact immutable dataset snapshot used during training. Logging the DVC manifest hash to MLflow provides end-to-end lineage tracking from raw observations to deployed models.

    How:
        Reads `dvc.lock`, parses the YAML stage graph, extracts the output MD5 checksum associated with `data/manifests`, and returns the hex digest string.

    Returns:
        The MD5 hash of the data/manifests directory, or None if unavailable.
    """
    try:
        from pathlib import Path

        import yaml

        dvc_lock_path = Path("dvc.lock")
        if not dvc_lock_path.exists():
            logger.debug("dvc.lock not found; skipping DVC hash logging.")
            return None

        with dvc_lock_path.open("r", encoding="utf-8") as f:
            dvc_data = yaml.safe_load(f)

        outs = dvc_data.get("stages", {}).get("ingestion", {}).get("outs", [])
        for out in outs:
            if out.get("path") == "data/manifests":
                return str(out.get("md5"))

        logger.debug("data/manifests output not found in dvc.lock")
    except Exception as e:
        logger.debug("Could not parse dvc.lock for dataset hashing: %s", e)
    return None


def _end_mlflow_run_safely(status: str) -> None:
    """End the MLflow run safely, logging any errors to debug.

    Args:
        status: The status to set the run to (e.g., "FAILED", "FINISHED").
    """
    try:
        mlflow.end_run(status=status)
    except Exception as err:
        logger.debug("Failed to set MLflow run status to %s: %s", status, err)


def _attach_mlflow_artifact_safely(log_file: Path | str | None) -> None:
    """Attach a log file as an MLflow artifact safely.

    Args:
        log_file: Path to the log file to attach.
    """
    if not log_file or not Path(log_file).exists():
        return
    try:
        mlflow.log_artifact(str(log_file))
    except Exception as err:
        logger.debug("Failed to attach log artifact to MLflow: %s", err)


@contextmanager
def mlflow_run_scope(
    run_name: str,
    params: dict[str, Any] | None = None,
    log_file: Path | str | None = None,
) -> Generator[mlflow.ActiveRun | None, None, None]:
    """Provide RAII (Resource Acquisition Is Initialization) lifecycle management for MLflow experiment tracking.

    RAII is a programming paradigm in which the lifetime of a resource is tied to the lifetime of a scope. In this case, the resource is an MLflow run, and the scope is the context manager.

    Why:
        Orphaned experiment runs left in 'RUNNING' status pollute MLOps registries when training processes abort, crash, or hit out-of-memory errors. Passing boolean flags (`is_active: bool`) across subroutines causes tramp data code smells and fragile error handling. A context manager guarantees deterministic lifecycle boundaries: setting up tracking, logging hyperparameter metadata, persisting run artifacts, and safely closing the run - marking failures as FAILED and successful completions as FINISHED - even during unhandled exceptions.

    How:
        1. Configures MLflow tracking URI and experiment name from project settings.
        2. Starts an active MLflow run with the given `run_name`.
        3. Logs initial training parameters and DVC dataset manifest hashes.
        4. Yields the active run object to the caller.
        5. If an unhandled exception occurs inside the block, marks the run status as FAILED via `mlflow.end_run(status="FAILED")` and re-raises.
        6. On normal exit, logs execution log file artifacts (if present) and marks the run as FINISHED via `mlflow.end_run(status="FINISHED")`.
        7. If MLflow tracking server is unreachable, catches connection errors gracefully, logs a debug notice, and yields None to allow offline training to proceed unimpeded.

    Args:
        run_name: Human-readable run name for identification in the MLflow UI.
        params: Optional dictionary of hyperparameter key-value pairs to log.
        log_file: Optional path to an execution log file to attach as an MLflow artifact.

    Yields:
        Active MLflow run instance if tracking succeeded, or None if unavailable.
    """
    active_run = None
    try:
        settings = get_settings()
        if settings.mlflow.tracking_uri:
            mlflow.set_tracking_uri(settings.mlflow.tracking_uri)
        if settings.mlflow.experiment_name:
            mlflow.set_experiment(settings.mlflow.experiment_name)

        active_run = mlflow.start_run(run_name=run_name)
        if params:
            mlflow.log_params(params)

        dataset_hash = _get_dvc_dataset_hash()
        if dataset_hash:
            mlflow.log_param("dvc_dataset_hash", dataset_hash)

        try:
            import subprocess

            git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
            git_branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], text=True).strip()
            mlflow.set_tag("git.commit_sha", git_sha)
            mlflow.set_tag("git.branch", git_branch)
        except Exception as git_err:
            logger.debug("Could not resolve git lineage tags: %s", git_err)
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
            mlflow.log_metrics(metrics, step=i + 1)
        mlflow.log_metric("duration_seconds", duration)
    except Exception as err:
        logger.debug("Failed to log MLflow metrics: %s", err)


def _promote_model_if_better(model: nn.Module, pr_auc: float) -> None:
    """Evaluate performance and conditionally promote the model via MLflow Model Registry.

    Why:
        Automated continuous deployment requires programmatic model promotion gates. Comparing Macro PR-AUC ensures newly trained models generalize effectively across both dominant and rare taxa before receiving traffic in production.

    How:
        Logs the validation PR-AUC promotion score and registers the PyTorch module in MLflow Model Registry. Compares the PR-AUC against the active 'Production' model alias. If superior, tags the model as 'Challenger' for Canary testing; otherwise tags it as 'Archived'. If no baseline exists, initializes the candidate as 'Production'.

    Args:
        model: The trained PyTorch classification head.
        pr_auc: The final validation Macro PR-AUC score.
    """
    from mlflow.tracking import MlflowClient

    mlflow.log_metric("val_promotion_score", pr_auc)

    model_name = "taxon_vision_classifier"
    try:
        model_info = mlflow.pytorch.log_model(model, "model", registered_model_name=model_name)
    except Exception as err:
        logger.warning("Failed to log model to registry: %s", err)
        return

    client = MlflowClient()
    try:
        prod_version = client.get_model_version_by_alias(model_name, "Production")
        if not prod_version.run_id:
            raise ValueError("Production model missing run_id")
        prod_run = client.get_run(prod_version.run_id)
        prod_score = prod_run.data.metrics.get("val_promotion_score", 0.0)

        if pr_auc > prod_score:
            logger.info("New model outperforms production (%.4f > %.4f). Tagging as Challenger.", pr_auc, prod_score)
            client.set_registered_model_alias(model_name, "Challenger", model_info.registered_model_version)
        else:
            logger.info("New model is worse than production (%.4f <= %.4f). Tagging as Archived.", pr_auc, prod_score)
            client.set_registered_model_alias(model_name, "Archived", model_info.registered_model_version)
    except Exception:
        logger.info("No production alias found. Tagging initial model as Production.")
        client.set_registered_model_alias(model_name, "Production", model_info.registered_model_version)
