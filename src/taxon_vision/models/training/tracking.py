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
import numpy as np
import torch
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


def _plot_confusion_matrix(
    val_labels: np.ndarray,
    pred_labels: np.ndarray,
    class_names: list[str],
) -> Any:
    """Render class-normalized confusion matrix figure.

    Args:
        val_labels: 1D array of ground truth label indices.
        pred_labels: 1D array of predicted class indices.
        class_names: List of human-readable class names.

    Returns:
        Matplotlib figure instance.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(val_labels, pred_labels)
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-12)

    fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
    cax = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues")
    fig.colorbar(cax)

    n_classes = len(class_names)
    ax.set_xticks(range(n_classes))
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(n_classes))
    ax.set_yticklabels(class_names, fontsize=8)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title("Validation Confusion Matrix (Normalized)")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            color = "white" if cm_norm[i, j] > 0.5 else "black"
            ax.text(j, i, str(val), ha="center", va="center", color=color, fontsize=8)

    fig.tight_layout()
    return fig


def _plot_precision_recall_curves(
    val_labels: np.ndarray,
    val_probs: np.ndarray,
    class_names: list[str],
) -> Any:
    """Render multiclass Precision-Recall curves figure.

    Args:
        val_labels: 1D array of ground truth label indices.
        val_probs: 2D array of class probabilities of shape (N, num_classes).
        class_names: List of human-readable class names.

    Returns:
        Matplotlib figure instance.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import average_precision_score, precision_recall_curve

    fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
    n_classes = val_probs.shape[1]

    for k in range(n_classes):
        y_true = (val_labels == k).astype(int)
        if np.sum(y_true) == 0:
            continue
        precision, recall, _ = precision_recall_curve(y_true, val_probs[:, k])
        ap = average_precision_score(y_true, val_probs[:, k])
        label = f"{class_names[k]} (AP={ap:.2f})" if k < len(class_names) else f"Class {k} (AP={ap:.2f})"
        ax.plot(recall, precision, lw=1.5, label=label)

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_ylim(0.0, 1.05)
    ax.set_xlim(0.0, 1.0)
    ax.set_title("Per-Class Precision-Recall Curves")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="lower left", fontsize=7)
    fig.tight_layout()
    return fig


def _plot_energy_distribution(
    val_logits: np.ndarray,
    calibrated_threshold: float | None = None,
) -> Any:
    """Render Helmholtz free energy distribution and epistemic decision boundary.

    Args:
        val_logits: 2D array of validation logits with shape (N, num_classes).
        calibrated_threshold: Optional calibrated scalar energy threshold.

    Returns:
        Matplotlib figure instance.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from taxon_vision.uncertainty.ood_detector import EnergyOODDetector

    detector = EnergyOODDetector()
    energies = np.array([detector.compute_energy(logit) for logit in val_logits])

    fig, ax = plt.subplots(figsize=(8, 5), dpi=100)
    ax.hist(energies, bins=25, alpha=0.75, color="#1f77b4", edgecolor="black", label="In-Distribution (Val)")

    if calibrated_threshold is not None:
        ax.axvline(
            calibrated_threshold,
            color="#d62728",
            linestyle="--",
            linewidth=2,
            label=f"Calibrated OOD Threshold ({calibrated_threshold:.2f})",
        )

    ax.set_xlabel("Helmholtz Free Energy E(x)")
    ax.set_ylabel("Observation Count")
    ax.set_title("Validation Free Energy Distribution & Epistemic Boundary")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    return fig


def log_training_visualizations(
    val_logits: np.ndarray,
    val_labels: np.ndarray,
    taxa_catalog: list[dict[str, Any]] | None = None,
    calibrated_energy_threshold: float | None = None,
) -> None:
    """Render and log publication-grade performance evaluation figures to MLflow.

    Args:
        val_logits: 2D NumPy array of validation logits with shape (N, num_classes).
        val_labels: 1D NumPy array of ground truth labels with shape (N,).
        taxa_catalog: Optional catalog containing taxa metadata dictionaries.
        calibrated_energy_threshold: Calibrated energy threshold boundary to annotate.
    """
    if not mlflow.active_run():
        return

    import matplotlib.pyplot as plt

    try:
        class_names = [
            str(t.get("common_name", t.get("scientific_name", f"Class {i}"))) for i, t in enumerate(taxa_catalog or [])
        ]
        num_classes = val_logits.shape[1]
        if len(class_names) < num_classes:
            class_names.extend([f"Class {i}" for i in range(len(class_names), num_classes)])

        # 1. Confusion Matrix
        pred_labels = np.argmax(val_logits, axis=1)
        fig_cm = _plot_confusion_matrix(val_labels, pred_labels, class_names)
        mlflow.log_figure(fig_cm, "visualizations/confusion_matrix.png")
        plt.close(fig_cm)

        # 2. Precision-Recall Curves
        shifted = val_logits - np.max(val_logits, axis=1, keepdims=True)
        exp_logits = np.exp(shifted)
        probs = exp_logits / (np.sum(exp_logits, axis=1, keepdims=True) + 1e-12)
        fig_pr = _plot_precision_recall_curves(val_labels, probs, class_names)
        mlflow.log_figure(fig_pr, "visualizations/precision_recall_curves.png")
        plt.close(fig_pr)

        # 3. Energy Distribution
        fig_energy = _plot_energy_distribution(val_logits, calibrated_energy_threshold)
        mlflow.log_figure(fig_energy, "visualizations/energy_distribution.png")
        plt.close(fig_energy)
    except Exception as err:
        logger.warning("Failed to render and log training visualizations to MLflow: %s", err)


def _prepare_model_signature_and_example(
    model: nn.Module,
    val_embeddings: torch.Tensor | None,
) -> tuple[Any, Any]:
    """Infer model signature and input example from real validation data or module geometry.

    Args:
        model: Trained PyTorch classification head.
        val_embeddings: Optional real validation embeddings.

    Returns:
        Tuple of (signature, input_example).
    """
    if val_embeddings is not None and val_embeddings.shape[0] > 0:
        sample_size = min(5, val_embeddings.shape[0])
        sample_input = val_embeddings[:sample_size]
        model.eval()
        with torch.no_grad():
            sample_output = model(sample_input)
        input_example = sample_input.cpu().numpy()
        output_example = sample_output.cpu().numpy()
        try:
            from mlflow.models.signature import infer_signature

            return infer_signature(input_example, output_example), input_example
        except Exception as sig_err:
            logger.debug("Could not infer model signature from validation batch: %s", sig_err)
            return None, input_example

    input_dim = 1280
    for module in model.modules():
        if isinstance(module, nn.Linear):
            input_dim = module.in_features
            break
    return None, torch.zeros(1, input_dim).numpy()


def _log_model_artifact(
    model: nn.Module,
    model_name: str,
    signature: Any,
    input_example: Any,
) -> Any:
    """Log PyTorch model to MLflow registry with signature and serialization fallback.

    Args:
        model: PyTorch model module.
        model_name: Target registered model name.
        signature: Inferred model signature.
        input_example: Real or zero-padded input example array.

    Returns:
        ModelInfo instance on success, or None on failure.
    """
    try:
        return mlflow.pytorch.log_model(
            model,
            name="model",
            registered_model_name=model_name,
            signature=signature,
            input_example=input_example,
        )
    except Exception:
        try:
            return mlflow.pytorch.log_model(
                model,
                name="model",
                registered_model_name=model_name,
                signature=signature,
                input_example=input_example,
                serialization_format="pickle",
            )
        except Exception as err:
            logger.warning("Failed to log model to registry: %s", err)
            return None


def _apply_model_version_metadata(
    client: Any,
    model_name: str,
    version: str,
    extractor: str | None,
    pr_auc: float,
) -> None:
    """Attach human-readable description and audit tags to registered model version.

    Args:
        client: MlflowClient instance.
        model_name: Registered model name.
        version: Model version identifier string.
        extractor: Vision foundation backbone name.
        pr_auc: Validation Macro PR-AUC score.
    """
    try:
        desc = f"TaxonVision classifier head ({extractor or 'backbone'}). Validation Macro PR-AUC: {pr_auc:.4f}."
        client.update_model_version(
            name=model_name,
            version=version,
            description=desc,
        )
        if extractor:
            client.set_model_version_tag(model_name, version, "extractor", extractor)
        client.set_model_version_tag(model_name, version, "val_pr_auc", f"{pr_auc:.4f}")
    except Exception as meta_err:
        logger.debug("Failed to set model version metadata: %s", meta_err)


def _determine_promotion_alias(
    client: Any,
    model_name: str,
    version: str,
    pr_auc: float,
) -> str:
    """Compare performance against current production alias and tag candidate accordingly.

    Args:
        client: MlflowClient instance.
        model_name: Registered model name.
        version: Model version identifier string.
        pr_auc: Validation Macro PR-AUC score.

    Returns:
        Applied alias string ('Challenger', 'Archived', or 'Production').
    """
    try:
        prod_version = client.get_model_version_by_alias(model_name, "Production")
        if not prod_version.run_id:
            raise ValueError("Production model missing run_id")
        prod_run = client.get_run(prod_version.run_id)
        prod_score = prod_run.data.metrics.get("val_promotion_score", 0.0)

        if pr_auc > prod_score:
            logger.info("New model outperforms production (%.4f > %.4f). Tagging as Challenger.", pr_auc, prod_score)
            client.set_registered_model_alias(model_name, "Challenger", version)
            return "Challenger"
        logger.info("New model is worse than production (%.4f <= %.4f). Tagging as Archived.", pr_auc, prod_score)
        client.set_registered_model_alias(model_name, "Archived", version)
        return "Archived"
    except Exception:
        logger.info("No production alias found. Tagging initial model as Production.")
        client.set_registered_model_alias(model_name, "Production", version)
        return "Production"


def _promote_model_if_better(
    model: nn.Module,
    pr_auc: float,
    val_embeddings: torch.Tensor | None = None,
    extractor: str | None = None,
) -> str | None:
    """Evaluate performance and conditionally promote the model via MLflow Model Registry.

    Why:
        Automated continuous deployment requires programmatic model promotion gates. Comparing Macro PR-AUC ensures newly trained models generalize effectively across both dominant and rare taxa before receiving traffic in production.

    How:
        Logs the validation PR-AUC promotion score and registers the PyTorch module in MLflow Model Registry. Compares the PR-AUC against the active 'Production' model alias. If superior, tags the model as 'Challenger' for Canary testing; otherwise tags it as 'Archived'. If no baseline exists, initializes the candidate as 'Production'.

    Args:
        model: The trained PyTorch classification head.
        pr_auc: The final validation Macro PR-AUC score.
        val_embeddings: Optional real validation embeddings used for signature inference.
        extractor: Optional backbone feature extractor identifier.

    Returns:
        The applied promotion alias string ('Challenger', 'Archived', 'Production'), or None.
    """
    from mlflow.tracking import MlflowClient

    mlflow.log_metric("val_promotion_score", pr_auc)
    model_name = "taxon_vision_classifier"

    signature, input_example = _prepare_model_signature_and_example(model, val_embeddings)
    model_info = _log_model_artifact(model, model_name, signature, input_example)
    if model_info is None:
        return None

    client = MlflowClient()
    version = str(model_info.registered_model_version)
    _apply_model_version_metadata(client, model_name, version, extractor, pr_auc)
    return _determine_promotion_alias(client, model_name, version, pr_auc)
