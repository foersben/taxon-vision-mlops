# SPDX-FileCopyrightText: 2026 Benjamin Förster
# SPDX-License-Identifier: MIT
"""Publication-grade performance diagnostic visualizations and MLflow figure logging."""

from __future__ import annotations

import logging
import warnings
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import optuna
import optuna.visualization.matplotlib as optuna_plt
from sklearn.metrics import average_precision_score, confusion_matrix, precision_recall_curve

from taxon_vision.uncertainty.ood_detector import EnergyOODDetector

logger = logging.getLogger(__name__)


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
    n_classes = len(class_names)
    cm = confusion_matrix(val_labels, pred_labels, labels=list(range(n_classes)))
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-12)

    fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
    cax = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues")
    fig.colorbar(cax)

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

        per_class_aps = [
            float(average_precision_score((val_labels == k).astype(int), probs[:, k]))
            for k in range(num_classes)
            if np.sum(val_labels == k) > 0
        ]
        if per_class_aps:
            mlflow.log_metric("worst_class_ap", float(np.min(per_class_aps)))
            mlflow.log_metric("median_class_ap", float(np.median(per_class_aps)))

        # 3. Energy Distribution
        fig_energy = _plot_energy_distribution(val_logits, calibrated_energy_threshold)
        mlflow.log_figure(fig_energy, "visualizations/energy_distribution.png")
        plt.close(fig_energy)
    except Exception as err:
        logger.warning("Failed to render and log training visualizations to MLflow: %s", err)


def log_optuna_diagnostic_plots(study: Any) -> None:
    """Render and log Optuna optimization history and parameter importance plots to MLflow.

    Args:
        study: Optuna study instance containing completed evaluation trials.
    """
    if study is None or len(study.trials) == 0:
        return

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            ax_hist = optuna_plt.plot_optimization_history(study)
            mlflow.log_figure(ax_hist.figure, "visualizations/optuna_optimization_history.png")
            plt.close()
        except Exception as err:
            logger.debug("Failed to log Optuna optimization history plot: %s", err)

        try:
            completed_values = [
                t.value for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None
            ]
            if len(set(completed_values)) > 1:
                ax_imp = optuna_plt.plot_param_importances(study)
                mlflow.log_figure(ax_imp.figure, "visualizations/optuna_param_importances.png")
                plt.close()
        except Exception as err:
            logger.debug("Failed to log Optuna parameter importance plot: %s", err)
