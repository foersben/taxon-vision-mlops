# SPDX-FileCopyrightText: 2026 Benjamin Förster
# SPDX-License-Identifier: MIT
"""Publication-grade performance diagnostic visualizations and MLflow figure logging."""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from typing import Any, Protocol

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


class ParetoProfileLike(Protocol):
    """Structural protocol defining candidate backbone Pareto attributes.

    Why:
        Enforces static typing discipline while decoupling visualisation plotting from specific benchmark model classes or dataclasses. This eliminates dynamic runtime reflection (getattr/hasattr) and tuple blindness, ensuring safe compile-time type verification across heterogeneous benchmark profile inputs.

    How:
        Defines read-only structural attributes matching backbone Pareto evaluation specifications: model name, parameter count, p95 ONNX latency, and throughput. Any candidate profile satisfying these attributes conforms via structural subtyping (PEP 544).

    Attributes:
        extractor: Canonical vision backbone extractor identifier.
        parameters_m: Parameter scale expressed in millions.
        p95_latency_ms: p95 ONNX runtime inference latency in milliseconds.
        throughput_fps: Steady-state evaluation throughput in frames per second.
    """

    extractor: str
    parameters_m: float
    p95_latency_ms: float
    throughput_fps: float


@dataclass(frozen=True)
class HoldoutVisualizationPayload:
    """Encapsulates test evaluation predictions, probabilities, and thresholds for diagnostic plotting.

    Why:
        Centralises all necessary evaluation artefacts from Stage 7 evaluate.py so that downstream plotting routines remain pure, deterministic, and decoupled from raw file I/O or model execution. Eliminates tuple blindness and parameter sprawl by replacing loose arguments with an immutable typed container.

    How:
        Binds holdout predictions, logits, softmax probabilities, conformal prediction set sizes, non-conformity threshold q_hat, coverage metrics, and calibrated OOD energy thresholds into a single frozen dataclass consumed by log_holdout_visualizations.

    Attributes:
        test_labels: Ground-truth class labels for test samples.
        test_probs: Soft probabilities (softmax outputs) for each class on test samples.
        test_logits: Raw logits prior to softmax for test samples.
        set_sizes: Size of the computed conformal prediction set for each test sample.
        q_hat: Conformal p-value threshold used for set construction.
        nominal_coverage: Target coverage level (1 - alpha) from calibration config.
        empirical_coverage: Observed coverage on the test set.
        calibrated_energy_threshold: Optional calibrated energy threshold for OOD detection.
        k_max: Maximum number of predicted taxa to include in ranked lists and plots.
        taxa_catalog: Optional taxa metadata mapping class indices to human-readable names.
        pareto_profiles: Optional Pareto efficiency profiles from pareto_front.py for trade-off visualisations.
        latency_slo_ms: Target latency threshold (in milliseconds) for the speed-accuracy trade-off plots.
    """

    test_labels: np.ndarray
    test_probs: np.ndarray
    test_logits: np.ndarray
    set_sizes: np.ndarray
    q_hat: float
    nominal_coverage: float
    empirical_coverage: float
    calibrated_energy_threshold: float | None
    k_max: int = 5
    taxa_catalog: list[dict[str, Any]] | None = None
    pareto_profiles: list[ParetoProfileLike] | None = None
    latency_slo_ms: float = 25.0


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


def _plot_conformal_set_sizes(
    set_sizes: np.ndarray,
    k_max: int,
    empirical_coverage: float,
    nominal_coverage: float,
) -> matplotlib.figure.Figure:
    """Render prediction set cardinality histogram and triage referral threshold.

    Why:
        Under Split Conformal Prediction (Angelopoulos & Bates, 2021), the prediction set size |C(x)| indicates classification ambiguity. Empty sets (|C(x)| = 0) and excessively large sets (|C(x)| > k_max) trigger expert human triage. Visualising the cardinality distribution verifies that the model produces compact, actionable prediction sets while maintaining marginal coverage.

    How:
        Computes discrete frequencies for each set cardinality, renders a bar chart, draws the human referral boundary at k_max, and annotates empirical versus nominal coverage.

    Complexity:
        O(N) histogram accumulation time where N is the number of holdout samples.

    Args:
        set_sizes: 1D array of prediction set sizes |C(x)|.
        k_max: Maximum set cardinality threshold before human review.
        empirical_coverage: Observed holdout test coverage fraction.
        nominal_coverage: Target mathematical coverage level (1 - alpha).

    Returns:
        Matplotlib figure instance.
    """
    fig, ax = plt.subplots(figsize=(8, 5), dpi=100)
    max_size = int(np.max(set_sizes)) if len(set_sizes) > 0 else k_max
    bins = np.arange(0, max(max_size + 2, k_max + 3)) - 0.5

    counts, edges = np.histogram(set_sizes, bins=bins)
    centres = 0.5 * (edges[:-1] + edges[1:])

    ax.bar(centres, counts, width=0.8, color="#2ca02c", edgecolor="black", alpha=0.75, label="Holdout Observations")
    ax.axvline(
        k_max + 0.5,
        color="#d62728",
        linestyle="--",
        linewidth=2,
        label=f"Referral Boundary (k > {k_max})",
    )

    ax.set_xlabel("Conformal Prediction Set Cardinality |C(X)|")
    ax.set_ylabel("Observation Count")
    ax.set_title(f"Holdout Conformal Set Sizes (Empirical: {empirical_coverage:.2%}, Nominal: {nominal_coverage:.2%})")
    max_tick = int(max(centres)) if len(centres) > 0 else k_max
    ax.set_xticks(range(0, max_tick + 1))
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    return fig


def _plot_reliability_diagram(
    test_probs: np.ndarray,
    test_labels: np.ndarray,
    n_bins: int = 10,
) -> matplotlib.figure.Figure:
    """Render 10-bin confidence reliability diagram and compute Expected Calibration Error.

    Why:
        Deep vision models on long-tailed biological distributions exhibit overconfidence on tail classes. A reliability diagram checks whether predicted confidence matches true empirical accuracy across uniform confidence intervals.

    How:
        Partitions top-1 predicted probabilities into n_bins uniform bins, calculates mean confidence and true accuracy per bin, computes Expected Calibration Error (ECE), and plots the reliability curve against the ideal diagonal reference.

    Complexity:
        O(N) binning and aggregation complexity where N is the number of test samples.

    Args:
        test_probs: 2D array of predicted class probabilities (N, num_classes).
        test_labels: 1D array of ground truth integer labels (N,).
        n_bins: Number of equal-width probability bins between 0.0 and 1.0.

    Returns:
        Matplotlib figure instance.
    """
    confidences = np.max(test_probs, axis=1)
    predictions = np.argmax(test_probs, axis=1)
    accuracies = (predictions == test_labels).astype(float)

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    bin_accs: list[float] = []
    bin_confs: list[float] = []
    ece = 0.0

    for lower, upper in zip(bin_lowers, bin_uppers, strict=False):
        in_bin = (confidences > lower) & (confidences <= upper)
        prop_in_bin = float(np.mean(in_bin))
        if prop_in_bin > 0:
            acc = float(np.mean(accuracies[in_bin]))
            conf = float(np.mean(confidences[in_bin]))
            bin_accs.append(acc)
            bin_confs.append(conf)
            ece += np.abs(acc - conf) * prop_in_bin

    fig, ax = plt.subplots(figsize=(7, 6), dpi=100)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Ideal Calibration")
    ax.plot(bin_confs, bin_accs, marker="o", color="#1f77b4", linewidth=2, label=f"Model (ECE={ece:.4f})")
    ax.bar(
        bin_confs,
        [abs(a - c) for a, c in zip(bin_accs, bin_confs, strict=False)],
        bottom=[min(a, c) for a, c in zip(bin_accs, bin_confs, strict=False)],
        width=0.04,
        alpha=0.3,
        color="#d62728",
        label="Calibration Gap",
    )

    ax.set_xlabel("Mean Predicted Confidence")
    ax.set_ylabel("Empirical Accuracy")
    ax.set_title(f"Holdout Reliability Diagram (10-bin ECE: {ece:.4f})")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    return fig


def _plot_pareto_frontier(
    profiles: list[ParetoProfileLike],
    latency_slo_ms: float = 25.0,
) -> matplotlib.figure.Figure:
    """Render backbone Pareto frontier comparing inference latency against parameter scale.

    Why:
        Visually situates the active model among candidate foundation backbones to verify empirical compliance against the target p95 latency ceiling.

    How:
        Extracts p95 latency, model parameters, and throughput from candidate profiles without dynamic reflection, renders a scatter plot with marker size proportional to model memory footprint, and marks the latency ceiling.

    Complexity:
        O(P) scatter plot construction complexity where P is the number of profiled backbones.

    Args:
        profiles: List of candidate backbone profile objects implementing ParetoProfileLike.
        latency_slo_ms: Target latency ceiling in milliseconds.

    Returns:
        Matplotlib figure instance.
    """
    fig, ax = plt.subplots(figsize=(8, 5), dpi=100)

    for p in profiles:
        lat = p.p95_latency_ms
        fps = p.throughput_fps
        name = p.extractor
        params_m = p.parameters_m
        ax.scatter(lat, fps, s=params_m * 8 + 50, alpha=0.75, edgecolors="black", label=f"{name} ({params_m:.1f}M)")
        ax.annotate(name, (lat, fps), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8)

    ax.axvline(
        latency_slo_ms, color="#d62728", linestyle="--", linewidth=1.5, label=f"SLO Ceiling ({latency_slo_ms} ms)"
    )
    ax.set_xlabel("p95 ONNX Inference Latency (ms)")
    ax.set_ylabel("Throughput (Frames Per Second)")
    ax.set_title("Candidate Backbone Latency vs. Throughput Frontier")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper right", fontsize=7)
    fig.tight_layout()
    return fig


def log_holdout_visualizations(payload: HoldoutVisualizationPayload) -> None:
    """Render and log complete holdout diagnostic visualisation figures to MLflow.

    Why:
        Holdout statistical verification (Stage 7 evaluate.py) requires publication-grade diagnostic artefacts documenting conformal set size bounds, 10-bin confidence calibration reliability, Helmholtz free energy OOD rejection thresholds, per-class precision-recall trade-offs, and candidate backbone Pareto efficiency.

    How:
        Extracts test predictions and metrics from the typed HoldoutVisualizationPayload, generates five discrete matplotlib figures, logs each figure to MLflow under the 'visualizations/' run artifact path, and deterministically closes each figure to prevent memory leaks.

    Complexity:
        O(N * K + P) where N is test sample count, K is class count, and P is candidate Pareto backbone count.

    Args:
        payload: Strongly-typed HoldoutVisualizationPayload container encapsulating holdout evaluation predictions, set sizes, and thresholds.
    """
    if not mlflow.active_run():
        return

    try:
        class_names = [
            str(t.get("common_name", t.get("scientific_name", f"Class {i}")))
            for i, t in enumerate(payload.taxa_catalog or [])
        ]
        num_classes = payload.test_logits.shape[1]
        if len(class_names) < num_classes:
            class_names.extend([f"Class {i}" for i in range(len(class_names), num_classes)])

        # 1. Conformal Set Cardinality Histogram
        fig_sizes = _plot_conformal_set_sizes(
            set_sizes=payload.set_sizes,
            k_max=payload.k_max,
            empirical_coverage=payload.empirical_coverage,
            nominal_coverage=payload.nominal_coverage,
        )
        mlflow.log_figure(fig_sizes, "visualizations/test_conformal_set_sizes.png")
        plt.close(fig_sizes)

        # 2. Reliability Diagram and ECE Gap
        fig_rel = _plot_reliability_diagram(
            test_probs=payload.test_probs,
            test_labels=payload.test_labels,
            n_bins=10,
        )
        mlflow.log_figure(fig_rel, "visualizations/test_reliability_diagram.png")
        plt.close(fig_rel)

        # 3. Energy Distribution and Epistemic Boundary
        fig_energy = _plot_energy_distribution(
            val_logits=payload.test_logits,
            calibrated_threshold=payload.calibrated_energy_threshold,
        )
        mlflow.log_figure(fig_energy, "visualizations/test_energy_distribution.png")
        plt.close(fig_energy)

        # 4. Holdout Confusion Matrix and Precision-Recall Curves
        pred_labels = np.argmax(payload.test_probs, axis=1)
        fig_cm = _plot_confusion_matrix(payload.test_labels, pred_labels, class_names)
        mlflow.log_figure(fig_cm, "visualizations/test_confusion_matrix.png")
        plt.close(fig_cm)

        fig_pr = _plot_precision_recall_curves(payload.test_labels, payload.test_probs, class_names)
        mlflow.log_figure(fig_pr, "visualizations/test_precision_recall_curves.png")
        plt.close(fig_pr)

        # 5. Pareto Frontier (if candidate profiles provided)
        if payload.pareto_profiles:
            fig_pareto = _plot_pareto_frontier(
                profiles=payload.pareto_profiles,
                latency_slo_ms=payload.latency_slo_ms,
            )
            mlflow.log_figure(fig_pareto, "visualizations/pareto_frontier.png")
            plt.close(fig_pareto)

    except Exception as err:
        logger.warning("Failed to render and log holdout visualisations to MLflow: %s", err)
