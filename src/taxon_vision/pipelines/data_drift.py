# SPDX-FileCopyrightText: 2026 Benjamin Förster
# SPDX-License-Identifier: MIT
"""Stage 3.5: Training Data Representation Drift Audit Pipeline.

Summary:
    Evaluates distribution shift between historical baseline reference embeddings and newly extracted training embeddings, logging statistical tests and Evidently reports to MLflow.

Why (Architectural Rationale & Trade-offs):
    Retraining on newly aggregated observations risks silent manifold distortion, covariate shift, or semantic drift relative to the historical production baseline. Evaluating drift immediately after feature extraction (Epistemic Diamond DAG Stage 3.5) creates an immutable historical-vs-recent data lineage audit (R7) prior to classification head updates. Gating or logging drift on the training embeddings ensures that downstream conformal coverage and empirical confidence guarantees remain statistically interpretable.

How (Algorithmic Mechanism & Invariants):
    1. Loads reference embeddings and current training embeddings as contiguous C-order float32 NumPy arrays via load_embeddings_array.
    2. Validates dimensional compatibility between reference and current representations.
    3. Computes Tier 1 1D Wasserstein-1 distance on pre-normalised L2 norms against calibrated alert ceiling tau_w1.
    4. Computes Tier 2 multivariate RBF MMD² with Monte Carlo permutation testing against significance ceiling alpha.
    5. Projects embeddings onto an 8-D PCA subspace and renders an Evidently DataDriftPreset HTML report snapshot.
    6. Logs parameters, metrics, tags ('drift_scope': 'training'), and the HTML artefact into MLflow using RAII run scoping.
    7. Persists an immutable metrics summary JSON to disk for DVC pipeline dependency tracking.

Complexity:
    O(B * (M + N)) for Tier 2 MMD permutation testing with B iterations and subsample budgets, plus O(D * min(M, N)^2) for PCA projection.

Standard Fields:
    References: Gretton et al. (2012) 'A Kernel Two-Sample Test'; Evidently AI 0.7+ Documentation.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import mlflow
import yaml

from taxon_vision.config import Settings, get_settings
from taxon_vision.models.training.tracking import mlflow_run_scope
from taxon_vision.monitoring.drift import compute_mmd2_permutation, compute_wasserstein_drift
from taxon_vision.monitoring.drift_runner import (
    generate_evidently_drift_report,
    load_embeddings_array,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DataDriftPipelineConfig:
    """Immutable configuration container for training data drift evaluation.

    Attributes:
        reference_path: Path to baseline reference PyTorch embeddings (.pt).
        current_path: Path to newly extracted training PyTorch embeddings (.pt).
        metrics_out_path: Destination path for summary metrics JSON report.
        report_out_path: Destination path for rendered Evidently standalone HTML report.
        tau_w1: Calibrated Wasserstein-1 alert threshold.
        alpha: Significance level ceiling for MMD permutation hypothesis rejection.
        max_samples: Subsample size limit for MMD kernel computation.
        permutations: Number of permutation draws for MMD empirical p-value estimation.
        seed: Random seed for deterministic MMD permutation draws.
    """

    reference_path: Path
    current_path: Path
    metrics_out_path: Path
    report_out_path: Path
    tau_w1: float
    alpha: float
    max_samples: int = 500
    permutations: int = 100
    seed: int = 42


@dataclass(frozen=True)
class DataDriftPipelineResult:
    """Immutable result of the training data drift evaluation.

    Attributes:
        drift_detected: Boolean flag indicating if Tier 1 or Tier 2 threshold was breached.
        w1_distance: Observed 1D Wasserstein-1 distance on embedding norms.
        tau_w1: Calibrated Wasserstein-1 threshold used for the alert check.
        mmd2: Observed Maximum Mean Discrepancy squared statistic.
        mmd2_p_value: Monte Carlo permutation empirical p-value.
        alpha: Significance level ceiling used for hypothesis rejection.
        n_reference: Number of reference observations evaluated.
        n_current: Number of current observations evaluated.
        feature_dim: Dimensionality of the embedding vectors.
        mlflow_run_id: MLflow run identifier if logged, else None.
        report_path: Path to the generated Evidently HTML report on disk.
    """

    drift_detected: bool
    w1_distance: float
    tau_w1: float
    mmd2: float
    mmd2_p_value: float
    alpha: float
    n_reference: int
    n_current: int
    feature_dim: int
    mlflow_run_id: str | None
    report_path: Path


def _resolve_drift_thresholds(
    config_path: Path | str = Path("config/drift_config.yaml"),
    settings: Settings | None = None,
) -> tuple[float, float]:
    """Resolve calibrated drift alert threshold tau_w1 and alpha from config or settings.

    Args:
        config_path: Path to calibrated drift configuration YAML.
        settings: Application settings fallback.

    Returns:
        Tuple of (tau_w1, alpha).
    """
    p = Path(config_path)
    if p.is_file():
        try:
            with p.open("r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            if isinstance(data, dict) and "drift" in data:
                drift_dict = data["drift"]
                tau_w1 = float(drift_dict.get("tau_w1", 0.15))
                alpha = float(drift_dict.get("alpha", 0.01))
                return tau_w1, alpha
        except Exception:
            logger.exception("Failed to parse drift config from %s; falling back to settings", p)

    resolved = settings or get_settings()
    return float(resolved.drift.tau_w1), float(resolved.drift.alpha)


def _log_data_drift_mlflow(
    config: DataDriftPipelineConfig,
    w1_distance: float,
    mmd2: float,
    mmd2_p_value: float,
    drift_detected: bool,
    n_reference: int,
    n_current: int,
    feature_dim: int,
    settings: Settings | None = None,
) -> str | None:
    """Log training data drift metrics and artefacts to MLflow under RAII run scope.

    Args:
        config: Pipeline configuration with file paths and parameters.
        w1_distance: Computed Wasserstein-1 distance.
        mmd2: Computed MMD² statistic.
        mmd2_p_value: Computed MMD² empirical p-value.
        drift_detected: Boolean alert flag.
        n_reference: Count of reference vectors.
        n_current: Count of current vectors.
        feature_dim: Representation vector dimensionality.
        settings: Application settings fallback.

    Returns:
        MLflow run ID string if successfully logged, or None.
    """
    resolved_settings = settings or get_settings()
    experiment_name = resolved_settings.mlflow.experiment_name

    try:
        with mlflow_run_scope(run_name="training_data_drift", experiment_name=experiment_name) as run:
            if run is None:
                return None

            run_id = run.info.run_id
            mlflow.set_tag("drift_scope", "training")
            mlflow.set_tag("stage", "data_drift")
            mlflow.set_tag("drift_detected", str(drift_detected).lower())

            mlflow.log_params(
                {
                    "data_drift.reference_path": str(config.reference_path),
                    "data_drift.current_path": str(config.current_path),
                    "data_drift.n_reference": n_reference,
                    "data_drift.n_current": n_current,
                    "data_drift.feature_dim": feature_dim,
                    "data_drift.tau_w1": config.tau_w1,
                    "data_drift.alpha": config.alpha,
                }
            )

            mlflow.log_metrics(
                {
                    "data_drift.w1_distance": float(w1_distance),
                    "data_drift.w1_threshold": float(config.tau_w1),
                    "data_drift.mmd2": float(mmd2),
                    "data_drift.mmd2_p_value": float(mmd2_p_value),
                    "data_drift.drift_detected": 1.0 if drift_detected else 0.0,
                }
            )

            if config.report_out_path.is_file():
                mlflow.log_artifact(str(config.report_out_path))

            return run_id
    except Exception:
        logger.exception("Failed to log training data drift run to MLflow")
        return None


def _persist_metrics_json(path: Path, result: DataDriftPipelineResult) -> None:
    """Persist structured summary metrics to disk for DVC metrics tracking.

    Args:
        path: Output file destination path.
        result: Evaluated pipeline result dataclass.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "w1_distance": float(result.w1_distance),
        "tau_w1": float(result.tau_w1),
        "mmd2": float(result.mmd2),
        "mmd2_p_value": float(result.mmd2_p_value),
        "drift_detected": result.drift_detected,
        "n_reference": result.n_reference,
        "n_current": result.n_current,
        "feature_dim": result.feature_dim,
        "mlflow_run_id": result.mlflow_run_id,
        "report_path": str(result.report_path),
        "evaluated_at": datetime.now(UTC).isoformat(),
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def evaluate_training_data_drift(
    config: DataDriftPipelineConfig,
    settings: Settings | None = None,
) -> DataDriftPipelineResult:
    """Execute end-to-end drift audit comparing historical reference and current training embeddings.

    Why:
        Detects representation shift and covariate instability before retraining begins, preserving historical lineage and mathematical conformal exchangeability.

    How:
        1. Deserialises reference and current embeddings via load_embeddings_array.
        2. Asserts dimension matching.
        3. Computes Tier 1 1D Wasserstein-1 distance on vector norms.
        4. Computes Tier 2 multivariate RBF MMD² with permutation testing.
        5. Renders compact PCA-projected Evidently report snapshot.
        6. Logs metrics and artefacts to MLflow under drift_scope='training'.
        7. Writes metrics JSON to disk and returns a frozen result dataclass.

    Args:
        config: Typed pipeline configuration.
        settings: Application settings fallback.

    Returns:
        DataDriftPipelineResult summarizing test statistics, drift alert flag, and report paths.

    Raises:
        FileNotFoundError: If reference or current embedding files do not exist.
        ValueError: If embedding dimensions do not match.
    """
    logger.info("Loading reference embeddings from: %s", config.reference_path)
    ref = load_embeddings_array(config.reference_path)

    logger.info("Loading current training embeddings from: %s", config.current_path)
    curr = load_embeddings_array(config.current_path)

    if ref.shape[1] != curr.shape[1]:
        raise ValueError(
            f"Feature dimension mismatch: reference has dim {ref.shape[1]}, but current has dim {curr.shape[1]}"
        )

    # Tier 1: 1D Wasserstein-1 on Embedding Norms
    w1_distance = compute_wasserstein_drift(ref, curr)

    # Tier 2: Multivariate RBF MMD² Permutation Test
    mmd_res = compute_mmd2_permutation(
        ref,
        curr,
        max_samples=config.max_samples,
        n_permutations=config.permutations,
        seed=config.seed,
    )

    drift_detected = bool(w1_distance > config.tau_w1 or mmd_res.p_value < config.alpha)

    # Evidently PCA Snapshot Report
    logger.info("Rendering Evidently drift report to: %s", config.report_out_path)
    try:
        generate_evidently_drift_report(ref, curr, config.report_out_path)
    except Exception:
        logger.exception("Failed to render Evidently drift report snapshot")

    # MLflow Logging
    mlflow_run_id = _log_data_drift_mlflow(
        config=config,
        w1_distance=w1_distance,
        mmd2=mmd_res.mmd2,
        mmd2_p_value=mmd_res.p_value,
        drift_detected=drift_detected,
        n_reference=ref.shape[0],
        n_current=curr.shape[0],
        feature_dim=ref.shape[1],
        settings=settings,
    )

    result = DataDriftPipelineResult(
        drift_detected=drift_detected,
        w1_distance=w1_distance,
        tau_w1=config.tau_w1,
        mmd2=mmd_res.mmd2,
        mmd2_p_value=mmd_res.p_value,
        alpha=config.alpha,
        n_reference=ref.shape[0],
        n_current=curr.shape[0],
        feature_dim=ref.shape[1],
        mlflow_run_id=mlflow_run_id,
        report_path=config.report_out_path,
    )

    _persist_metrics_json(config.metrics_out_path, result)
    logger.info(
        "Data drift evaluation complete: W1=%.4f (tau=%.4f), MMD²=%.4f (p=%.4f), drift_detected=%s",
        w1_distance,
        config.tau_w1,
        mmd_res.mmd2,
        mmd_res.p_value,
        drift_detected,
    )
    return result


def build_parser() -> argparse.ArgumentParser:
    """Construct command-line argument parser for Stage 3.5 data drift audit.

    Returns:
        Configured ArgumentParser instance.
    """
    parser = argparse.ArgumentParser(
        description="Audit representation drift between historical reference and current training embeddings.",
    )
    parser.add_argument(
        "--reference",
        type=Path,
        default=Path("data/reference/reference_train_embeddings.pt"),
        help="Path to baseline reference training embeddings (.pt)",
    )
    parser.add_argument(
        "--current",
        type=Path,
        default=Path("data/embeddings/train_embeddings.pt"),
        help="Path to current training embeddings (.pt)",
    )
    parser.add_argument(
        "--metrics-out",
        type=Path,
        default=Path("reports/data_drift.json"),
        help="Path to output summary metrics JSON file",
    )
    parser.add_argument(
        "--report-out",
        type=Path,
        default=Path("reports/data_drift_report.html"),
        help="Path to output Evidently standalone HTML report",
    )
    parser.add_argument(
        "--tau-w1",
        type=float,
        default=None,
        help="Override calibrated 1D Wasserstein-1 threshold",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=None,
        help="Override MMD significance ceiling",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for Stage 3.5 data drift audit pipeline.

    Args:
        argv: Command-line arguments, or None to read from sys.argv.

    Returns:
        Exit code: 0 on successful evaluation, 1 on unhandled exception.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        settings = get_settings()
        calibrated_tau, calibrated_alpha = _resolve_drift_thresholds(
            config_path=Path(settings.drift.config_path),
            settings=settings,
        )
        resolved_tau = args.tau_w1 if args.tau_w1 is not None else calibrated_tau
        resolved_alpha = args.alpha if args.alpha is not None else calibrated_alpha

        config = DataDriftPipelineConfig(
            reference_path=args.reference,
            current_path=args.current,
            metrics_out_path=args.metrics_out,
            report_out_path=args.report_out,
            tau_w1=resolved_tau,
            alpha=resolved_alpha,
            max_samples=settings.drift.mmd_max_samples,
            permutations=settings.drift.mmd_permutations,
        )
        evaluate_training_data_drift(config, settings=settings)
        return 0
    except Exception:
        logger.exception("Fatal error during training data drift evaluation")
        return 1


if __name__ == "__main__":
    sys.exit(main())
