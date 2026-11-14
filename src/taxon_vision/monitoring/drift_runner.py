# SPDX-FileCopyrightText: 2026 Benjamin Förster
# SPDX-License-Identifier: MIT
"""Unified statistical drift evaluation runner.

Summary:
    Orchestrates continuous distribution drift evaluations across live production embeddings against holdout calibration baselines, logging statistical metrics to MLflow and persisting verdicts for telemetry scraping.

Why (Architecture Decisions & Trade-offs):
    Periodic drift evaluation must decouple heavy statistical matrix math (MMD permutation tests, PCA dimensionality reduction) and reporting generation from the request-serving hot path. Encapsulating this orchestration within a single runner enables synchronous execution by offline CLI tools, asynchronous offloading via FastAPI background workers (`anyio.to_thread.run_sync`), and scheduled execution by Airflow DAGs without code duplication.

How (Algorithmic Mechanism & Invariants):
    1. Loads and caches baseline calibration embeddings (`cal_embeddings.pt`) in system memory to eliminate repeated disk deserialisation.
    2. Queries the latest N prediction embeddings from TelemetryStore; aborts gracefully with a typed 'insufficient_data' verdict if sample count is below `min_samples`.
    3. Evaluates Tier 1 1D Wasserstein-1 distance on pre-normalised L2 norms against calibrated threshold tau_w1.
    4. Evaluates Tier 2 multivariate RBF MMD² with Monte Carlo permutation testing against significance ceiling alpha.
    5. Fits an 8-component PCA projection on reference embeddings, compiles a compact tabular projection, and renders a standalone HTML report via Evidently 0.7 Snapshot API.
    6. Logs metrics and the HTML report into MLflow experiment `taxon-vision-drift` using RAII run scoping.
    7. Persists a DriftRunRecord to the telemetry repository and returns a frozen DriftVerdict dataclass.

Standard Fields:
    References: Gretton et al. (2012) 'A Kernel Two-Sample Test'; Evidently AI Documentation (0.7+).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
from evidently import Report
from evidently.core.datasets import Dataset
from evidently.core.metric_types import SingleValue, SingleValueCalculation, SingleValueMetric
from evidently.core.report import Context
from evidently.presets import DataDriftPreset
from evidently.ui.workspace import Workspace
from sklearn.decomposition import PCA

from taxon_vision.config import Settings, get_settings
from taxon_vision.models.training.tracking import mlflow_run_scope
from taxon_vision.monitoring.drift import compute_mmd2_permutation, compute_wasserstein_drift
from taxon_vision.service.telemetry_store import (
    DriftRunRecord,
    TelemetryStore,
    get_telemetry_store,
)

logger = logging.getLogger(__name__)

_REFERENCE_CACHE: npt.NDArray[np.float32] | None = None


@dataclass(frozen=True)
class DriftVerdict:
    """Immutable outcome of an end-to-end drift evaluation cycle.

    Attributes:
        drift_detected: Boolean flag indicating if Tier 1 or Tier 2 threshold was exceeded.
        status: Evaluation status string ('evaluated', 'insufficient_data', 'reference_missing').
        w1_distance: Observed Wasserstein-1 distance on embedding norms.
        w1_threshold: Calibrated Wasserstein-1 alert threshold.
        mmd2: Observed Maximum Mean Discrepancy squared, or None if skipped.
        mmd2_p_value: Finite-sample permutation test p-value, or None if skipped.
        window_size: Number of live observation vectors evaluated in this cycle.
        n_reference: Number of baseline calibration observations evaluated.
        mlflow_run_id: MLflow run identifier if successfully logged, or None.
        report_path: Path to rendered Evidently HTML report on disk, or None.
    """

    drift_detected: bool
    status: str
    w1_distance: float
    w1_threshold: float
    mmd2: float | None
    mmd2_p_value: float | None
    window_size: int
    n_reference: int
    mlflow_run_id: str | None
    report_path: str | None


class PrecomputedDriftMetric(SingleValueMetric):  # type: ignore[misc]
    """Immutable Evidently SingleValueMetric carrier for exact precomputed Tier 1 and Tier 2 drift scalars.

    Summary:
        Wraps an already evaluated scalar drift statistic (`W_1`, `MMD_u^2`, or permutation p-value) from `drift.py` as an Evidently 0.7 `SingleValueMetric` dashboard card.

    Why:
        Evidently's `DataDriftPreset` evaluates column shifts across the compact `(1 + 8)` norm plus PCA projection dataframe, whereas authoritative drift gating relies on exact L2 norms (`W_1`) and full-dimensional unbiased RBF MMD^2 permutation tests (`MMD_u^2`). Surfacing these precomputed scalars as top-level snapshot cards prevents reporting discrepancy between statistical tests and dashboard telemetry.

    How:
        Declares `metric_name` and `value` as validated Pydantic attributes consumed directly by `PrecomputedDriftCalculation`.
    """

    metric_name: str
    value: float


class PrecomputedDriftCalculation(SingleValueCalculation[PrecomputedDriftMetric]):  # type: ignore[misc]
    """Evidently 0.7 calculation adapter rendering `PrecomputedDriftMetric` scalar cards.

    Summary:
        Implements Evidently's `SingleValueCalculation` protocol for `PrecomputedDriftMetric`.

    Why:
        Decouples metric identity from calculation state in Evidently 0.7 whilst maintaining strict static type guarantees without dynamic reflection.

    How:
        Returns the configured `metric_name` as display label and emits `self.result(float(self.metric.value))` inside `calculate(...)`.
    """

    def display_name(self) -> str:
        """Return the human-readable dashboard label for this drift scalar."""
        return self.metric.metric_name

    def calculate(
        self,
        context: Context,
        current_data: Dataset,
        reference_data: Dataset | None,
    ) -> SingleValue:
        """Emit the precomputed scalar wrapped as an Evidently SingleValue result."""
        _ = (context, current_data, reference_data)
        return self.result(float(self.metric.value))


def _build_evidently_metrics(
    *,
    w1_distance: float | None,
    w1_threshold: float | None,
    mmd2: float | None,
    mmd2_p_value: float | None,
    mmd2_alpha: float | None,
) -> list[Any]:
    """Assemble Evidently 0.7 summary scalar cards and Wasserstein-configured DataDriftPreset."""
    metrics: list[Any] = []
    if w1_distance is not None:
        w1_label = (
            f"Tier 1 W1 Distance (tau <= {w1_threshold:.4f})" if w1_threshold is not None else "Tier 1 W1 Distance"
        )
        metrics.append(PrecomputedDriftMetric(metric_name=w1_label, value=float(w1_distance)))
    if mmd2 is not None:
        metrics.append(PrecomputedDriftMetric(metric_name="Tier 2 RBF MMD^2 (Unbiased)", value=float(mmd2)))
    if mmd2_p_value is not None:
        p_label = (
            f"Tier 2 MMD^2 Permutation p-value (alpha = {mmd2_alpha:.4f})"
            if mmd2_alpha is not None
            else "Tier 2 MMD^2 Permutation p-value"
        )
        metrics.append(PrecomputedDriftMetric(metric_name=p_label, value=float(mmd2_p_value)))
    metrics.append(DataDriftPreset(num_method="wasserstein", num_threshold=0.1))
    return metrics


def load_embeddings_array(path: Path | str) -> npt.NDArray[np.float32]:
    """Load pre-computed embeddings from a PyTorch checkpoint into a contiguous float32 NumPy array.

    Why:
        Provides an allocation-disciplined loader for offline pipeline stages and drift runners without polluting module-level caches.

    How:
        Deserialises the PyTorch dictionary or tensor checkpoint, extracts the 'embeddings' key if present, detaches to CPU, converts to float32, and returns a contiguous C-order array.

    Args:
        path: Relative or absolute file path to the PyTorch checkpoint (.pt).

    Returns:
        Contiguous float32 NumPy array of shape (N, D).

    Raises:
        FileNotFoundError: If the specified checkpoint does not exist on disk.
        ValueError: If the checkpoint payload is neither a Tensor nor a dictionary with an 'embeddings' key.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Embeddings file not found: {p}")

    checkpoint: Any = torch.load(p, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict) and "embeddings" in checkpoint:
        tensor = checkpoint["embeddings"]
    elif isinstance(checkpoint, torch.Tensor):
        tensor = checkpoint
    else:
        raise ValueError(f"Unrecognised checkpoint format in {p}")

    arr = tensor.detach().cpu().numpy().astype(np.float32)
    return np.ascontiguousarray(arr)


def load_reference_embeddings(path_str: str) -> npt.NDArray[np.float32]:
    """Load and cache pre-normalised reference embeddings from a PyTorch tensor checkpoint.

    Why:
        Eliminates repeated disk I/O and unpickling overhead during frequent drift monitoring sweeps.

    How:
        Checks module-level cache; if absent, deserialises the PyTorch dictionary checkpoint via load_embeddings_array and caches in system memory.

    Args:
        path_str: Relative or absolute file path to the reference PyTorch checkpoint.

    Returns:
        Contiguous float32 NumPy array of shape (N, D).

    Raises:
        FileNotFoundError: If the specified reference checkpoint does not exist on disk.
    """
    global _REFERENCE_CACHE
    if _REFERENCE_CACHE is not None:
        return _REFERENCE_CACHE

    path = Path(path_str)
    if not path.is_file():
        raise FileNotFoundError(f"Reference embeddings file not found: {path}")

    _REFERENCE_CACHE = load_embeddings_array(path)
    return _REFERENCE_CACHE


def _persist_to_evidently_workspace(
    snapshot: Any,
    workspace_dir: Path,
    project_name: str,
    run_name: str | None = None,
) -> None:
    """Persist an evaluated Evidently snapshot into the mounted Evidently UI Workspace directory.

    Why:
        Rendering standalone HTML files isolates each drift check from historical longitudinal series rendered by the cluster Evidently UI container on port 30850. Writing the snapshot directly into the mounted workspace directory populates interactive project timelines without blocking serving workers.

    How:
        Ensures workspace_dir exists, opens or creates the workspace via Workspace.create, looks up project_name with search_project (creating the project if absent), and records the snapshot metadata via add_run(include_data=False).

    Args:
        snapshot: Evaluated Evidently Snapshot produced by Report.run().
        workspace_dir: Target workspace directory served by evidently-service.
        project_name: Project display name in the Evidently UI.
        run_name: Optional human-readable snapshot timestamp label.
    """
    workspace_dir.mkdir(parents=True, exist_ok=True)
    ws = Workspace.create(str(workspace_dir))
    projects = ws.search_project(project_name)
    project = (
        projects[0]
        if projects
        else ws.create_project(
            name=project_name,
            description="TaxonVision hourly embedding distribution drift monitoring",
        )
    )
    ws.add_run(project.id, snapshot, include_data=False, name=run_name)


def generate_evidently_drift_report(
    reference: npt.NDArray[np.float32],
    current: npt.NDArray[np.float32],
    output_path: Path,
    *,
    workspace_dir: Path | None = None,
    project_name: str = "TaxonVision Production Drift",
    run_name: str | None = None,
    w1_distance: float | None = None,
    w1_threshold: float | None = None,
    mmd2: float | None = None,
    mmd2_p_value: float | None = None,
    mmd2_alpha: float | None = None,
) -> Path:
    """Generate and save an Evidently DataDriftPreset report and optional workspace snapshot using PCA projection.

    Summary:
        Produces an HTML drift report and optional longitudinal Evidently workspace snapshot combining exact Tier 1/2 scalar cards with Wasserstein feature drift checks across embedding norm and PCA projections.

    Why:
        High-dimensional embedding tables (384 to 1280 columns) result in slow report generation and visually illegible drift matrices. Projecting vectors down to Euclidean norms plus the top 8 principal components yields an interpretable, fast tabular snapshot, whilst configuring `DataDriftPreset(num_method="wasserstein", num_threshold=0.1)` aligns Evidently's per-column drift detection with the Tier 1 Wasserstein distance metric alongside exact precomputed full-space scalars.

    How:
        Fits an 8-component PCA model on reference vectors, projects both reference and current matrices, computes L2 vector norms, builds pandas DataFrames, prepends optional PrecomputedDriftMetric scalar cards before DataDriftPreset(num_method="wasserstein", num_threshold=0.1), runs Evidently's Report snapshot, saves the standalone HTML file, and optionally persists the snapshot to an Evidently UI Workspace.

    Args:
        reference: Reference embedding matrix of shape (M, D).
        current: Current observation embedding matrix of shape (N, D).
        output_path: Destination file path for the standalone HTML report.
        workspace_dir: Optional Evidently UI workspace directory for longitudinal dashboard persistence.
        project_name: Project name within workspace_dir when persisting runs.
        run_name: Optional human-readable label attached to the stored workspace run.
        w1_distance: Optional precomputed Tier 1 Wasserstein-1 distance on L2 norms.
        w1_threshold: Optional calibrated Tier 1 Wasserstein-1 alert threshold tau_w1.
        mmd2: Optional precomputed Tier 2 unbiased RBF MMD^2 statistic.
        mmd2_p_value: Optional Monte Carlo permutation test p-value for Tier 2 MMD^2.
        mmd2_alpha: Optional significance threshold alpha for Tier 2 permutation testing.

    Returns:
        Destination path of the rendered HTML report.
    """
    n_components = min(8, reference.shape[1], reference.shape[0])
    pca = PCA(n_components=n_components, random_state=42)
    ref_pca = pca.fit_transform(reference)
    curr_pca = pca.transform(current)

    ref_norms = np.linalg.norm(reference, axis=-1, keepdims=True)
    curr_norms = np.linalg.norm(current, axis=-1, keepdims=True)

    pca_cols = [f"pca_{i}" for i in range(n_components)]
    columns = ["norm", *pca_cols]

    ref_df = pd.DataFrame(np.hstack([ref_norms, ref_pca]), columns=columns)
    curr_df = pd.DataFrame(np.hstack([curr_norms, curr_pca]), columns=columns)

    evidently_metrics = _build_evidently_metrics(
        w1_distance=w1_distance,
        w1_threshold=w1_threshold,
        mmd2=mmd2,
        mmd2_p_value=mmd2_p_value,
        mmd2_alpha=mmd2_alpha,
    )
    report = Report(metrics=evidently_metrics)
    snapshot = report.run(reference_data=ref_df, current_data=curr_df)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot.save_html(str(output_path))
    if workspace_dir is not None:
        try:
            _persist_to_evidently_workspace(
                snapshot=snapshot,
                workspace_dir=workspace_dir,
                project_name=project_name,
                run_name=run_name,
            )
        except Exception:
            logger.exception("Failed to persist drift snapshot to Evidently workspace at %s", workspace_dir)
    return output_path


def evaluate_recent_drift(
    store: TelemetryStore | None = None,
    settings: Settings | None = None,
    extractor: str | None = None,
    reference_embeddings: npt.NDArray[np.float32] | None = None,
) -> DriftVerdict:
    """Execute end-to-end drift evaluation cycle over the latest prediction window.

    Why:
        Enforces statistical exchangeability auditing on live service embeddings, guaranteeing that marginal conformal coverage assumptions hold in active production serving.

    How:
        1. Obtains reference embeddings via cache or injected array.
        2. Retrieves the most recent N observation vectors from TelemetryStore.
        3. Checks sample threshold against min_samples; returns insufficient_data if below.
        4. Calculates Tier 1 Wasserstein-1 distance on embedding norms.
        5. Calculates Tier 2 multivariate RBF MMD² with permutation testing.
        6. Renders Evidently PCA drift report to `reports/drift_report.html`.
        7. Logs metrics, tags, and HTML artefact to MLflow `taxon-vision-drift`.
        8. Records DriftRunRecord in TelemetryStore and returns typed DriftVerdict.

    Args:
        store: Telemetry store instance to query, or None to use default singleton.
        settings: Application settings, or None to load from environment.
        extractor: Optional model extractor name to filter live embeddings.
        reference_embeddings: Optional pre-loaded reference array for test isolation.

    Returns:
        Immutable DriftVerdict dataclass detailing statistical outcomes and run metadata.
    """
    resolved_settings = settings or get_settings()
    resolved_store = store or get_telemetry_store()

    # 1. Resolve Reference Embeddings
    ref = reference_embeddings
    if ref is None:
        try:
            ref = load_reference_embeddings(resolved_settings.drift.reference_embeddings_path)
        except Exception:
            logger.exception("Failed to load reference embeddings for drift detection")
            return DriftVerdict(
                drift_detected=False,
                status="reference_missing",
                w1_distance=0.0,
                w1_threshold=resolved_settings.drift.tau_w1,
                mmd2=None,
                mmd2_p_value=None,
                window_size=0,
                n_reference=0,
                mlflow_run_id=None,
                report_path=None,
            )

    # 2. Query Live Window
    current = resolved_store.recent_embeddings(
        limit=resolved_settings.drift.window_size,
        extractor=extractor,
    )

    if current.shape[0] < resolved_settings.drift.min_samples:
        logger.info(
            "Insufficient live predictions for drift detection: %d < %d",
            current.shape[0],
            resolved_settings.drift.min_samples,
        )
        return DriftVerdict(
            drift_detected=False,
            status="insufficient_data",
            w1_distance=0.0,
            w1_threshold=resolved_settings.drift.tau_w1,
            mmd2=None,
            mmd2_p_value=None,
            window_size=current.shape[0],
            n_reference=ref.shape[0],
            mlflow_run_id=None,
            report_path=None,
        )

    # 3. Tier 1: 1D Wasserstein-1 on Embedding Norms
    tau_w1 = resolved_settings.drift.tau_w1
    w1_distance = compute_wasserstein_drift(ref, current)

    # 4. Tier 2: Multivariate RBF MMD² Permutation Test
    mmd_res = compute_mmd2_permutation(
        ref,
        current,
        max_samples=resolved_settings.drift.mmd_max_samples,
        n_permutations=resolved_settings.drift.mmd_permutations,
        seed=42,
    )
    mmd2 = mmd_res.mmd2
    p_val = mmd_res.p_value

    # 5. Composite Drift Verdict
    drift_detected = bool(w1_distance > tau_w1 or p_val < resolved_settings.drift.alpha)

    # 6. Render Evidently Compact Projection Report & Persist to Workspace
    report_file = Path("reports/drift_report.html")
    run_label = f"drift_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}"
    try:
        generate_evidently_drift_report(
            ref,
            current,
            report_file,
            workspace_dir=Path(resolved_settings.drift.evidently_workspace_dir),
            project_name=resolved_settings.drift.evidently_project_name,
            run_name=run_label,
            w1_distance=w1_distance,
            w1_threshold=tau_w1,
            mmd2=mmd2,
            mmd2_p_value=p_val,
            mmd2_alpha=resolved_settings.drift.alpha,
        )
        report_path_str: str | None = str(report_file)
    except Exception:
        logger.exception("Failed to render Evidently drift report")
        report_path_str = None

    # 7. Log Directly to Configured DagsHub MLflow Run Scope
    mlflow_run_id: str | None = None
    experiment_name = resolved_settings.mlflow.drift_experiment_name
    try:
        with mlflow_run_scope(run_name="drift_evaluation", experiment_name=experiment_name) as run:
            if run is not None:
                mlflow_run_id = run.info.run_id
                mlflow.log_metrics(
                    {
                        "drift_w1": float(w1_distance),
                        "drift_w1_threshold": float(tau_w1),
                        "drift_mmd2": float(mmd2),
                        "drift_mmd2_p_value": float(p_val),
                        "window_size": float(current.shape[0]),
                    }
                )
                mlflow.set_tag("drift_detected", str(drift_detected).lower())
                if report_path_str and Path(report_path_str).is_file():
                    mlflow.log_artifact(report_path_str)
    except Exception:
        logger.exception("Failed to log drift evaluation metrics to MLflow")

    # 8. Record in Telemetry Store
    record = DriftRunRecord(
        id=None,
        created_at=datetime.now(UTC),
        window_size=current.shape[0],
        w1_distance=w1_distance,
        w1_threshold=tau_w1,
        mmd2=mmd2,
        mmd2_p_value=p_val,
        drift_detected=drift_detected,
        mlflow_run_id=mlflow_run_id,
    )
    resolved_store.record_drift_run(record)

    return DriftVerdict(
        drift_detected=drift_detected,
        status="evaluated",
        w1_distance=w1_distance,
        w1_threshold=tau_w1,
        mmd2=mmd2,
        mmd2_p_value=p_val,
        window_size=current.shape[0],
        n_reference=ref.shape[0],
        mlflow_run_id=mlflow_run_id,
        report_path=report_path_str,
    )
