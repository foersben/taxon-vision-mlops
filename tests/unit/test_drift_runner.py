# SPDX-FileCopyrightText: 2026 Benjamin Förster
# SPDX-License-Identifier: MIT
"""Unit tests for the unified drift evaluation runner, Evidently PCA reporting, and MLflow telemetry."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
import torch

import taxon_vision.monitoring.drift_runner as drift_runner
from taxon_vision.config import get_settings
from taxon_vision.monitoring.drift_runner import (
    DriftVerdict,
    evaluate_recent_drift,
    generate_evidently_drift_report,
    load_reference_embeddings,
)
from taxon_vision.service.telemetry_store import (
    BoundedInMemoryTelemetryStore,
    PredictionRecord,
)


@pytest.fixture(autouse=True)
def reset_reference_cache() -> Iterator[None]:
    """Ensure the module-level reference cache is cleared before and after each test."""
    drift_runner._REFERENCE_CACHE = None
    yield
    drift_runner._REFERENCE_CACHE = None


def test_load_reference_embeddings_from_dict_checkpoint(tmp_path: Path) -> None:
    """Verify loading pre-normalised reference tensor from a dictionary checkpoint."""
    tensor = torch.randn(30, 16, dtype=torch.float32)
    checkpoint_file = tmp_path / "cal_embeddings.pt"
    torch.save({"embeddings": tensor, "feature_dim": 16}, checkpoint_file)

    loaded = load_reference_embeddings(str(checkpoint_file))
    assert isinstance(loaded, np.ndarray)
    assert loaded.shape == (30, 16)
    assert loaded.dtype == np.float32

    # Second call should return the exact cached object
    cached = load_reference_embeddings(str(checkpoint_file))
    assert cached is loaded


def test_load_reference_embeddings_from_raw_tensor(tmp_path: Path) -> None:
    """Verify loading pre-normalised reference tensor when saved directly as a Tensor."""
    tensor = torch.randn(25, 32, dtype=torch.float32)
    checkpoint_file = tmp_path / "raw_tensor.pt"
    torch.save(tensor, checkpoint_file)

    loaded = load_reference_embeddings(str(checkpoint_file))
    assert loaded.shape == (25, 32)
    assert loaded.dtype == np.float32


def test_load_reference_embeddings_file_not_found() -> None:
    """Verify FileNotFoundError is raised when pointing to a non-existent path."""
    with pytest.raises(FileNotFoundError, match="Reference embeddings file not found"):
        load_reference_embeddings("non_existent_directory/cal_embeddings.pt")


def test_load_reference_embeddings_invalid_format(tmp_path: Path) -> None:
    """Verify ValueError is raised when checkpoint has an unexpected structure."""
    checkpoint_file = tmp_path / "invalid.pt"
    torch.save(["not", "a", "valid", "checkpoint"], checkpoint_file)

    with pytest.raises(ValueError, match="Unrecognised checkpoint format"):
        load_reference_embeddings(str(checkpoint_file))


def test_generate_evidently_drift_report(tmp_path: Path) -> None:
    """Verify generation of a standalone Evidently HTML report using PCA projection."""
    rng = np.random.default_rng(42)
    ref = rng.standard_normal((40, 16)).astype(np.float32)
    curr = rng.standard_normal((40, 16)).astype(np.float32) + 0.5
    out_file = tmp_path / "reports" / "test_report.html"

    result_path = generate_evidently_drift_report(ref, curr, out_file)
    assert result_path == out_file
    assert out_file.is_file()
    assert out_file.stat().st_size > 1000


def test_evaluate_recent_drift_reference_missing() -> None:
    """Verify graceful handling when reference embeddings checkpoint is missing."""
    settings = get_settings().model_copy(deep=True)
    settings.drift.reference_embeddings_path = "non_existent/cal_embeddings.pt"
    store = BoundedInMemoryTelemetryStore(maxlen=100)

    verdict = evaluate_recent_drift(store=store, settings=settings)
    assert isinstance(verdict, DriftVerdict)
    assert verdict.status == "reference_missing"
    assert verdict.drift_detected is False
    assert verdict.window_size == 0
    assert verdict.mlflow_run_id is None


def test_evaluate_recent_drift_insufficient_data() -> None:
    """Verify typed insufficient_data outcome when live predictions fall below min_samples."""
    settings = get_settings().model_copy(deep=True)
    settings.drift.min_samples = 50
    store = BoundedInMemoryTelemetryStore(maxlen=100)

    # Insert only 10 predictions
    for i in range(10):
        store.record(
            PredictionRecord(
                observation_id=f"obs_{i}",
                model_version="1.0.0",
                extractor="dinov3_vits16",
                top1_taxon=1,
                top1_confidence=0.9,
                conformal_set=[1],
                ood_energy=-3.0,
                is_ood=False,
                requires_review=False,
                latency_ms=5.0,
                embedding=np.ones(16, dtype=np.float32),
            )
        )

    ref = np.random.randn(60, 16).astype(np.float32)
    verdict = evaluate_recent_drift(
        store=store,
        settings=settings,
        reference_embeddings=ref,
    )

    assert verdict.status == "insufficient_data"
    assert verdict.drift_detected is False
    assert verdict.window_size == 10
    assert verdict.n_reference == 60
    assert verdict.mmd2 is None
    assert verdict.mmd2_p_value is None


def test_evaluate_recent_drift_in_distribution(tmp_path: Path) -> None:
    """Verify that in-distribution samples yield drift_detected=False and record the run."""
    settings = get_settings().model_copy(deep=True)
    settings.drift.min_samples = 30
    settings.drift.window_size = 50
    settings.drift.tau_w1 = 0.50
    settings.drift.alpha = 0.01
    settings.drift.evidently_workspace_dir = str(tmp_path / "evidently_ws")

    rng = np.random.default_rng(42)
    ref = rng.standard_normal((60, 16)).astype(np.float32)

    store = BoundedInMemoryTelemetryStore(maxlen=100)
    for i in range(40):
        emb = rng.standard_normal(16).astype(np.float32)
        store.record(
            PredictionRecord(
                observation_id=f"obs_{i}",
                model_version="1.0.0",
                extractor="dinov3_vits16",
                top1_taxon=1,
                top1_confidence=0.95,
                conformal_set=[1],
                ood_energy=-3.2,
                is_ood=False,
                requires_review=False,
                latency_ms=4.2,
                embedding=emb,
            )
        )

    verdict = evaluate_recent_drift(
        store=store,
        settings=settings,
        reference_embeddings=ref,
    )

    assert verdict.status == "evaluated"
    assert verdict.drift_detected is False
    assert verdict.window_size == 40
    assert verdict.w1_distance >= 0.0
    assert verdict.mmd2 is not None
    assert verdict.mmd2_p_value is not None
    assert verdict.mmd2_p_value > 0.01

    # Verify run was recorded in store
    last_run = store.latest_drift_run()
    assert last_run is not None
    assert last_run.window_size == 40
    assert last_run.drift_detected is False
    assert np.isclose(last_run.w1_distance, verdict.w1_distance)


def test_evaluate_recent_drift_detected_under_shift(tmp_path: Path) -> None:
    """Verify that strong distribution displacement triggers drift_detected=True."""
    settings = get_settings().model_copy(deep=True)
    settings.drift.min_samples = 30
    settings.drift.window_size = 50
    settings.drift.tau_w1 = 0.10
    settings.drift.alpha = 0.01
    settings.drift.evidently_workspace_dir = str(tmp_path / "evidently_ws")

    rng = np.random.default_rng(42)
    ref = rng.standard_normal((60, 16)).astype(np.float32)

    store = BoundedInMemoryTelemetryStore(maxlen=100)
    # Inject large mean shift (+3.0)
    for i in range(35):
        emb = (rng.standard_normal(16) + 3.0).astype(np.float32)
        store.record(
            PredictionRecord(
                observation_id=f"obs_{i}",
                model_version="1.0.0",
                extractor="dinov3_vits16",
                top1_taxon=2,
                top1_confidence=0.45,
                conformal_set=[1, 2, 3],
                ood_energy=-0.5,
                is_ood=True,
                requires_review=True,
                latency_ms=8.1,
                embedding=emb,
            )
        )

    verdict = evaluate_recent_drift(
        store=store,
        settings=settings,
        reference_embeddings=ref,
    )

    assert verdict.status == "evaluated"
    assert verdict.drift_detected is True
    assert verdict.w1_distance > settings.drift.tau_w1

    last_run = store.latest_drift_run()
    assert last_run is not None
    assert last_run.drift_detected is True


def test_evaluate_recent_drift_extractor_filtering(tmp_path: Path) -> None:
    """Verify that extractor parameter filters live observations strictly by model family."""
    settings = get_settings().model_copy(deep=True)
    settings.drift.min_samples = 5
    settings.drift.window_size = 50
    settings.drift.evidently_workspace_dir = str(tmp_path / "evidently_ws")

    rng = np.random.default_rng(42)
    store = BoundedInMemoryTelemetryStore(maxlen=100)
    # Insert 6 records for dinov3 and 20 records for bioclip2
    for i in range(6):
        store.record(
            PredictionRecord(
                observation_id=f"dino_{i}",
                model_version="1.0.0",
                extractor="dinov3_vits16",
                top1_taxon=1,
                top1_confidence=0.9,
                conformal_set=[1],
                ood_energy=-3.0,
                is_ood=False,
                requires_review=False,
                latency_ms=5.0,
                embedding=rng.standard_normal(16).astype(np.float32),
            )
        )
    for i in range(20):
        store.record(
            PredictionRecord(
                observation_id=f"bioclip_{i}",
                model_version="1.0.0",
                extractor="bioclip2",
                top1_taxon=1,
                top1_confidence=0.9,
                conformal_set=[1],
                ood_energy=-3.0,
                is_ood=False,
                requires_review=False,
                latency_ms=5.0,
                embedding=rng.standard_normal(16).astype(np.float32),
            )
        )

    ref = rng.standard_normal((20, 16)).astype(np.float32)
    verdict = evaluate_recent_drift(
        store=store,
        settings=settings,
        extractor="dinov3_vits16",
        reference_embeddings=ref,
    )

    assert verdict.status == "evaluated"
    assert verdict.window_size == 6
