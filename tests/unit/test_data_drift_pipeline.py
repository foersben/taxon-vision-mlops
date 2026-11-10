# SPDX-FileCopyrightText: 2026 Benjamin Förster
# SPDX-License-Identifier: MIT
"""Unit tests for Stage 3.5 training data drift pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from taxon_vision.pipelines.data_drift import (
    DataDriftPipelineConfig,
    evaluate_training_data_drift,
    main,
)


@pytest.fixture
def synthetic_embedding_files(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Generate temporary baseline and drifted embedding tensor checkpoints."""
    ref_path = tmp_path / "ref_embeddings.pt"
    identical_path = tmp_path / "identical_embeddings.pt"
    drifted_path = tmp_path / "drifted_embeddings.pt"

    np.random.seed(42)
    # 60 samples, 16 features
    ref_arr = np.random.randn(60, 16).astype(np.float32)
    drifted_arr = (ref_arr + 10.0).astype(np.float32)

    torch.save({"embeddings": torch.from_numpy(ref_arr)}, ref_path)
    torch.save({"embeddings": torch.from_numpy(ref_arr.copy())}, identical_path)
    torch.save({"embeddings": torch.from_numpy(drifted_arr)}, drifted_path)

    return ref_path, identical_path, drifted_path


def test_evaluate_training_data_drift_identical(
    synthetic_embedding_files: tuple[Path, Path, Path],
    tmp_path: Path,
) -> None:
    """Verify that identical datasets trigger zero drift and pass hypothesis tests."""
    ref_path, identical_path, _ = synthetic_embedding_files
    metrics_out = tmp_path / "data_drift.json"
    report_out = tmp_path / "data_drift_report.html"

    config = DataDriftPipelineConfig(
        reference_path=ref_path,
        current_path=identical_path,
        metrics_out_path=metrics_out,
        report_out_path=report_out,
        tau_w1=0.15,
        alpha=0.01,
        max_samples=50,
        permutations=50,
        seed=42,
    )

    result = evaluate_training_data_drift(config)

    assert not result.drift_detected
    assert result.w1_distance == pytest.approx(0.0, abs=1e-5)
    assert result.mmd2_p_value >= 0.05
    assert result.n_reference == 60
    assert result.n_current == 60
    assert result.feature_dim == 16

    assert metrics_out.is_file()
    with metrics_out.open("r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["drift_detected"] is False
    assert data["w1_distance"] == pytest.approx(0.0, abs=1e-5)
    assert report_out.is_file()


def test_evaluate_training_data_drift_detected(
    synthetic_embedding_files: tuple[Path, Path, Path],
    tmp_path: Path,
) -> None:
    """Verify that shifted datasets trigger drift alert."""
    ref_path, _, drifted_path = synthetic_embedding_files
    metrics_out = tmp_path / "data_drift.json"
    report_out = tmp_path / "data_drift_report.html"

    config = DataDriftPipelineConfig(
        reference_path=ref_path,
        current_path=drifted_path,
        metrics_out_path=metrics_out,
        report_out_path=report_out,
        tau_w1=0.15,
        alpha=0.01,
        max_samples=50,
        permutations=50,
        seed=42,
    )

    result = evaluate_training_data_drift(config)

    assert result.drift_detected
    assert result.w1_distance > config.tau_w1

    with metrics_out.open("r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["drift_detected"] is True


def test_evaluate_training_data_drift_dimension_mismatch(tmp_path: Path) -> None:
    """Verify ValueError is raised when embedding dimensions mismatch."""
    ref_path = tmp_path / "ref.pt"
    curr_path = tmp_path / "curr.pt"

    torch.save({"embeddings": torch.randn(20, 16)}, ref_path)
    torch.save({"embeddings": torch.randn(20, 32)}, curr_path)

    config = DataDriftPipelineConfig(
        reference_path=ref_path,
        current_path=curr_path,
        metrics_out_path=tmp_path / "metrics.json",
        report_out_path=tmp_path / "report.html",
        tau_w1=0.15,
        alpha=0.01,
    )

    with pytest.raises(ValueError, match="Feature dimension mismatch"):
        evaluate_training_data_drift(config)


def test_data_drift_cli_main(
    synthetic_embedding_files: tuple[Path, Path, Path],
    tmp_path: Path,
) -> None:
    """Verify CLI main entrypoint executes and returns exit code 0."""
    ref_path, identical_path, _ = synthetic_embedding_files
    metrics_out = tmp_path / "cli_metrics.json"
    report_out = tmp_path / "cli_report.html"

    argv = [
        "--reference",
        str(ref_path),
        "--current",
        str(identical_path),
        "--metrics-out",
        str(metrics_out),
        "--report-out",
        str(report_out),
        "--tau-w1",
        "0.20",
        "--alpha",
        "0.05",
    ]
    exit_code = main(argv)
    assert exit_code == 0
    assert metrics_out.is_file()
    assert report_out.is_file()
