# SPDX-FileCopyrightText: 2026 Benjamin Förster
# SPDX-License-Identifier: MIT
"""Unit tests for the Tier 1 drift threshold calibration pipeline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from taxon_vision.pipelines.calibrate_drift import (
    _compute_file_sha256,
    _load_embedding_norms,
    calibrate_drift_threshold,
    main as calibrate_drift_main,
)


@pytest.fixture
def sample_embedding_partitions(tmp_path: Path) -> tuple[Path, Path]:
    """Create synthetic reference and test embedding partitions on disk."""
    rng = np.random.default_rng(42)
    cal_embs = torch.tensor(rng.standard_normal((50, 16)), dtype=torch.float32)
    test_embs = torch.tensor(rng.standard_normal((50, 16)), dtype=torch.float32)

    cal_path = tmp_path / "cal_embeddings.pt"
    test_path = tmp_path / "test_embeddings.pt"

    torch.save({"embeddings": cal_embs, "labels": torch.zeros(50)}, cal_path)
    torch.save({"embeddings": test_embs, "labels": torch.zeros(50)}, test_path)

    return cal_path, test_path


def test_calibrate_drift_threshold_success(sample_embedding_partitions: tuple[Path, Path]) -> None:
    """Verify that calibrate_drift_threshold produces a valid drift configuration payload."""
    cal_path, test_path = sample_embedding_partitions
    payload = calibrate_drift_threshold(
        cal_path,
        test_path,
        window_size=30,
        n_bootstraps=100,
        percentile=99.0,
        alpha=0.01,
        seed=42,
    )

    assert "drift" in payload
    drift = payload["drift"]
    assert drift["tau_w1"] > 0.0
    assert drift["alpha"] == 0.01
    assert drift["window_size"] == 30
    assert drift["n_bootstraps"] == 100
    assert drift["percentile"] == 99.0
    assert len(drift["reference_sha256"]) == 64
    assert drift["null_median_w1"] <= drift["tau_w1"]


def test_calibrate_drift_missing_file_raises(tmp_path: Path) -> None:
    """Verify that non-existent paths raise FileNotFoundError."""
    missing = tmp_path / "does_not_exist.pt"
    valid = tmp_path / "valid.pt"
    torch.save(torch.randn(10, 8), valid)

    with pytest.raises(FileNotFoundError):
        _load_embedding_norms(missing)

    with pytest.raises(FileNotFoundError):
        calibrate_drift_threshold(missing, valid)


def test_calibrate_drift_empty_partition_raises(tmp_path: Path) -> None:
    """Verify that embedding files with 0 rows raise ValueError."""
    empty_path = tmp_path / "empty.pt"
    torch.save(torch.empty((0, 16)), empty_path)

    with pytest.raises(ValueError, match="contains zero records"):
        _load_embedding_norms(empty_path)


def test_compute_file_sha256(tmp_path: Path) -> None:
    """Verify deterministic cryptographic SHA256 file hashing."""
    test_file = tmp_path / "test.txt"
    test_file.write_text("TaxonVision MLOps", encoding="utf-8")

    digest1 = _compute_file_sha256(test_file)
    digest2 = _compute_file_sha256(test_file)
    assert digest1 == digest2
    assert len(digest1) == 64


def test_calibrate_drift_cli_main(sample_embedding_partitions: tuple[Path, Path], tmp_path: Path) -> None:
    """Verify that the CLI main entrypoint writes a valid YAML file and exits with 0."""
    cal_path, test_path = sample_embedding_partitions
    out_yaml = tmp_path / "drift_config.yaml"

    exit_code = calibrate_drift_main(
        [
            "--cal-path",
            str(cal_path),
            "--test-path",
            str(test_path),
            "--output-yaml",
            str(out_yaml),
            "--window-size",
            "20",
            "--n-bootstraps",
            "50",
            "--percentile",
            "95.0",
        ]
    )

    assert exit_code == 0
    assert out_yaml.exists()

    with out_yaml.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    assert "drift" in data
    assert data["drift"]["tau_w1"] > 0.0
    assert data["drift"]["window_size"] == 20
