# SPDX-FileCopyrightText: 2026 Benjamin Förster
# SPDX-License-Identifier: MIT
"""Unit tests for monitoring, representation drift, and MMD hypothesis testing."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import ortho_group

from taxon_vision.monitoring.drift import (
    MMDResult,
    compute_mmd2_permutation,
    compute_wasserstein_drift,
)


def test_compute_wasserstein_drift() -> None:
    """Verify 1D Wasserstein approximation computes positive drift distance under shift."""
    ref = np.random.randn(100, 16)
    curr = np.random.randn(100, 16) + 2.0
    drift = compute_wasserstein_drift(ref, curr)
    assert drift > 0.0


def test_compute_wasserstein_drift_unnormalized_vs_normalized_paradox() -> None:
    """Verify that unnormalized activations detect magnitude shift, whereas unit vectors collapse to 0."""
    rng = np.random.default_rng(42)
    # Reference activations centered with scale 1.0, current activations shifted with scale 3.0
    ref_raw = rng.normal(loc=0.0, scale=1.0, size=(100, 32))
    curr_raw = rng.normal(loc=0.0, scale=3.0, size=(100, 32))

    # Pre-normalized activations preserve norm differences
    drift_unnorm = compute_wasserstein_drift(ref_raw, curr_raw)
    assert drift_unnorm > 1.0

    # If vectors were unit-normalized prior to drift calculation (the L2-norm paradox):
    ref_unit = ref_raw / np.linalg.norm(ref_raw, axis=-1, keepdims=True)
    curr_unit = curr_raw / np.linalg.norm(curr_raw, axis=-1, keepdims=True)
    drift_unit = compute_wasserstein_drift(ref_unit, curr_unit)

    # Unit vectors collapse to static 1.0 vs 1.0 comparison (drift ~ 0.0)
    assert np.isclose(drift_unit, 0.0, atol=1e-6)


def test_compute_mmd2_same_distribution() -> None:
    """Verify that independent draws from the same distribution yield p-value > 0.05."""
    rng = np.random.default_rng(42)
    ref = rng.standard_normal(size=(120, 16))
    curr = rng.standard_normal(size=(120, 16))

    result = compute_mmd2_permutation(ref, curr, max_samples=200, n_permutations=200, seed=42)
    assert isinstance(result, MMDResult)
    assert result.p_value > 0.05
    assert result.n_reference == 120
    assert result.n_current == 120
    assert result.bandwidth > 0.0


def test_compute_mmd2_mean_shift() -> None:
    """Verify that multivariate mean displacement triggers rejection of null hypothesis with p < 0.01."""
    rng = np.random.default_rng(42)
    ref = rng.standard_normal(size=(120, 16))
    curr = rng.standard_normal(size=(120, 16)) + 0.5

    result = compute_mmd2_permutation(ref, curr, max_samples=200, n_permutations=200, seed=42)
    assert result.p_value < 0.01
    assert result.mmd2 > 0.0


def test_compute_mmd2_rotation_blind_spot() -> None:
    """Verify that orthogonal manifold rotations are detected by MMD while invisible to 1D norm W1."""
    rng = np.random.default_rng(42)
    # Construct anisotropic covariance matrix (elongated in first dimension)
    cov = np.diag([10.0, 1.0, 0.1] + [1.0] * 13)
    ref = rng.multivariate_normal(np.zeros(16), cov, size=150)

    # Apply random orthogonal rotation matrix R in O(16)
    rot_matrix = ortho_group.rvs(16, random_state=42)
    curr_rotated = ref @ rot_matrix.T

    # Tier 1: Norms are invariant under orthogonal rotation (||Rx||_2 == ||x||_2)
    w1_dist = compute_wasserstein_drift(ref, curr_rotated)
    assert np.isclose(w1_dist, 0.0, atol=1e-6)

    # Tier 2: MMD detects the angular distribution displacement
    mmd_res = compute_mmd2_permutation(ref, curr_rotated, max_samples=200, n_permutations=200, seed=42)
    assert mmd_res.p_value < 0.01
    assert mmd_res.mmd2 > 0.0


def test_compute_mmd2_deterministic_seed() -> None:
    """Verify that identical inputs and seeds yield identical bitwise MMD results."""
    rng = np.random.default_rng(42)
    ref = rng.standard_normal(size=(80, 8))
    curr = rng.standard_normal(size=(80, 8)) + 0.2

    res1 = compute_mmd2_permutation(ref, curr, max_samples=50, n_permutations=100, seed=123)
    res2 = compute_mmd2_permutation(ref, curr, max_samples=50, n_permutations=100, seed=123)

    assert res1.mmd2 == res2.mmd2
    assert res1.p_value == res2.p_value
    assert res1.bandwidth == res2.bandwidth


def test_compute_mmd2_input_validation() -> None:
    """Verify that populations with fewer than 2 samples raise ValueError."""
    ref = np.ones((1, 8))
    curr = np.ones((10, 8))

    with pytest.raises(ValueError, match="at least two samples per group"):
        compute_mmd2_permutation(ref, curr)

    with pytest.raises(ValueError, match="at least two samples per group"):
        compute_mmd2_permutation(curr, ref)


def test_generate_evidently_drift_report_persists_workspace(tmp_path) -> None:
    """Persist Evidently snapshots into the configured UI workspace alongside standalone HTML output."""
    import numpy as np

    from taxon_vision.monitoring import drift_runner

    rng = np.random.default_rng(42)
    ref = rng.normal(0.0, 1.0, size=(30, 8)).astype(np.float32)
    cur = rng.normal(0.2, 1.0, size=(25, 8)).astype(np.float32)

    ws_dir = tmp_path / "evidently_ws"
    html_out = tmp_path / "reports" / "drift.html"
    drift_runner.generate_evidently_drift_report(
        ref,
        cur,
        html_out,
        workspace_dir=ws_dir,
        project_name="TaxonVision Drift Monitor",
    )

    assert html_out.is_file()
    assert ws_dir.is_dir()
    assert any(ws_dir.rglob("*.json"))


def test_structured_stdout_logging_handler_idempotent(monkeypatch: pytest.MonkeyPatch) -> None:
    """Configure stdout handlers with StructuredJsonFormatter when TAXON_STRUCTURED_STDOUT=true without duplicates."""
    import logging

    from taxon_vision.service.structured_logging import StructuredJsonFormatter, _ensure_structured_stdout_handler

    monkeypatch.setenv("TAXON_STRUCTURED_STDOUT", "true")
    isolated_logger = logging.getLogger("taxon_vision.test_stdout_isolation")
    isolated_logger.handlers.clear()
    _ensure_structured_stdout_handler(isolated_logger)
    _ensure_structured_stdout_handler(isolated_logger)
    assert len(isolated_logger.handlers) == 1
    assert isinstance(isolated_logger.handlers[0].formatter, StructuredJsonFormatter)
