# SPDX-FileCopyrightText: 2026 Benjamin Förster
# SPDX-License-Identifier: MIT
"""Unit tests for monitoring and telemetry components."""

from __future__ import annotations

import numpy as np

from taxon_vision.monitoring.drift import compute_wasserstein_drift


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
