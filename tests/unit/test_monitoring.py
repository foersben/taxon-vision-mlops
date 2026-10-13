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
