# SPDX-FileCopyrightText: 2026 Benjamin Förster
# SPDX-License-Identifier: MIT
"""Unit tests for out-of-distribution (OOD) energy score detection."""

from __future__ import annotations

import numpy as np

from taxon_vision.uncertainty.ood_detector import EnergyOODDetector


def test_ood_energy_detector() -> None:
    """Verify energy-based detector flags flat/noisy logits as out-of-distribution."""
    # Test with default calibrated threshold (-2.40)
    detector = EnergyOODDetector()
    assert detector.energy_threshold == -2.40

    # Strong in-distribution logits (e.g. 88.7% top-1 logit ~3.8, energy ~ -3.92)
    id_logits = np.array([3.8, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5])
    assert not detector.is_ood(id_logits)

    # Flat / noisy OOD logits (uniform noise ~0, energy ~ -2.30 > -2.40)
    ood_logits = np.zeros(10)
    assert detector.is_ood(ood_logits)

    # Custom threshold override
    custom_detector = EnergyOODDetector(energy_threshold=-10.0)
    assert not custom_detector.is_ood(np.array([12.0, 1.0, -2.0]))
    assert custom_detector.is_ood(np.array([0.1, -0.2, 0.0]))


def test_ood_entropy_and_calibration() -> None:
    """Verify predictive entropy detection and empirical threshold calibration."""
    detector = EnergyOODDetector(entropy_threshold=1.60)
    assert detector.entropy_threshold == 1.60

    # Low entropy: confident prediction (90% top-1)
    confident_probs = np.array([0.90, 0.05, 0.05])
    assert detector.compute_entropy(confident_probs) < 0.5
    assert not detector.is_high_entropy(confident_probs)

    # High entropy: diffuse prediction (e.g. 48.2% with 9 other classes sharing remainder)
    diffuse_probs = np.array([0.482] + [0.518 / 9] * 9)
    assert detector.compute_entropy(diffuse_probs) > 1.60
    assert detector.is_high_entropy(diffuse_probs)

    # Threshold calibration
    synthetic_val_logits = np.random.RandomState(42).randn(100, 10) + 2.0
    calibrated = detector.calibrate_threshold(synthetic_val_logits, percentile=95.0)
    assert isinstance(calibrated, float)
    assert detector.energy_threshold == calibrated

    stateless_calib = EnergyOODDetector.calibrate_threshold_from_logits(synthetic_val_logits, percentile=95.0)
    assert np.isclose(stateless_calib, calibrated)
