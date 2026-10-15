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
