"""Test ood energy.py.

This module provides functionality related to test_ood_energy.
"""

import numpy as np

from taxon_vision.uncertainty.ood_detector import EnergyOODDetector


def test_ood_energy_detector() -> None:
    """Test ood energy detector."""
    detector = EnergyOODDetector(energy_threshold=-10.0)
    # Strong in-distribution logits
    id_logits = np.array([12.0, 1.0, -2.0])
    assert not detector.is_ood(id_logits)

    # Flat / noisy OOD logits
    ood_logits = np.array([0.1, -0.2, 0.0])
    assert detector.is_ood(ood_logits)
