# SPDX-FileCopyrightText: 2026 Benjamin Förster
# SPDX-License-Identifier: MIT
"""Energy-based Out-Of-Distribution and empty image detector."""

from __future__ import annotations

import numpy as np


class EnergyOODDetector:
    """Detects uninformative, blurry, or non-organism images via energy scoring.

    Attributes:
        energy_threshold: Decision boundary threshold above which inputs are classified as OOD.
    """

    def __init__(self, energy_threshold: float = -12.5) -> None:
        """Initialize the energy-based out-of-distribution detector.

        Args:
            energy_threshold: Decision boundary threshold above which inputs are classified as OOD.
        """
        self.energy_threshold = energy_threshold

    def compute_energy(self, logits: np.ndarray, temperature: float = 1.0) -> float:
        """Compute the energy of the logits.

        The energy of the logits is defined as: E = -T * logsumexp(logits / T)
        where T is the temperature.

        Args:
            logits: The logits of the model.
            temperature: The temperature to use for the computation.

        Returns:
            The energy of the logits.
        """
        scaled = logits / temperature
        max_val = np.max(scaled)
        energy = -temperature * (max_val + np.log(np.sum(np.exp(scaled - max_val))))
        return float(energy)

    def is_ood(self, logits: np.ndarray) -> bool:
        """Determine whether the input logits correspond to an out-of-distribution sample.

        Args:
            logits: 1D array of unnormalized class logits.

        Returns:
            True if the sample energy exceeds the threshold; False otherwise.
        """
        return self.compute_energy(logits) > self.energy_threshold
