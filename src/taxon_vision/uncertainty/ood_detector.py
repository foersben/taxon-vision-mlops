# SPDX-FileCopyrightText: 2026 Benjamin Förster
# SPDX-License-Identifier: MIT
"""Energy-based Out-Of-Distribution and empty image detector."""

from __future__ import annotations

import numpy as np


class EnergyOODDetector:
    """Detects uninformative, blurry, or non-organism images via energy scoring."""

    def __init__(self, energy_threshold: float = -12.5) -> None:
        self.energy_threshold = energy_threshold

    def compute_energy(self, logits: np.ndarray, temperature: float = 1.0) -> float:
        """Energy = -T * logsumexp(logits / T)."""
        scaled = logits / temperature
        max_val = np.max(scaled)
        energy = -temperature * (max_val + np.log(np.sum(np.exp(scaled - max_val))))
        return float(energy)

    def is_ood(self, logits: np.ndarray) -> bool:
        return self.compute_energy(logits) > self.energy_threshold
