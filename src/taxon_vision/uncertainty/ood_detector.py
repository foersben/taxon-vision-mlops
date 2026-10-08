# SPDX-FileCopyrightText: 2026 Benjamin Förster
# SPDX-License-Identifier: MIT
"""Energy-based Out-Of-Distribution and empty image detector."""

from __future__ import annotations

import numpy as np


class EnergyOODDetector:
    """Detects uninformative, blurry, or non-organism images via Helmholtz free energy scoring.

    Why:
        Standard vision classifiers suffer from overconfidence when evaluated on non-biological noise, corrupt sensor transmissions, synthetic artifacts, or out-of-domain taxa. Classical Maximum Softmax Probability (MSP) detection fails because softmax normalization forces outputs onto a simplex where uniformly high uncalibrated logits still yield deceptive confidence spikes. Helmholtz free energy scoring (Liu et al., 2020) maps unnormalized logits directly to an energy surface that correlates monotonically with input density, providing a mathematically rigorous out-of-distribution detection boundary. In Phase 1, this class serves as intentional architectural scaffolding, prepared for calibrated integration into the epistemic gating pipeline once empirical noise benchmarks are set.

    How:
        Evaluates the negative Helmholtz free energy of the logit vector z at temperature T: E(x; T) = -T * log( sum_i exp(z_i / T) ). In-distribution observations exhibit low (highly negative) free energy values due to concentrated logit activations. Out-of-distribution inputs exhibit elevated (closer to zero or positive) energy. Observations whose calculated energy exceeds `energy_threshold` are rejected as OOD.

    References:
        Liu, W., Wang, X., Owens, J., & Li, Y. (2020). Energy-based Out-of-distribution Detection. Advances in Neural Information Processing Systems (NeurIPS 2020).

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
        """Compute the Helmholtz free energy score for a given logit vector.

        Why:
            Softmax normalization divides by the partition function, discarding the absolute magnitude of logit activations. Free energy preserves logit magnitude by computing the negative log-partition function directly. This formulation avoids softmax compression and yields superior separation between in-distribution taxa and out-of-domain noise.

        How:
            Applies numerically stable LogSumExp via max subtraction: max_val = max(z / T); E = -T * (max_val + log( sum_i exp(z_i / T - max_val) )). Preventing floating-point overflow during exponentiation.

        Args:
            logits: 1D NumPy array of unnormalized class logits z.
            temperature: Scaling hyperparameter T modulating the logit sharpness (default: 1.0).

        Returns:
            Scalar float representing the input's Helmholtz free energy score.
        """
        scaled = logits / temperature
        max_val = np.max(scaled)
        energy = -temperature * (max_val + np.log(np.sum(np.exp(scaled - max_val))))
        return float(energy)

    def is_ood(self, logits: np.ndarray) -> bool:
        """Determine whether the input logits correspond to an out-of-distribution sample.

        Why:
            Provides a binary epistemic safety decision boundary, preventing downstream automated processing or database ingestion for corrupted, irrelevant, or non-organism image captures.

        How:
            Computes scalar free energy E(x) and evaluates whether E(x) > energy_threshold. Higher energy indicates lower density relative to the training distribution.

        Args:
            logits: 1D array of unnormalized class logits.

        Returns:
            True if sample energy exceeds the threshold (flagged as OOD); False otherwise.
        """
        return self.compute_energy(logits) > self.energy_threshold
