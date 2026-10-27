# SPDX-FileCopyrightText: 2026 Benjamin Förster
# SPDX-License-Identifier: MIT
"""Energy-based Out-Of-Distribution and empty image detector."""

from __future__ import annotations

import numpy as np

from taxon_vision.config import get_settings


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
        temperature: Scaling hyperparameter T modulating the logit sharpness.
        entropy_threshold: Shannon predictive entropy threshold for uncertainty triage.
    """

    def __init__(
        self,
        energy_threshold: float | None = None,
        temperature: float | None = None,
        entropy_threshold: float | None = None,
    ) -> None:
        """Initialize the energy-based out-of-distribution detector.

        Args:
            energy_threshold: Decision boundary threshold above which inputs are classified as OOD. Defaults to settings.model.ood_energy_threshold.
            temperature: Softmax logit scaling temperature for probability calibration. Defaults to settings.model.ood_temperature.
            entropy_threshold: Shannon entropy threshold for uncertainty triage. Defaults to settings.ood.entropy_threshold.
        """
        settings = get_settings()
        self.energy_threshold = (
            energy_threshold if energy_threshold is not None else settings.model.ood_energy_threshold
        )
        self.temperature = temperature if temperature is not None else settings.model.ood_temperature
        self.entropy_threshold = entropy_threshold if entropy_threshold is not None else settings.ood.entropy_threshold

    def compute_energy(self, logits: np.ndarray, temperature: float | None = None) -> float:
        """Compute the Helmholtz free energy score for a given logit vector.

        Why:
            Softmax normalization divides by the partition function, discarding the absolute magnitude of logit activations. Free energy preserves logit magnitude by computing the negative log-partition function directly. This formulation avoids softmax compression and yields superior separation between in-distribution taxa and out-of-domain noise.

        How:
            Applies numerically stable LogSumExp via max subtraction: max_val = max(z / T); E = -T * (max_val + log( sum_i exp(z_i / T - max_val) )). Preventing floating-point overflow during exponentiation.

        Args:
            logits: 1D NumPy array of unnormalized class logits z.
            temperature: Scaling hyperparameter T modulating the logit sharpness (defaults to self.temperature).

        Returns:
            Scalar float representing the input's Helmholtz free energy score.
        """
        temp = self.temperature if temperature is None else temperature
        scaled = logits / temp
        max_val = np.max(scaled)
        energy = -temp * (max_val + np.log(np.sum(np.exp(scaled - max_val))))
        return float(energy)

    @staticmethod
    def compute_entropy(probabilities: np.ndarray) -> float:
        """Compute Shannon predictive entropy of class probability distribution.

        Args:
            probabilities: 1D array of class probabilities summing to 1.0.

        Returns:
            Scalar Shannon entropy in nats.
        """
        eps = 1e-12
        probs = np.clip(probabilities, eps, 1.0)
        return float(-np.sum(probs * np.log(probs)))

    def is_high_entropy(self, probabilities: np.ndarray, threshold: float | None = None) -> bool:
        """Evaluate whether predicted probabilities exhibit excessive predictive entropy.

        Args:
            probabilities: 1D array of class probabilities summing to 1.0.
            threshold: Optional threshold override (defaults to self.entropy_threshold).

        Returns:
            True if predictive entropy exceeds the threshold; False otherwise.
        """
        thresh = self.entropy_threshold if threshold is None else threshold
        return self.compute_entropy(probabilities) > thresh

    def is_ood(
        self,
        logits: np.ndarray,
        threshold: float | None = None,
        temperature: float | None = None,
    ) -> bool:
        """Determine whether the input logits correspond to an out-of-distribution sample.

        Why:
            Provides a binary epistemic safety decision boundary, preventing downstream automated processing or database ingestion for corrupted, irrelevant, or non-organism image captures.

        How:
            Computes scalar free energy E(x) and evaluates whether E(x) > energy_threshold. Higher energy indicates lower density relative to the training distribution.

        Args:
            logits: 1D array of unnormalized class logits.
            threshold: Optional energy threshold override (defaults to self.energy_threshold).
            temperature: Optional temperature override.

        Returns:
            True if sample energy exceeds the threshold (flagged as OOD); False otherwise.
        """
        thresh = self.energy_threshold if threshold is None else threshold
        return self.compute_energy(logits, temperature=temperature) > thresh

    def calibrate_threshold(
        self,
        val_logits: np.ndarray,
        percentile: float = 95.0,
        temperature: float | None = None,
    ) -> float:
        """Calibrate energy threshold empirically from in-distribution validation logits.

        Args:
            val_logits: 2D array of validation logits with shape (N, num_classes).
            percentile: Percentile of in-distribution energies (e.g. 95.0 for 5% ID rejection rate).
            temperature: Optional energy temperature scaling parameter.

        Returns:
            Calibrated scalar energy threshold.
        """
        temp = self.temperature if temperature is None else temperature
        scaled = val_logits / temp
        max_val = np.max(scaled, axis=1, keepdims=True)
        energies = -temp * (max_val.squeeze(1) + np.log(np.sum(np.exp(scaled - max_val), axis=1)))
        self.energy_threshold = float(np.percentile(energies, percentile))
        return self.energy_threshold

    @classmethod
    def calibrate_threshold_from_logits(
        cls,
        val_logits: np.ndarray,
        percentile: float = 95.0,
        temperature: float = 1.0,
    ) -> float:
        """Stateless class method to compute empirical energy threshold from validation logits.

        Args:
            val_logits: 2D array of validation logits with shape (N, num_classes).
            percentile: Percentile of in-distribution energies (e.g. 95.0 for 5% ID rejection rate).
            temperature: Energy temperature scaling parameter.

        Returns:
            Calibrated scalar energy threshold.
        """
        scaled = val_logits / temperature
        max_val = np.max(scaled, axis=1, keepdims=True)
        energies = -temperature * (max_val.squeeze(1) + np.log(np.sum(np.exp(scaled - max_val), axis=1)))
        return float(np.percentile(energies, percentile))
