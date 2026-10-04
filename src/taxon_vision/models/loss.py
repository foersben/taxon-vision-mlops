# SPDX-FileCopyrightText: 2026 Benjamin Förster
# SPDX-License-Identifier: MIT
r"""Class-Balanced Loss (Cui et al., 2019) for long-tailed species distributions.

This module implements the Class-Balanced Loss proposed in Cui et al. (CVPR 2019):
"Class-Balanced Loss Based on Effective Number of Samples". Biological observations
naturally follow an extreme power-law (Zipfian) distribution, where a small subset
of common species accounts for the majority of sightings while thousands of rare
species possess minimal observations.

Rather than naive inverse frequency weighting (which causes severe gradient variance
on rare classes), Class-Balanced Loss introduces the concept of 'effective number of
samples' $E_n = (1 - \beta^n) / (1 - \beta)$, where $\beta \in [0, 1)$ captures
data overlap in feature space.

Typical usage example:
    samples_per_class = [1200, 450, 30, 5]
    criterion = ClassBalancedLoss(samples_per_class, beta=0.999)
    loss = criterion(logits, targets)
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassBalancedLoss(nn.Module):
    r"""Loss function dynamically weighted by the effective number of samples per class.

    Given $N_i$ training examples for class $i$, the effective number of samples is:
    $$E_{N_i} = \frac{1 - \beta^{N_i}}{1 - \beta}$$

    The corresponding class weight is inversely proportional to the effective volume:
    $$W_i = \frac{1 - \beta}{1 - \beta^{N_i}}$$
    normalized such that $\sum_{i=1}^K W_i = K$ where $K$ is the number of classes.

    Attributes:
        beta: Hyperparameter controlling the scale of effective samples,
            typically chosen in $[0.9, 0.9999]$. When $\beta \to 1$, weights approach
            inverse class frequency. When $\beta = 0$, weights collapse to standard
            unweighted cross-entropy.
        weights: 1D tensor of pre-calculated normalized class weights of shape `(num_classes,)`.
    """

    def __init__(
        self,
        samples_per_class: Sequence[int],
        beta: float = 0.999,
        epsilon: float = 1e-8,
    ) -> None:
        """Initialize the ClassBalancedLoss module.

        Args:
            samples_per_class: Number of training examples per class. Length corresponds
                to the total number of classes $K$.
            beta: Overlap hyperparameter in $[0.0, 1.0)$.
            epsilon: Small constant to prevent division by zero for classes with zero observations.

        Raises:
            ValueError: If `samples_per_class` is empty or if `beta` is outside $[0.0, 1.0)$.
        """
        super().__init__()
        if not samples_per_class:
            raise ValueError("samples_per_class cannot be empty.")
        if not (0.0 <= beta < 1.0):
            raise ValueError(f"beta must be in [0.0, 1.0), got {beta}.")

        self.beta = beta
        num_classes = len(samples_per_class)

        # Convert counts to float array and clamp minimum sample count to 1 to avoid zero-division
        counts = np.maximum(np.array(samples_per_class, dtype=np.float64), 1.0)

        # Calculate effective number of samples: (1 - beta^n) / (1 - beta)
        effective_num = 1.0 - np.power(self.beta, counts)
        # Class weights: (1 - beta) / effective_num
        weights = (1.0 - self.beta) / (effective_num + epsilon)

        # Normalize weights so that the sum equals num_classes (mean weight is 1.0)
        norm_factor = np.sum(weights)
        if norm_factor > 0:
            weights = (weights / norm_factor) * num_classes

        self.weights: torch.Tensor
        self.register_buffer("weights", torch.tensor(weights, dtype=torch.float32))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute the class-balanced cross-entropy loss.

        Args:
            logits: Unnormalized class predictions of shape `(batch_size, num_classes)`.
            targets: Ground-truth class labels of shape `(batch_size,)`.

        Returns:
            Scalar loss tensor averaged over the batch.
        """
        device_weights = self.weights.to(logits.device)
        return F.cross_entropy(logits, targets, weight=device_weights)
