# SPDX-FileCopyrightText: 2026 Benjamin Förster
# SPDX-License-Identifier: MIT
"""Class-Balanced Loss (Cui et al., 2019) for long-tail species distributions."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassBalancedLoss(nn.Module):
    """Cross-entropy loss weighted by the effective number of samples per class.

    Attributes:
        Various internal state and configuration variables used by the class.
    """

    def __init__(self, samples_per_class: list[int], beta: float = 0.999) -> None:
        """Init  .

        Args:
            samples_per_class: The samples per class parameter.
            beta: The beta parameter.
        """
        super().__init__()
        effective_num = 1.0 - np.power(beta, samples_per_class)
        weights = (1.0 - beta) / np.array(effective_num, dtype=np.float32)
        weights = weights / np.sum(weights) * len(samples_per_class)
        self.weights = torch.tensor(weights, dtype=torch.float32)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Forward.

        Args:
            logits: The logits parameter.
            targets: The targets parameter.

        Returns:
            The resulting value from the operation.
        """
        device_weights = self.weights.to(logits.device)
        return F.cross_entropy(logits, targets, weight=device_weights)
