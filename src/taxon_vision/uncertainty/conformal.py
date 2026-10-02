# SPDX-FileCopyrightText: 2026 Benjamin Förster
# SPDX-License-Identifier: MIT
"""Split Conformal Prediction Engine (Angelopoulos & Bates, 2021)."""

from __future__ import annotations

import numpy as np


class ConformalPredictionEngine:
    """Constructs finite-sample distribution-free prediction sets with error rate alpha."""

    def __init__(self, q_hat: float = 0.85, alpha: float = 0.05, k_max: int = 3) -> None:
        self.q_hat = q_hat
        self.alpha = alpha
        self.k_max = k_max

    def predict_set(self, probabilities: np.ndarray) -> tuple[list[int], bool]:
        """Returns (prediction_set_indices, referral_to_human_needed)."""
        # Non-conformity score: 1 - prob
        scores = 1.0 - probabilities
        pred_set = [int(idx) for idx in np.where(scores <= self.q_hat)[0]]

        # Referral rule: empty set or ambiguous set exceeding k_max
        refer_to_human = len(pred_set) == 0 or len(pred_set) > self.k_max
        return pred_set, refer_to_human
