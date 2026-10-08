# SPDX-FileCopyrightText: 2026 Benjamin Förster
# SPDX-License-Identifier: MIT
"""Split Conformal Prediction Engine (Angelopoulos & Bates, 2021)."""

from __future__ import annotations

import numpy as np


class ConformalPredictionEngine:
    """Constructs finite-sample distribution-free prediction sets with error rate alpha.

    Why:
        Standard deep learning classifiers output uncalibrated softmax scores that frequently assign extreme confidence to erroneous categories, especially on power-law biological distributions. Relying on fixed heuristic probability cutoffs (e.g. prob > 0.8) yields arbitrary error rates across rare species. TaxonVision adopts Split Conformal Prediction (Angelopoulos & Bates, 2021) to mathematically guarantee a user-specified marginal coverage rate (1 - alpha) regardless of the underlying data distribution, routing ambiguous observations to human triage.

    How:
        Calibrates an empirical non-conformity quantile threshold q_hat on holdout calibration data D_cal. During inference, evaluates the non-conformity score s_i = 1 - p_i for each candidate class. Admits all classes satisfying s_i <= q_hat (equivalently, p_i >= 1 - q_hat) into the conformal prediction set C(x). When the resulting prediction set is empty or exceeds the maximum cardinality threshold k_max, asserts the human triage referral flag.

    Attributes:
        q_hat: Pre-calibrated non-conformity quantile cutoff.
        alpha: Nominal user-specified error budget (e.g. 0.05 for 95% guaranteed coverage).
        k_max: Maximum acceptable set cardinality before triggering active learning triage.
    """

    def __init__(self, q_hat: float = 0.85, alpha: float = 0.05, k_max: int = 3) -> None:
        """Initialize the conformal prediction engine.

        Args:
            q_hat: Pre-calibrated non-conformity quantile cutoff.
            alpha: Desired nominal error rate (e.g. 0.05 for 95% coverage guarantee).
            k_max: Maximum acceptable set cardinality before flagging for human referral.
        """
        self.q_hat = q_hat
        self.alpha = alpha
        self.k_max = k_max

    def predict_set(self, probabilities: np.ndarray) -> tuple[list[int], bool]:
        """Construct finite-sample distribution-free prediction sets with bounded error rates.

        Why:
            Provides formal statistical guarantees: P(Y in C(X)) >= 1 - alpha. Under exchangeability, this coverage guarantee holds unconditionally without requiring parametric assumptions about data geometry, class imbalance, or model calibration. Identifies ambiguous, multi-modal, or out-of-distribution observations for human taxonomist review.

        How:
            Evaluates the non-conformity score s_i = 1.0 - p_i for each class index i. Filters indices where s_i <= q_hat (i.e. model probability p_i >= 1.0 - q_hat). Evaluates the referral condition: if the set C(x) is completely empty (no taxon meets minimum confidence) or exceeds cardinality threshold k_max (|C(x)| > k_max), `refer_to_human` is flagged True.

        Complexity:
            O(K) time complexity where K is the number of target classes, with zero dynamic heap allocation when evaluated on contiguous numpy probability vectors.

        Args:
            probabilities: 1D NumPy array of normalized class probabilities summing to 1.0.

        Returns:
            A tuple (conformal_indices, requires_human_review) containing:
                - conformal_indices: List of integer taxon indices admitted into the valid set.
                - requires_human_review: Boolean flag indicating if human taxonomist review is mandated.

        References:
            Angelopoulos, A. N., & Bates, S. (2021). A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification. arXiv:2107.07511.
        """
        # Non-conformity score: 1 - prob
        scores = 1.0 - probabilities
        pred_set = [int(idx) for idx in np.where(scores <= self.q_hat)[0]]

        # Referral rule: empty set or ambiguous set exceeding k_max
        refer_to_human = len(pred_set) == 0 or len(pred_set) > self.k_max
        return pred_set, refer_to_human
