# SPDX-FileCopyrightText: 2026 Benjamin Förster
# SPDX-License-Identifier: MIT
"""Unit tests for Split Conformal Prediction set generation and triage referrals."""

from __future__ import annotations

import numpy as np

from taxon_vision.uncertainty.conformal import ConformalPredictionEngine


def test_conformal_prediction_set() -> None:
    """Verify conformal set construction and triage referral trigger on set cardinality."""
    engine = ConformalPredictionEngine(q_hat=0.8, alpha=0.05, k_max=2)
    # High confidence on class 0 (prob = 0.9 -> score = 0.1 <= 0.8)
    probs_confident = np.array([0.9, 0.05, 0.05])
    pred_set, refer = engine.predict_set(probs_confident)
    assert pred_set == [0]
    assert not refer

    # Ambiguous predictions (probabilities diffuse)
    probs_ambiguous = np.array([0.34, 0.33, 0.33])  # scores ~ 0.66 <= 0.8 -> all 3 admitted
    pred_set, refer = engine.predict_set(probs_ambiguous)
    assert len(pred_set) == 3
    assert refer  # len > k_max (2)

    # Low-confidence singleton prediction set (e.g. 48.2% < 60% threshold, other classes < 20%)
    probs_low_conf = np.array([0.482, 0.18, 0.18, 0.158])
    pred_set_low, refer_low = engine.predict_set(probs_low_conf)
    assert pred_set_low == [0]
    assert refer_low  # Flagged for human review due to min_confidence_threshold
