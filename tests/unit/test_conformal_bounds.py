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


def test_determine_routing_tier_contract() -> None:
    """Verify 4-tier conformal routing contract: DECISIVE, MODERATE, SEVERE, ABSTENTION."""
    from taxon_vision.service.inference import determine_routing_tier
    from taxon_vision.service.schemas import ConformalRoutingTier

    # 1. DECISIVE: |C| = 1 and pi_max >= tau_conf
    assert (
        determine_routing_tier(cardinality=1, top_confidence=0.85, min_confidence_threshold=0.60)
        == ConformalRoutingTier.DECISIVE
    )

    # 2. MODERATE_AMBIGUITY: 1 < |C| <= 3
    assert (
        determine_routing_tier(cardinality=2, top_confidence=0.55, min_confidence_threshold=0.60)
        == ConformalRoutingTier.MODERATE_AMBIGUITY
    )
    assert (
        determine_routing_tier(cardinality=3, top_confidence=0.40, min_confidence_threshold=0.60)
        == ConformalRoutingTier.MODERATE_AMBIGUITY
    )

    # 3. SEVERE_AMBIGUITY: |C| > 3 OR (|C| = 1 and pi_max < tau_conf)
    assert (
        determine_routing_tier(cardinality=4, top_confidence=0.30, min_confidence_threshold=0.60)
        == ConformalRoutingTier.SEVERE_AMBIGUITY
    )
    assert (
        determine_routing_tier(cardinality=1, top_confidence=0.48, min_confidence_threshold=0.60)
        == ConformalRoutingTier.SEVERE_AMBIGUITY
    )

    # 4. ABSTENTION: |C| = 0
    assert (
        determine_routing_tier(cardinality=0, top_confidence=0.10, min_confidence_threshold=0.60)
        == ConformalRoutingTier.ABSTENTION
    )
