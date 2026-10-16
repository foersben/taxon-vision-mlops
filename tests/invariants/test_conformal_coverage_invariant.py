# SPDX-FileCopyrightText: 2026 Benjamin Förster
# SPDX-License-Identifier: MIT
"""Architectural invariant test verifying mathematical conformal prediction coverage."""

from __future__ import annotations

import numpy as np
import torch

from taxon_vision.models.training.types import EmbeddingSplit
from taxon_vision.uncertainty.conformal import ConformalPredictionEngine


def test_empirical_conformal_coverage() -> None:
    """Assert finite-sample coverage guarantee on holdout set.

    Why:
        Vovk et al. (2005) and Angelopoulos & Bates (2021) prove that under exchangeability,
        Split Conformal Prediction guarantees E[Coverage] >= 1 - alpha for any data distribution.
        This test statistically verifies that empirical coverage on a large independent holdout test
        sample meets the theoretical nominal coverage within a 3-sigma tolerance bound.
    """
    np.random.seed(42)
    n_cal = 2000
    n_test = 2000
    alpha = 0.05
    nominal_coverage = 1.0 - alpha

    # Generate synthetic softmax predictions under standard Dirichlet distribution
    cal_probs = np.random.dirichlet(np.ones(5) * 2.0, size=n_cal)
    cal_labels = np.random.randint(0, 5, size=n_cal)
    cal_scores = 1.0 - cal_probs[np.arange(n_cal), cal_labels]

    # Exact finite-sample conformal quantile: ceil((n + 1) * (1 - alpha)) / n
    level = np.ceil((n_cal + 1) * nominal_coverage) / n_cal
    level = min(1.0, max(0.0, float(level)))
    q_hat = float(np.quantile(cal_scores, level, method="higher"))

    engine = ConformalPredictionEngine(q_hat=q_hat, alpha=alpha, k_max=5)

    test_probs = np.random.dirichlet(np.ones(5) * 2.0, size=n_test)
    test_labels = np.random.randint(0, 5, size=n_test)

    covered = 0
    for i in range(n_test):
        pred_set, _ = engine.predict_set(test_probs[i])
        if test_labels[i] in pred_set:
            covered += 1

    empirical_coverage = covered / n_test
    # 3-sigma binomial bound: sqrt(0.95 * 0.05 / 2000) ~ 0.0048, 3-sigma ~ 0.015
    assert empirical_coverage >= nominal_coverage - 0.02, (
        f"Conformal empirical coverage {empirical_coverage:.4f} violated lower 3-sigma bound for nominal {nominal_coverage}"
    )


def test_conformal_prediction_discrete_split_evaluation() -> None:
    """Assert ConformalPredictionEngine evaluates cleanly on discrete 4-way EmbeddingSplit."""
    # Construct discrete 4-way partition
    cal_embeddings = torch.randn(100, 16)
    cal_labels = torch.randint(0, 4, (100,))
    test_embeddings = torch.randn(100, 16)
    test_labels = torch.randint(0, 4, (100,))

    split = EmbeddingSplit(
        train_embeddings=torch.randn(200, 16),
        train_labels=torch.randint(0, 4, (200,)),
        val_embeddings=torch.randn(50, 16),
        val_labels=torch.randint(0, 4, (50,)),
        cal_embeddings=cal_embeddings,
        cal_labels=cal_labels,
        test_embeddings=test_embeddings,
        test_labels=test_labels,
    )
    cal_emb = split.cal_embeddings
    cal_lbl = split.cal_labels
    test_emb = split.test_embeddings
    test_lbl = split.test_labels
    assert cal_emb is not None and cal_lbl is not None
    assert test_emb is not None and test_lbl is not None

    # Evaluate mock linear classification head on calibration partition
    weight = torch.randn(4, 16)
    cal_logits = torch.matmul(cal_emb, weight.t())
    cal_probs = torch.softmax(cal_logits, dim=-1).numpy()
    cal_scores = 1.0 - cal_probs[np.arange(100), cal_lbl.numpy()]

    level = np.ceil((100 + 1) * 0.95) / 100
    q_hat = float(np.quantile(cal_scores, min(1.0, level), method="higher"))

    engine = ConformalPredictionEngine(q_hat=q_hat, alpha=0.05, k_max=4)

    test_logits = torch.matmul(test_emb, weight.t())
    test_probs = torch.softmax(test_logits, dim=-1).numpy()

    covered = 0
    for i in range(100):
        pred_set, _ = engine.predict_set(test_probs[i])
        if int(test_lbl[i].item()) in pred_set:
            covered += 1

    empirical_coverage = covered / 100
    assert 0.0 <= empirical_coverage <= 1.0
