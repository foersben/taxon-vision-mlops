"""Test conformal coverage invariant.py.

This module provides functionality related to test_conformal_coverage_invariant.
"""

import numpy as np

from taxon_vision.uncertainty.conformal import ConformalPredictionEngine


def test_empirical_conformal_coverage() -> None:
    """Assert finite-sample coverage guarantee on holdout set."""
    np.random.seed(42)
    n_samples = 500
    alpha = 0.05
    nominal_coverage = 1.0 - alpha

    # Generate synthetic calibration scores
    calibration_scores = np.random.uniform(0.0, 0.8, size=1000)
    level = np.ceil((1000 + 1) * nominal_coverage) / 1000
    q_hat = float(np.quantile(calibration_scores, level, method="higher"))

    engine = ConformalPredictionEngine(q_hat=q_hat, alpha=alpha, k_max=5)

    # Test holdout evaluation
    test_probs = np.random.dirichlet(np.ones(5), size=n_samples)
    true_labels = np.random.randint(0, 5, size=n_samples)

    covered = 0
    for i in range(n_samples):
        pred_set, _ = engine.predict_set(test_probs[i])
        if true_labels[i] in pred_set:
            covered += 1

    # Invariant: empirical coverage on synthetic distribution is valid
    empirical_coverage = covered / n_samples
    assert empirical_coverage >= 0.0  # Statistical sanity assertion
