#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Benjamin Förster
# SPDX-License-Identifier: MIT
"""Calibrate Split Conformal Prediction non-conformity quantiles.

Implements Angelopoulos & Bates (2021) conformal prediction:
- Input: Holdout validation softmax logits and ground truth labels.
- Output: Exact conformal non-conformity threshold q_hat for significance level alpha.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

from taxon_vision.config import get_settings

logger = logging.getLogger(__name__)


def compute_conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """Compute distribution-free split conformal prediction quantile.

    Why:
        Split conformal prediction provides mathematical finite-sample coverage guarantees
        (1 - alpha) by calibrating on non-conformity scores calculated on holdout data.

    How:
        Evaluates the empirical quantile at ceil((n + 1) * (1 - alpha)) / n using the
        higher order statistic.

    Complexity:
        O(n log n) sorting overhead or O(n) order statistic selection.

    Args:
        scores: The non-conformity scores array of shape (N,).
        alpha: The user-specified significance error tolerance.

    Returns:
        The calibrated conformal quantile threshold q_hat.
    """
    n = len(scores)
    level = np.ceil((n + 1) * (1.0 - alpha)) / n
    level = min(1.0, max(0.0, float(level)))
    q_hat = float(np.quantile(scores, level, method="higher"))
    return q_hat


def main() -> None:
    """Calibrate conformal quantile from empirical validation scores or permitted synthetic fallback.

    Why:
        Maintains epistemic guarantees by calibrating non-conformity thresholds from empirical
        evaluation scores, strictly failing fast when real validation artefacts are missing unless
        synthetic testing is explicitly authorised.

    How:
        Parses CLI flags, attempts to load real validation scores from disk, conditionally generates
        synthetic scores only if permitted, computes the conformal quantile, and persists metadata.

    Complexity:
        O(N log N) quantile calibration plus O(1) filesystem persistence.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Calibrate conformal quantile.")
    parser.add_argument(
        "--alpha",
        type=float,
        default=settings.conformal.alpha,
        help=f"Significance level (default: {settings.conformal.alpha})",
    )
    parser.add_argument("--out", type=Path, default=Path("config/conformal_calibration.json"))
    parser.add_argument(
        "--scores-path",
        type=Path,
        default=Path("data/conformal/calibration_scores.npy"),
        help="Path to calibration non-conformity scores or probabilities",
    )
    parser.add_argument(
        "--allow-synthetic",
        action="store_true",
        default=settings.model.allow_untrained_fallback,
        help="Allow synthetic score generation when calibration scores file is absent",
    )
    args = parser.parse_args()

    if args.scores_path.exists():
        nonconformity_scores = np.load(str(args.scores_path))
        sample_size = len(nonconformity_scores)
    else:
        if not args.allow_synthetic:
            logger.error(
                "Calibration scores file not found at '%s'. Pass --allow-synthetic to permit synthetic scores.",
                args.scores_path,
            )
            sys.exit(1)
        logger.warning(
            "Calibration scores file not found at '%s'; generating synthetic non-conformity scores for testing.",
            args.scores_path,
        )
        np.random.seed(settings.training.random_seed)
        sample_size = 1000
        mock_true_class_probs = np.random.beta(a=8.0, b=1.0, size=sample_size)
        nonconformity_scores = 1.0 - mock_true_class_probs

    q_hat = compute_conformal_quantile(nonconformity_scores, args.alpha)
    print(f"Calibrated Split Conformal Quantile for alpha={args.alpha} (Nominal Coverage {1 - args.alpha:.0%}):")
    print(f"  q_hat = {q_hat:.4f}")

    calibration_meta = {
        "alpha": args.alpha,
        "nominal_coverage": 1.0 - args.alpha,
        "q_hat": q_hat,
        "num_calibration_samples": sample_size,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        json.dump(calibration_meta, f, indent=2)
    print(f"Calibration metadata saved to {args.out}")


if __name__ == "__main__":
    main()
