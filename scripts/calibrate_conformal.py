#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Benjamin Förster
# SPDX-License-Identifier: MIT
"""Calibrate Split Conformal Prediction non-conformity quantiles and OOD thresholds.

Implements Angelopoulos & Bates (2021) conformal prediction and Helmholtz free energy calibration:
- Input: Holdout calibration embeddings and ground truth labels (or precomputed scores).
- Output: Exact conformal non-conformity threshold q_hat and OOD energy threshold tau_ood.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from taxon_vision.config import get_settings
from taxon_vision.models.training.types import EmbeddingSplit
from taxon_vision.uncertainty.ood_detector import EnergyOODDetector

logger = logging.getLogger(__name__)


def compute_conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """Compute distribution-free split conformal prediction quantile.

    Why:
        Split conformal prediction provides mathematical finite-sample coverage guarantees (1 - alpha)
        by calibrating on non-conformity scores calculated on holdout data.

    How:
        Evaluates the empirical quantile at ceil((n + 1) * (1 - alpha)) / n using the higher order statistic.

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


def _load_scores_from_cache(
    cache_path: Path,
    checkpoint_path: Path | None,
    split_name: str,
) -> tuple[np.ndarray, np.ndarray | None, int]:
    """Load calibration embeddings from cache and evaluate head checkpoint.

    Args:
        cache_path: Path to cached_features.pt.
        checkpoint_path: Optional path to trained classification head checkpoint.
        split_name: Name of the split to extract ('cal' or 'val').

    Returns:
        Tuple of (non-conformity scores, logits or None, sample size).
    """
    split = EmbeddingSplit.from_cache(cache_path)
    if split_name == "cal" and split.cal_embeddings is not None and split.cal_labels is not None:
        embeddings = split.cal_embeddings
        labels = split.cal_labels
    else:
        embeddings = split.val_embeddings
        labels = split.val_labels

    sample_size = len(labels)
    if checkpoint_path is None or not checkpoint_path.exists():
        logger.warning("Checkpoint %s not found; returning uniform baseline scores.", checkpoint_path)
        mock_probs = np.full(sample_size, 0.5)
        return 1.0 - mock_probs, None, sample_size

    raw_state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state_dict = raw_state.get("state_dict", raw_state.get("model_state_dict", raw_state))
    if any(k.startswith("head.") for k in state_dict):
        state_dict = {k.removeprefix("head."): v for k, v in state_dict.items() if k.startswith("head.")}

    weight_key = "1.weight" if "1.weight" in state_dict else "weight"
    weight_tensor = state_dict[weight_key]
    num_classes, feat_dim = weight_tensor.shape[0], weight_tensor.shape[1]

    head = nn.Sequential(nn.Dropout(0.0), nn.Linear(feat_dim, num_classes))
    head.load_state_dict(state_dict, strict=False)
    head.eval()

    with torch.no_grad():
        logits_tensor = head(embeddings)
        logits = logits_tensor.numpy()

    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(shifted)
    probs = exp_logits / (np.sum(exp_logits, axis=1, keepdims=True) + 1e-12)

    labels_np = labels.numpy()
    true_class_probs = probs[np.arange(sample_size), labels_np]
    scores = 1.0 - true_class_probs
    return scores, logits, sample_size


def _resolve_calibration_data(
    args: argparse.Namespace,
    random_seed: int,
) -> tuple[np.ndarray, np.ndarray | None, int]:
    """Resolve scores and optional logits from disk or synthetic generation.

    Args:
        args: Parsed command-line arguments.
        random_seed: PRNG seed for synthetic generation.

    Returns:
        Tuple of (non-conformity scores, logits or None, sample size).
    """
    if args.cache_path is not None and Path(args.cache_path).exists():
        checkpoint_p = Path(args.checkpoint) if args.checkpoint else None
        return _load_scores_from_cache(Path(args.cache_path), checkpoint_p, args.split)

    scores_p = Path(args.scores_path)
    if scores_p.exists():
        scores = np.load(str(scores_p))
        return scores, None, len(scores)

    if not args.allow_synthetic:
        logger.error(
            "Neither cache file (%s) nor scores file (%s) found. Pass --allow-synthetic to permit synthetic fallback.",
            args.cache_path,
            args.scores_path,
        )
        sys.exit(1)

    logger.warning("Generating synthetic non-conformity scores and logits for testing.")
    np.random.seed(random_seed)
    sample_size = 1000
    mock_true_class_probs = np.random.beta(a=8.0, b=1.0, size=sample_size)
    synthetic_scores = 1.0 - mock_true_class_probs
    synthetic_logits = np.random.randn(sample_size, 10).astype(np.float32)
    return synthetic_scores, synthetic_logits, sample_size


def _save_calibration_artifacts(
    out_path: Path,
    metrics_path: Path | None,
    alpha: float,
    q_hat: float,
    tau_ood: float | None,
    sample_size: int,
    empirical_coverage: float,
) -> None:
    """Save calibration configuration and optional DVC summary metrics to disk.

    Args:
        out_path: Path to target configuration JSON file.
        metrics_path: Optional path to target summary metrics JSON file.
        alpha: Error significance parameter.
        q_hat: Calibrated non-conformity quantile.
        tau_ood: Calibrated Helmholtz energy threshold or None.
        sample_size: Total calibration observation count.
        empirical_coverage: Empirical coverage rate achieved on calibration samples.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    calibration_meta: dict[str, Any] = {
        "alpha": alpha,
        "nominal_coverage": round(1.0 - alpha, 4),
        "q_hat": round(q_hat, 4),
        "tau_ood": round(tau_ood, 4) if tau_ood is not None else None,
        "num_calibration_samples": sample_size,
    }
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(calibration_meta, f, indent=2)
    logger.info("Saved calibration configuration to %s", out_path)

    if metrics_path is not None:
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_meta = {
            "alpha": alpha,
            "nominal_coverage": round(1.0 - alpha, 4),
            "empirical_calibration_coverage": round(empirical_coverage, 4),
            "q_hat": round(q_hat, 4),
            "tau_ood": round(tau_ood, 4) if tau_ood is not None else None,
            "num_calibration_samples": sample_size,
        }
        with metrics_path.open("w", encoding="utf-8") as f:
            json.dump(metrics_meta, f, indent=2)
        logger.info("Saved calibration summary metrics to %s", metrics_path)


def main() -> None:
    """Execute Split Conformal Prediction and OOD energy calibration pipeline."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Calibrate Split Conformal Prediction and OOD energy thresholds.")
    parser.add_argument(
        "--alpha",
        type=float,
        default=settings.conformal.alpha,
        help=f"Significance level (default: {settings.conformal.alpha})",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("config/conformal_calibration.json"),
        help="Destination path for calibrated conformal configuration JSON",
    )
    parser.add_argument(
        "--metrics-out",
        type=Path,
        default=None,
        help="Optional destination path for DVC summary metrics JSON",
    )
    parser.add_argument(
        "--cache-path",
        type=str,
        default=None,
        help="Path to serialized .pt feature embedding cache file",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="cal",
        help="Dataset split to calibrate against ('cal' or 'val')",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=settings.model.head_checkpoint_path,
        help="Path to trained classification head checkpoint (.pt)",
    )
    parser.add_argument(
        "--scores-path",
        type=Path,
        default=Path("data/conformal/calibration_scores.npy"),
        help="Fallback path to precomputed calibration non-conformity scores array",
    )
    parser.add_argument(
        "--allow-synthetic",
        action="store_true",
        default=settings.model.allow_untrained_fallback,
        help="Allow synthetic score generation when calibration files are absent",
    )
    args = parser.parse_args()

    scores, logits, sample_size = _resolve_calibration_data(args, settings.training.random_seed)
    q_hat = compute_conformal_quantile(scores, args.alpha)
    empirical_coverage = float(np.mean(scores <= q_hat))

    tau_ood: float | None = None
    if logits is not None:
        tau_ood = EnergyOODDetector.calibrate_threshold_from_logits(logits, percentile=95.0)

    print(f"Calibrated Split Conformal Quantile for alpha={args.alpha} (Nominal Coverage {1 - args.alpha:.0%}):")
    print(f"  q_hat = {q_hat:.4f}")
    print(f"  Empirical Calibration Coverage = {empirical_coverage:.2%}")
    if tau_ood is not None:
        print(f"  Calibrated OOD Energy Threshold (95th pctl) = {tau_ood:.4f}")

    _save_calibration_artifacts(
        out_path=args.out,
        metrics_path=args.metrics_out,
        alpha=args.alpha,
        q_hat=q_hat,
        tau_ood=tau_ood,
        sample_size=sample_size,
        empirical_coverage=empirical_coverage,
    )


if __name__ == "__main__":
    main()
