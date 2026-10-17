# SPDX-FileCopyrightText: 2026 Benjamin Förster
# SPDX-License-Identifier: MIT
"""Stage 6: Conformal Prediction Calibration & OOD Energy Thresholds.

Consumes the exported quantized ONNX graph (models/checkpoints/model_int8.onnx) and discrete
calibration embeddings (data/embeddings/cal_embeddings.pt) to guarantee distribution-free coverage
(1 - alpha) and calibrate Helmholtz free energy OOD rejection bounds without PyTorch head.pt fallback.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

from taxon_vision.config import get_settings
from taxon_vision.inference.onnx_engine import ONNXInferenceEngine
from taxon_vision.uncertainty.conformal import SplitConformalPredictor
from taxon_vision.uncertainty.ood_detector import EnergyOODDetector

logger = logging.getLogger(__name__)


def load_embeddings_and_labels(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load embeddings and corresponding labels from a serialized .pt split file.

    Args:
        path: Path to serialized PyTorch embedding partition dictionary.

    Returns:
        Tuple of (embeddings, labels) as numpy arrays.
    """
    if not path.exists():
        raise FileNotFoundError(f"Embedding partition file not found: {path}")

    data = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(data, dict):
        embs = data.get("embeddings", data.get("cal_embeddings", data.get("test_embeddings")))
        lbls = data.get("labels", data.get("cal_labels", data.get("test_labels")))
    else:
        raise ValueError(f"Expected dict with embeddings and labels in {path}, got {type(data)}")

    if embs is None or lbls is None:
        raise ValueError(f"Could not extract embeddings or labels from {path}")

    embs_np = embs.numpy() if isinstance(embs, torch.Tensor) else np.asarray(embs, dtype=np.float32)
    lbls_np = lbls.numpy() if isinstance(lbls, torch.Tensor) else np.asarray(lbls, dtype=np.int64)
    return embs_np, lbls_np


def _save_calibration_artifacts(
    out_path: Path,
    metrics_path: Path | None,
    alpha: float,
    q_hat: float,
    tau_ood: float | None,
    sample_size: int,
    empirical_coverage: float,
) -> None:
    """Save calibration configuration and optional DVC summary metrics to disk."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    yaml_payload = {
        "calibration": {
            "method": "split-conformal",
            "nominal_coverage": round(1.0 - alpha, 4),
            "alpha": round(alpha, 4),
            "q_hat": round(q_hat, 4),
            "tau_ood": round(tau_ood, 4) if tau_ood is not None else None,
            "max_set_size_for_auto_accept": 3,
            "empty_set_action": "route_to_human_triage",
            "min_confidence_threshold": 0.60,
            "num_calibration_samples": sample_size,
        }
    }
    with out_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(yaml_payload, f, sort_keys=False)
    logger.info("Saved calibration configuration to %s", out_path)

    if metrics_path is not None:
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_meta = {
            "alpha": round(alpha, 4),
            "nominal_coverage": round(1.0 - alpha, 4),
            "empirical_calibration_coverage": round(empirical_coverage, 4),
            "q_hat": round(q_hat, 4),
            "tau_ood": round(tau_ood, 4) if tau_ood is not None else None,
            "num_calibration_samples": sample_size,
        }
        with metrics_path.open("w", encoding="utf-8") as f:
            json.dump(metrics_meta, f, indent=2)
        logger.info("Saved calibration summary metrics to %s", metrics_path)


def _build_arg_parser(default_alpha: float) -> argparse.ArgumentParser:
    """Construct CLI argument parser for conformal calibration."""
    parser = argparse.ArgumentParser(description="Calibrate Split Conformal Prediction and OOD energy thresholds.")
    parser.add_argument(
        "--onnx-model",
        type=Path,
        default=Path("models/checkpoints/model_int8.onnx"),
        help="Path to exported ONNX model (FP32 or INT8).",
    )
    parser.add_argument(
        "--embeddings",
        type=Path,
        default=Path("data/embeddings/cal_embeddings.pt"),
        help="Path to calibration embeddings partition (.pt).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("config/conformal_config.yaml"),
        help="Destination path for calibrated conformal configuration YAML.",
    )
    parser.add_argument(
        "--metrics-out",
        type=Path,
        default=None,
        help="Optional destination path for DVC summary metrics JSON.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=default_alpha,
        help=f"Significance level error budget (default: {default_alpha}).",
    )
    parser.add_argument(
        "--ood-tpr",
        type=float,
        default=0.95,
        help="Target true positive rate for in-distribution calibration (default: 0.95).",
    )
    parser.add_argument(
        "--allow-synthetic",
        action="store_true",
        default=False,
        help="Allow synthetic score generation when calibration files are absent.",
    )
    # Legacy flags for compatibility
    parser.add_argument("--cache-path", type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--split", type=str, default="cal", help=argparse.SUPPRESS)
    parser.add_argument("--checkpoint", type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--scores-path", type=Path, default=None, help=argparse.SUPPRESS)
    return parser


def calibrate_cli(argv: list[str] | None = None) -> int:
    """Execute Split Conformal Prediction and OOD energy calibration pipeline.

    Args:
        argv: Optional command-line arguments.

    Returns:
        Integer exit code (0 on success).
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    settings = get_settings()
    parser = _build_arg_parser(settings.conformal.alpha)
    args = parser.parse_args(argv)

    emb_path = args.embeddings
    if not emb_path.exists() and args.cache_path and Path(args.cache_path).exists():
        emb_path = Path(args.cache_path)

    if not emb_path.exists():
        if not args.allow_synthetic and not settings.model.allow_untrained_fallback:
            logger.error("Calibration embeddings not found at %s. Aborting.", emb_path)
            return 1
        logger.warning("Generating synthetic non-conformity scores and logits for testing.")
        np.random.seed(settings.training.random_seed)
        sample_size = 1000
        mock_probs = np.random.beta(a=8.0, b=1.0, size=sample_size)
        scores = 1.0 - mock_probs
        logits = np.random.randn(sample_size, 10).astype(np.float32)
    else:
        embeddings, labels = load_embeddings_and_labels(emb_path)
        sample_size = len(labels)
        engine = ONNXInferenceEngine(args.onnx_model)
        logits = engine.project_embeddings(embeddings)
        probs = engine.predict_from_embeddings(embeddings)
        scores = 1.0 - probs[np.arange(sample_size), labels]

    q_hat = SplitConformalPredictor.compute_quantile(scores, args.alpha)
    empirical_coverage = float(np.mean(scores <= q_hat))

    percentile = args.ood_tpr * 100.0 if args.ood_tpr <= 1.0 else args.ood_tpr
    tau_ood = EnergyOODDetector.calibrate_threshold_from_logits(logits, percentile=percentile)

    print(f"Calibrated Split Conformal Quantile for alpha={args.alpha} (Nominal Coverage {1 - args.alpha:.0%}):")
    print(f"  q_hat = {q_hat:.4f}")
    print(f"  Empirical Calibration Coverage = {empirical_coverage:.2%}")
    print(f"  Calibrated OOD Energy Threshold ({percentile:.1f}th pctl) = {tau_ood:.4f}")

    _save_calibration_artifacts(
        out_path=args.out,
        metrics_path=args.metrics_out,
        alpha=args.alpha,
        q_hat=q_hat,
        tau_ood=tau_ood,
        sample_size=sample_size,
        empirical_coverage=empirical_coverage,
    )
    return 0


main = calibrate_cli

if __name__ == "__main__":
    sys.exit(calibrate_cli())
