# SPDX-FileCopyrightText: 2026 Benjamin Förster
# SPDX-License-Identifier: MIT
"""Stage 4: Classification Head Training Pipeline Stage.

First-class CLI entrypoint for training the linear classification head on frozen
backbone representations using Class-Balanced Loss and optional Optuna Bayesian tuning.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

from taxon_vision.config import get_settings
from taxon_vision.models.training import runner as training_runner

logger = logging.getLogger(__name__)


def _build_arg_parser() -> argparse.ArgumentParser:
    """Construct CLI argument parser for Stage 4 training."""
    settings = get_settings()
    parser = argparse.ArgumentParser(description="TaxonVision Model Head Training CLI")
    parser.add_argument(
        "--extractor",
        type=str,
        default=settings.model.default_extractor,
        help=f"Backbone feature extractor (default: {settings.model.default_extractor})",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=settings.training.epochs,
        help=f"Number of training epochs (default: {settings.training.epochs})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=settings.training.batch_size,
        help=f"Mini-batch size (default: {settings.training.batch_size})",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=settings.training.learning_rate,
        help=f"Learning rate (default: {settings.training.learning_rate})",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default=None,
        help=f"Output checkpoint file path (defaults to {settings.model.head_checkpoint_path})",
    )
    parser.add_argument(
        "--api-url",
        type=str,
        default=None,
        help="Optional running API server URL to trigger remotely",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Optional log file destination",
    )
    parser.add_argument(
        "--cuda-memory-fraction",
        type=float,
        default=None,
        help="Optional maximum fraction of GPU VRAM (0.0 to 1.0) for training under time-slicing",
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Execute Optuna Bayesian hyperparameter search before final head training",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=15,
        help="Number of Optuna evaluation trials (when --tune is enabled)",
    )
    parser.add_argument(
        "--cache-path",
        type=str,
        default=None,
        help="Optional path to serialized .pt feature embedding cache file",
    )
    parser.add_argument(
        "--train-embeddings",
        type=str,
        default=None,
        help="Optional path to train_embeddings.pt partition file",
    )
    parser.add_argument(
        "--val-embeddings",
        type=str,
        default=None,
        help="Optional path to val_embeddings.pt partition file",
    )
    parser.add_argument(
        "--require-cache",
        action="store_true",
        help="Disallow fallback to synthetic data if embedding cache is absent",
    )
    parser.add_argument(
        "--study-name",
        type=str,
        default=None,
        help="Optional custom Optuna study name identifier",
    )
    parser.add_argument(
        "--reset-study",
        action="store_true",
        help="Reset existing Optuna study history to start with a fresh Bayesian optimization prior",
    )
    parser.add_argument(
        "--allow-synthetic",
        action="store_true",
        default=settings.model.allow_untrained_fallback,
        help="Permit fallback to synthetic Gaussian embeddings if cache is absent",
    )
    parser.add_argument(
        "--require-promotion",
        action="store_true",
        help="Exit with non-zero status if trained model is tagged as Archived instead of Challenger/Production",
    )
    parser.add_argument(
        "--metrics-out",
        type=str,
        default=None,
        help="Optional destination path to serialize training summary metrics as JSON",
    )
    return parser


def _save_metrics_json(metrics_out: str | None, result: dict[str, Any]) -> None:
    """Save execution summary metrics to target JSON destination if requested."""
    if not metrics_out:
        return
    metrics_file = Path(metrics_out)
    metrics_file.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": result.get("status"),
        "extractor": result.get("extractor"),
        "epochs_trained": result.get("epochs_trained"),
        "final_train_loss": result.get("final_train_loss"),
        "final_val_loss": result.get("final_val_loss"),
        "final_val_accuracy": result.get("final_val_accuracy"),
        "duration_seconds": result.get("duration_seconds"),
        "promotion_status": result.get("promotion_status"),
    }
    with metrics_file.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    logger.info("Saved training summary metrics to %s", metrics_file)


def train_cli(argv: list[str] | None = None) -> int:
    """CLI entrypoint for Stage 4: Classification Head Training.

    Args:
        argv: Optional list of command-line argument strings.

    Returns:
        Exit code: 0 for success, 1 on failure.
    """
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    training_runner.setup_logging(level=logging.INFO, log_file=args.log_file)

    if args.api_url:
        return training_runner._dispatch_remote_training(
            args.api_url, args.extractor, args.epochs, args.batch_size, args.lr
        )

    logger.info("Starting TaxonVision Training Pipeline [%s]", args.extractor)
    logger.info(
        "  Epochs: %d | Batch Size: %d | Learning Rate: %s | Tune: %s",
        args.epochs,
        args.batch_size,
        args.lr,
        args.tune,
    )

    result = training_runner.run_training_pipeline(
        extractor=args.extractor,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        checkpoint_path=args.checkpoint_path,
        log_file=args.log_file,
        cuda_memory_fraction=args.cuda_memory_fraction,
        tune=args.tune,
        n_trials=args.n_trials,
        cache_path=args.cache_path,
        train_embeddings=args.train_embeddings,
        val_embeddings=args.val_embeddings,
        require_cache=args.require_cache,
        allow_synthetic=args.allow_synthetic,
        study_name=args.study_name,
        reset_study=args.reset_study,
    )

    logger.info("Training Completed Successfully")
    logger.info("  Duration:         %.3fs", result["duration_seconds"])
    logger.info("  Final Train Loss: %s", result["final_train_loss"])
    logger.info("  Final Val Loss:   %s", result["final_val_loss"])
    logger.info("  Final Val Acc:    %.1f%%", result["final_val_accuracy"] * 100)
    logger.info("  Saved Checkpoint: %s", result["checkpoint_path"])

    _save_metrics_json(args.metrics_out, result)

    if args.require_promotion and result.get("promotion_status") == "Archived":
        logger.error("Model did not outperform production baseline (status: Archived). Failing promotion gate.")
        return 1

    return 0


main = train_cli

if __name__ == "__main__":
    sys.exit(main())
