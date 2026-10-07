# SPDX-FileCopyrightText: 2026 Benjamin Förster
# SPDX-License-Identifier: MIT
"""Rapid linear head training and automated hyperparameter optimization with Optuna and MLflow.

This module acts as a backwards-compatible facade for the `taxon_vision.models.training` package.
Please see the individual modules in `taxon_vision.models.training` for underlying implementations.
"""

import sys

from taxon_vision.models.training.embeddings import train_head_on_cached_embeddings
from taxon_vision.models.training.loops import evaluate_head, train_head_epoch
from taxon_vision.models.training.runner import run_training_pipeline, setup_logging, train_cli
from taxon_vision.models.training.tuning import tune_hyperparameters
from taxon_vision.models.training.types import EmbeddingSplit, HeadTrainingConfig, TuningConfig

__all__ = [
    "EmbeddingSplit",
    "HeadTrainingConfig",
    "TuningConfig",
    "evaluate_head",
    "run_training_pipeline",
    "setup_logging",
    "train_cli",
    "train_head_epoch",
    "train_head_on_cached_embeddings",
    "tune_hyperparameters",
]

if __name__ == "__main__":
    sys.exit(train_cli())
