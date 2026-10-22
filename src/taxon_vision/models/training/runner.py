# SPDX-FileCopyrightText: 2026 Benjamin Förster
# SPDX-License-Identifier: MIT
"""Baseline classification head training pipeline runner.

The training pipeline consists of the following steps:

1. Initialization of MLflow tracking server and logging of metadata.
2. Preparation of embedding splits for rapid head training.
3. Training of the classification head on cached embeddings.
4. Saving the trained head weights and invalidating the runtime classifier cache.
5. Logging of training metrics and conditional promotion of the model via MLflow.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import yaml
from torch import nn, optim

from taxon_vision.config import get_settings
from taxon_vision.models.factory import create_feature_extractor, get_feature_dimension
from taxon_vision.models.loss import ClassBalancedLoss
from taxon_vision.models.training.checkpoint import _save_checkpoint
from taxon_vision.models.training.embeddings import train_head_on_cached_embeddings
from taxon_vision.models.training.tracking import (
    _log_dataset_lineage,
    _log_mlflow_metrics,
    _promote_model_if_better,
    log_training_visualizations,
    mlflow_run_scope,
)
from taxon_vision.models.training.tuning import tune_hyperparameters
from taxon_vision.models.training.types import EmbeddingSplit, HeadTrainingConfig, TuningConfig
from taxon_vision.service.inference import load_taxa_catalog
from taxon_vision.uncertainty.ood_detector import EnergyOODDetector

logger = logging.getLogger(__name__)


def _resolve_feature_dim(extractor: str) -> int:
    """Resolve embedding dimension for the specified backbone extractor.

    Why:
        Different foundation backbones project representations into distinct vector spaces (e.g., MobileNetV4 at 1280, ViT-Base / BioCLIP at 768, ViT-Large at 1024). Dynamically querying the model architecture prevents hardcoding projection layer dimensions in the training configuration, decoupling the classifier head geometry from the upstream feature extraction architecture.

    How:
        Instantiates a zero-weight feature extractor instance via the factory, inspects the terminal pooling or classifier layer feature dimension, and falls back to a safe default (1280) if dynamic inspection fails.

    Args:
        extractor: Name of the vision foundation backbone.

    Returns:
        Integer dimensionality of the extracted representation vectors.
    """
    try:
        backbone = create_feature_extractor(extractor, pretrained=False, allow_fallback=True)
        return get_feature_dimension(backbone)
    except Exception as err:
        logger.error("Could not resolve backbone feature dimension for '%s': %s", extractor, err)
        raise RuntimeError(f"Could not determine feature dimension for backbone '{extractor}': {err}") from err


def _try_load_cached_split(candidate: Path, dim: int) -> EmbeddingSplit | None:
    """Attempt to deserialize and validate an embedding split from a candidate filesystem path.

    Why:
        Encapsulates file existence checks, PyTorch deserialization, and dimension validation
        into an isolated helper, eliminating nested branching in the split initialization loop.

    Args:
        candidate: File path pointing to a potential serialized `.pt` embedding cache.
        dim: Expected feature projection dimension.

    Returns:
        Validated EmbeddingSplit instance if compatible, otherwise None.
    """
    if not candidate.exists():
        return None

    try:
        payload = torch.load(candidate, map_location="cpu", weights_only=False)
        split: EmbeddingSplit | None = None
        if isinstance(payload, EmbeddingSplit):
            split = payload
        elif isinstance(payload, dict) and isinstance(payload.get("split"), EmbeddingSplit):
            split = payload["split"]
        elif (
            isinstance(payload, dict)
            and "train_embeddings" in payload
            and "val_embeddings" in payload
            and "train_labels" in payload
            and "val_labels" in payload
        ):
            split = EmbeddingSplit(
                train_embeddings=payload["train_embeddings"],
                train_labels=payload["train_labels"],
                val_embeddings=payload["val_embeddings"],
                val_labels=payload["val_labels"],
                cal_embeddings=payload.get("cal_embeddings"),
                cal_labels=payload.get("cal_labels"),
                test_embeddings=payload.get("test_embeddings"),
                test_labels=payload.get("test_labels"),
                samples_per_class=payload.get("samples_per_class"),
            )

        if split is not None and split.feature_dim == dim:
            logger.info(
                "Loaded pre-computed feature embeddings from '%s' (%d train, %d val samples, dim=%d)",
                candidate,
                split.train_embeddings.shape[0],
                split.val_embeddings.shape[0],
                split.feature_dim,
            )
            return split
        if split is not None:
            logger.warning(
                "Candidate cache '%s' has incompatible feature dimension (expected %d, got %d); skipping",
                candidate,
                dim,
                split.feature_dim,
            )
    except Exception as err:
        logger.warning("Could not deserialize cache '%s' (%s); falling back", candidate, err)

    return None


def _generate_synthetic_embedding_split(dim: int, num_classes: int) -> EmbeddingSplit:
    """Generate deterministic synthetic Gaussian embeddings for rapid offline testing.

    Why:
        Ensures headless CI runners without GPU resources or S3 credentials can verify
        training loops, metric logging, and checkpointing deterministically with zero latency.

    Args:
        dim: Dimensionality of feature vectors.
        num_classes: Total number of classification categories.

    Returns:
        Deterministic EmbeddingSplit seeded with PRNG seed 42.
    """
    seed = get_settings().training.random_seed
    logger.info("Initializing deterministic synthetic embedding split (seed=%d, %d dim)", seed, dim)
    torch.manual_seed(seed)
    num_train = 120
    num_val = 30
    num_cal = 20
    num_test = 20
    train_z = torch.randn(num_train, dim)
    train_y = torch.randint(0, num_classes, (num_train,))
    val_z = torch.randn(num_val, dim)
    val_y = torch.randint(0, num_classes, (num_val,))
    cal_z = torch.randn(num_cal, dim)
    cal_y = torch.randint(0, num_classes, (num_cal,))
    test_z = torch.randn(num_test, dim)
    test_y = torch.randint(0, num_classes, (num_test,))
    samples_per_class = [max(1, int((train_y == c).sum().item())) for c in range(num_classes)]

    return EmbeddingSplit(
        train_embeddings=train_z,
        train_labels=train_y,
        val_embeddings=val_z,
        val_labels=val_y,
        cal_embeddings=cal_z,
        cal_labels=cal_y,
        test_embeddings=test_z,
        test_labels=test_y,
        samples_per_class=samples_per_class,
    )


def _try_load_split_from_pair(train_path: Path, val_path: Path, dim: int) -> EmbeddingSplit | None:
    """Load train and validation partitions directly from discrete .pt tensor files.

    Args:
        train_path: Path to train_embeddings.pt.
        val_path: Path to val_embeddings.pt.
        dim: Expected feature dimension.

    Returns:
        Instantiated EmbeddingSplit or None if missing or incompatible.
    """
    if not train_path.exists() or not val_path.exists():
        return None
    try:
        train_payload = torch.load(train_path, map_location="cpu", weights_only=False)
        val_payload = torch.load(val_path, map_location="cpu", weights_only=False)

        train_z = train_payload["embeddings"] if isinstance(train_payload, dict) else train_payload
        train_y = train_payload["labels"] if isinstance(train_payload, dict) else None
        samples_per_class = train_payload.get("samples_per_class") if isinstance(train_payload, dict) else None
        feat_dim = (
            train_payload.get("feature_dim", train_z.shape[1]) if isinstance(train_payload, dict) else train_z.shape[1]
        )

        val_z = val_payload["embeddings"] if isinstance(val_payload, dict) else val_payload
        val_y = val_payload["labels"] if isinstance(val_payload, dict) else None

        if not isinstance(train_z, torch.Tensor) or not isinstance(train_y, torch.Tensor):
            return None
        if not isinstance(val_z, torch.Tensor) or not isinstance(val_y, torch.Tensor):
            return None
        if feat_dim != dim:
            logger.warning("Split embeddings at %s have dimension %d, expected %d", train_path, feat_dim, dim)
            return None

        split = EmbeddingSplit(
            train_embeddings=train_z,
            train_labels=train_y,
            val_embeddings=val_z,
            val_labels=val_y,
            samples_per_class=samples_per_class,
        )
        logger.info(
            "Loaded isolated train (%d) and val (%d) embeddings from %s, %s (dim=%d)",
            len(train_z),
            len(val_z),
            train_path,
            val_path,
            dim,
        )
        return split
    except Exception as err:
        logger.warning("Failed loading split tensors from %s / %s: %s", train_path, val_path, err)
        return None


def _prepare_embedding_split(
    dim: int,
    num_classes: int,
    cache_path: Path | str | None = None,
    train_embeddings_path: Path | str | None = None,
    val_embeddings_path: Path | str | None = None,
    allow_synthetic: bool = False,
) -> EmbeddingSplit:
    """Construct or restore deterministic embedding splits for rapid head training.

    Why:
        During rapid iteration, CI verification, and local benchmarking, loading pre-computed
        representations avoids incurring hours of redundant forward passes through the frozen
        vision foundation backbone. Restoring cached representations from disk enables training
        in seconds. Synthetic data generation is strictly opt-in to avoid masking data pipeline bugs.

    How:
        1. Checks whether discrete train/val split files exist.
        2. Checks whether an embedding cache exists at `cache_path` (or standard `data/embeddings/cached_features.pt`).
        3. If found, deserializes the payload, validates that `split.feature_dim == dim`, and returns the cached split.
        4. If absent or shape-incompatible and allow_synthetic is False, logs an error and raises FileNotFoundError.
        5. If allow_synthetic is True, generates a deterministic synthetic Gaussian split.

    Complexity:
        O(N * D) where N is the number of samples and D is the feature dimension.

    Args:
        dim: Dimensionality of feature vectors.
        num_classes: Total number of classification categories.
        cache_path: Optional custom filesystem path to a serialized `.pt` embedding cache.
        train_embeddings_path: Optional custom path to train_embeddings.pt.
        val_embeddings_path: Optional custom path to val_embeddings.pt.
        allow_synthetic: Whether to fall back to synthetic Gaussian embeddings if cache is missing.

    Returns:
        Pre-allocated and validated EmbeddingSplit data container.

    Raises:
        FileNotFoundError: If cache is absent and allow_synthetic is False.
    """
    if train_embeddings_path is not None and val_embeddings_path is not None:
        pair_loaded = _try_load_split_from_pair(Path(train_embeddings_path), Path(val_embeddings_path), dim)
        if pair_loaded is not None:
            return pair_loaded

    default_train = Path("data/embeddings/train_embeddings.pt")
    default_val = Path("data/embeddings/val_embeddings.pt")
    if cache_path is None and default_train.exists() and default_val.exists():
        pair_loaded = _try_load_split_from_pair(default_train, default_val, dim)
        if pair_loaded is not None:
            return pair_loaded

    candidates: list[Path] = (
        [Path(cache_path)] if cache_path is not None else [Path("data/embeddings/cached_features.pt")]
    )

    for candidate in candidates:
        loaded = _try_load_cached_split(candidate, dim)
        if loaded is not None:
            return loaded

    if not allow_synthetic:
        logger.error(
            "Embedding cache not found at candidates (%s). Synthetic fallback is disabled.",
            candidates,
        )
        raise FileNotFoundError(
            f"Embedding cache not found at candidates ({candidates}). Synthetic fallback is disabled via --require-cache."
        )

    logger.warning("Embedding cache not found; generating synthetic Gaussian split for testing.")
    return _generate_synthetic_embedding_split(dim, num_classes)


def configure_cuda_memory_budget(fraction: float | None = None) -> float | None:
    """Configure per-process CUDA memory fraction for safe multi-tenant GPU time-slicing.

    Why:
        Under NVIDIA Device Plugin time-slicing on shared single-GPU nodes (e.g. RTX 5070 Ti),
        compute is time-multiplexed across virtual slices, but VRAM is shared globally. Unbounded
        PyTorch allocations can trigger Out-Of-Memory (OOM) failures in co-located real-time
        inference services. Capping training VRAM (e.g. to 70% / 11.2 GB) permanently reserves
        sufficient memory for ONNX inference serving and system displays.

    How:
        Inspects CUDA availability. If available, reads `fraction` or the `TAXON_CUDA_MEMORY_FRACTION`
        environment variable (defaulting to 0.7), and applies `torch.cuda.set_per_process_memory_fraction`.

    Args:
        fraction: Optional explicit memory fraction between 0.0 and 1.0.

    Returns:
        Applied memory fraction, or None if CUDA is unavailable.
    """
    if not torch.cuda.is_available():
        return None

    target_fraction = fraction
    if target_fraction is None:
        raw_env = os.getenv("TAXON_CUDA_MEMORY_FRACTION", "0.7")
        try:
            target_fraction = float(raw_env)
        except ValueError:
            target_fraction = 0.7

    target_fraction = max(0.1, min(1.0, target_fraction))
    torch.cuda.set_per_process_memory_fraction(target_fraction)
    logger.info("Enforced CUDA per-process VRAM budget fraction: %.2f", target_fraction)
    return target_fraction


@dataclass(frozen=True)
class TrainingHyperparameters:
    """Resolved hyperparameter tuple and provenance tracking."""

    learning_rate: float
    weight_decay: float
    dropout: float
    cb_beta: float
    batch_size: int
    noise_std: float
    hyperparameter_source: str


def _load_persisted_best_params(
    extractor: str,
    path: Path | str = "models/checkpoints/best_params.yaml",
) -> dict[str, Any] | None:
    """Load persisted best hyperparameters for an extractor from YAML checkpoint.

    Args:
        extractor: Vision backbone extractor identifier.
        path: Filesystem path to best_params.yaml.

    Returns:
        Dictionary of best parameters if found, or None.
    """
    param_path = Path(path)
    if not param_path.is_file():
        return None
    try:
        with param_path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if isinstance(data, dict) and extractor in data:
            entry = data[extractor]
            if isinstance(entry, dict) and "best_params" in entry and isinstance(entry["best_params"], dict):
                return dict(entry["best_params"])
            if isinstance(entry, dict):
                return dict(entry)
    except Exception as err:
        logger.warning("Failed to load persisted hyperparameters from %s: %s", param_path, err)
    return None


def _resolve_tier_hyperparameters(
    extractor: str,
    tune: bool,
    n_trials: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    data: EmbeddingSplit,
    num_classes: int,
    study_name: str | None,
    reset_study: bool,
    settings: Any,
) -> TrainingHyperparameters:
    """Resolve training hyperparameters via 3-tier hierarchy: live study, persisted yaml, or default.

    Why:
        Decouples parameter discovery (Bayesian search) from routine training while enabling
        deterministic reuse of peak-performing hyperparameters across CI/CD and deployment runs.

    Args:
        extractor: Name of the feature extractor.
        tune: Whether to run live Optuna Bayesian search.
        n_trials: Number of trials if tuning.
        epochs: Number of training epochs.
        batch_size: User-specified or default mini-batch size.
        learning_rate: User-specified or default learning rate.
        seed: Random seed for deterministic reproducibility.
        data: Split containing train and validation embeddings.
        num_classes: Total number of classification categories.
        study_name: Optional custom study identifier.
        reset_study: Whether to purge prior trial records before starting.
        settings: Application settings container.

    Returns:
        Validated TrainingHyperparameters container.
    """
    if tune:
        logger.info("Tier 1: Executing Optuna hyperparameter optimization across %d trials...", n_trials)
        tuning_cfg = TuningConfig(
            n_trials=n_trials,
            epochs_per_trial=epochs,
            batch_size=batch_size,
            seed=seed,
            extractor_name=extractor,
            study_name=study_name,
            reset_study=reset_study,
        )
        best_params, _ = tune_hyperparameters(data=data, num_classes=num_classes, config=tuning_cfg)
        return TrainingHyperparameters(
            learning_rate=float(best_params.get("learning_rate", best_params.get("lr", learning_rate))),
            weight_decay=float(best_params.get("weight_decay", settings.training.weight_decay)),
            dropout=float(best_params.get("dropout", settings.training.dropout)),
            cb_beta=float(best_params.get("cb_beta", best_params.get("beta", settings.training.cb_beta))),
            batch_size=int(best_params.get("batch_size", batch_size)),
            noise_std=float(best_params.get("noise_std", settings.training.noise_std)),
            hyperparameter_source="optuna_live_study",
        )

    persisted = _load_persisted_best_params(extractor)
    if persisted:
        logger.info("Tier 2: Loaded persisted best hyperparameters for '%s' from best_params.yaml", extractor)
        return TrainingHyperparameters(
            learning_rate=float(persisted.get("learning_rate", persisted.get("lr", learning_rate))),
            weight_decay=float(persisted.get("weight_decay", settings.training.weight_decay)),
            dropout=float(persisted.get("dropout", settings.training.dropout)),
            cb_beta=float(persisted.get("cb_beta", persisted.get("beta", settings.training.cb_beta))),
            batch_size=int(persisted.get("batch_size", batch_size)),
            noise_std=float(persisted.get("noise_std", settings.training.noise_std)),
            hyperparameter_source="persisted_best_params",
        )

    logger.info("Tier 3: Using baseline default hyperparameters from configuration")
    return TrainingHyperparameters(
        learning_rate=learning_rate,
        weight_decay=settings.training.weight_decay,
        dropout=settings.training.dropout,
        cb_beta=settings.training.cb_beta,
        batch_size=batch_size,
        noise_std=settings.training.noise_std,
        hyperparameter_source="params_yaml_default",
    )


def _build_training_components(
    dim: int,
    num_classes: int,
    samples: Sequence[int],
    tune: bool,
    n_trials: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    data: EmbeddingSplit,
    extractor: str,
    applied_vram_fraction: float | None = None,
    study_name: str | None = None,
    reset_study: bool = False,
) -> tuple[nn.Module, ClassBalancedLoss, optim.Optimizer, HeadTrainingConfig, dict[str, Any]]:
    """Build and wire classification head, loss criterion, optimizer, config, and params.

    Args:
        dim: Dimensionality of feature vectors.
        num_classes: Total number of classification categories.
        samples: Class sample distribution counts.
        tune: Whether to run Optuna Bayesian hyperparameter search.
        n_trials: Number of trials for tuning.
        epochs: Number of training epochs.
        batch_size: Mini-batch size.
        learning_rate: Base learning rate.
        seed: Random seed for deterministic reproducibility.
        data: Split containing train and validation embeddings.
        extractor: Name of the feature extractor.
        applied_vram_fraction: Applied GPU memory fraction or None.
        study_name: Optional custom Optuna study name identifier.
        reset_study: Whether to purge prior trial history before tuning.

    Returns:
        Tuple containing head module, loss criterion, optimizer, training config, and params dict.
    """
    settings = get_settings()
    hparams = _resolve_tier_hyperparameters(
        extractor=extractor,
        tune=tune,
        n_trials=n_trials,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        seed=seed,
        data=data,
        num_classes=num_classes,
        study_name=study_name,
        reset_study=reset_study,
        settings=settings,
    )

    head = nn.Sequential(nn.Dropout(p=hparams.dropout), nn.Linear(dim, num_classes))
    criterion = ClassBalancedLoss(samples_per_class=samples, beta=hparams.cb_beta)
    optimizer = optim.AdamW(head.parameters(), lr=hparams.learning_rate, weight_decay=hparams.weight_decay)

    config = HeadTrainingConfig(
        epochs=epochs,
        batch_size=hparams.batch_size,
        early_stopping_patience=settings.training.early_stopping_patience,
        reduce_lr_patience=settings.training.reduce_lr_patience,
        noise_std=hparams.noise_std,
    )
    params = {
        "extractor": extractor,
        "epochs": epochs,
        "batch_size": hparams.batch_size,
        "learning_rate": hparams.learning_rate,
        "weight_decay": hparams.weight_decay,
        "dropout": hparams.dropout,
        "cb_beta": hparams.cb_beta,
        "noise_std": hparams.noise_std,
        "num_classes": num_classes,
        "feature_dim": dim,
        "loss": "ClassBalancedLoss",
        "optuna_tuned": tune,
        "hyperparameter_source": hparams.hyperparameter_source,
        "cuda_memory_fraction": applied_vram_fraction if applied_vram_fraction is not None else "cpu",
    }
    return head, criterion, optimizer, config, params


def _resolve_training_hyperparameters(
    extractor: str | None,
    epochs: int | None,
    batch_size: int | None,
    learning_rate: float | None,
    checkpoint_path: Path | str | None,
    settings: Any,
) -> tuple[str, int, int, float, Path | str]:
    """Resolve hyperparameter defaults from application settings.

    Why:
        Encapsulates ternary fallback logic to decouple configuration loading from
        training loop execution, reducing cognitive complexity in the main runner.

    Complexity:
        O(1) constant time resolution.

    Args:
        extractor: Optional user-specified feature extractor backbone name.
        epochs: Optional training epoch count.
        batch_size: Optional mini-batch size.
        learning_rate: Optional optimizer learning rate.
        checkpoint_path: Optional custom destination path for head weights.
        settings: Application settings container.

    Returns:
        Tuple of resolved (extractor, epochs, batch_size, learning_rate, checkpoint_path).
    """
    resolved_extractor = extractor or settings.model.default_extractor
    resolved_epochs = epochs if epochs is not None else settings.training.epochs
    resolved_batch_size = batch_size if batch_size is not None else settings.training.batch_size
    resolved_lr = learning_rate if learning_rate is not None else settings.training.learning_rate
    resolved_checkpoint = checkpoint_path or settings.model.head_checkpoint_path
    return resolved_extractor, resolved_epochs, resolved_batch_size, resolved_lr, resolved_checkpoint


def _resolve_synthetic_permission(
    require_cache: bool,
    allow_synthetic: bool | None,
    default_fallback: bool,
) -> bool:
    """Determine whether synthetic embedding generation is permitted.

    Why:
        Ensures strict precedence: explicit --require-cache takes precedence,
        followed by explicit user flags, falling back to global settings.

    Complexity:
        O(1) constant time resolution.

    Args:
        require_cache: Strict requirement for real embedding cache.
        allow_synthetic: Optional explicit user flag.
        default_fallback: Global fallback setting.

    Returns:
        True if synthetic generation is authorized; otherwise False.
    """
    if require_cache:
        return False
    if allow_synthetic is not None:
        return allow_synthetic
    return default_fallback


def _extract_metric_last(history: dict[str, list[float]], key: str, default: float = 0.0) -> float:
    """Safely extract the final floating-point value from a training history metric list.

    Why:
        Guarantees safe scalar extraction across sparse or empty epoch histories without
        IndexError exceptions or verbose ternary conditionals.

    Complexity:
        O(1) constant time indexing.

    Args:
        history: Dictionary mapping metric names to epoch-indexed floats.
        key: Target metric key.
        default: Fallback scalar if history is empty.

    Returns:
        Final recorded float metric value.
    """
    series = history.get(key)
    if not series:
        return default
    return float(series[-1])


def run_training_pipeline(
    extractor: str | None = None,
    epochs: int | None = None,
    batch_size: int | None = None,
    learning_rate: float | None = None,
    checkpoint_path: Path | str | None = None,
    log_file: Path | str | None = None,
    cuda_memory_fraction: float | None = None,
    tune: bool = False,
    n_trials: int = 15,
    cache_path: Path | str | None = None,
    train_embeddings: Path | str | None = None,
    val_embeddings: Path | str | None = None,
    require_cache: bool = False,
    allow_synthetic: bool | None = None,
    study_name: str | None = None,
    reset_study: bool = False,
) -> dict[str, Any]:
    """Execute end-to-end classification head training on cached or synthetic embeddings.

    Why:
        Orchestrates the foundational model head training workflow: loads taxonomic target classes, resolves feature extractor projection dimensionality, prepares embedding splits, optionally tunes hyperparameters via Bayesian optimization with Optuna, initializes the linear classifier with Class-Balanced Loss, trains across specified epochs with early stopping, saves weight checkpoints, and records metrics and model lineage in MLflow.

    How:
        1. Resolves taxa count and feature dimension.
        2. Prepares deterministic embedding splits (from cache or synthetic fallback) and sample-per-class distributions.
        3. If `tune` is True, executes Optuna Bayesian search to find optimal LR, weight decay, dropout, and CB beta.
        4. Configures AdamW optimizer and ClassBalancedLoss with optimal or default hyperparameters.
        5. Enters `mlflow_run_scope` RAII context manager for clean tracking.
        6. Executes optimization loop over cached representations with early stopping and learning rate scheduling.
        7. Persists weights to checkpoint path and invalidates runtime inference cache.
        8. Logs metrics, attaches execution logs, and performs model registry promotion inside the active run.

    Args:
        extractor: Name of the feature extractor backbone.
        epochs: Number of training epochs to execute.
        batch_size: Mini-batch size for gradient updates.
        learning_rate: Optimizer learning rate.
        checkpoint_path: Optional custom file path to save the trained head weights.
        log_file: Optional path to an execution log file to record and attach to MLflow.
        cuda_memory_fraction: Optional CUDA VRAM fraction limit under time-slicing.
        tune: Whether to execute Optuna Bayesian hyperparameter search prior to final training.
        n_trials: Number of Optuna evaluation trials when `tune` is enabled.
        cache_path: Optional custom filesystem path to serialized `.pt` embedding cache.
        train_embeddings: Optional custom path to train_embeddings.pt partition.
        val_embeddings: Optional custom path to val_embeddings.pt partition.
        require_cache: Whether to strictly require real embedding cache and disable synthetic fallback.
        allow_synthetic: Explicit override to permit synthetic embeddings when cache is absent.
        study_name: Optional custom Optuna study name identifier.
        reset_study: Whether to purge prior trial history before tuning.

    Returns:
        Dictionary containing training metrics, checkpoint path, duration, and status.
    """
    settings = get_settings()
    (
        resolved_extractor,
        resolved_epochs,
        resolved_batch_size,
        resolved_lr,
        resolved_checkpoint,
    ) = _resolve_training_hyperparameters(extractor, epochs, batch_size, learning_rate, checkpoint_path, settings)

    applied_vram_fraction = configure_cuda_memory_budget(cuda_memory_fraction)
    t0 = time.perf_counter()
    logger.info(
        "Initiating head training for extractor '%s' across %d epochs (tune=%s)",
        resolved_extractor,
        resolved_epochs,
        tune,
    )

    catalog = load_taxa_catalog()
    if not catalog:
        raise RuntimeError("Authoritative taxa catalog contains 0 taxa.")
    num_classes = len(catalog)
    dim = _resolve_feature_dim(resolved_extractor)

    resolved_allow_synthetic = _resolve_synthetic_permission(
        require_cache=require_cache,
        allow_synthetic=allow_synthetic,
        default_fallback=settings.model.allow_untrained_fallback,
    )

    data = _prepare_embedding_split(
        dim,
        num_classes,
        cache_path=cache_path,
        train_embeddings_path=train_embeddings,
        val_embeddings_path=val_embeddings,
        allow_synthetic=resolved_allow_synthetic,
    )
    samples = data.samples_per_class or [1] * num_classes

    run_name = f"train_{resolved_extractor}_{int(time.time())}"
    promotion_status: str | None = None
    mlflow_run_id: str | None = None

    initial_params = {
        "extractor": resolved_extractor,
        "epochs": resolved_epochs,
        "batch_size": resolved_batch_size,
        "learning_rate": resolved_lr,
        "num_classes": num_classes,
        "tune": tune,
        "cuda_memory_fraction": applied_vram_fraction if applied_vram_fraction is not None else "cpu",
    }

    with mlflow_run_scope(run_name=run_name, params=initial_params, log_file=log_file) as active_run:
        if active_run:
            mlflow_run_id = active_run.info.run_id
            _log_dataset_lineage(
                train_emb=data.train_embeddings.cpu().numpy(),
                train_lbl=data.train_labels.cpu().numpy(),
            )

        head, criterion, optimizer, config, params = _build_training_components(
            dim=dim,
            num_classes=num_classes,
            samples=samples,
            tune=tune,
            n_trials=n_trials,
            epochs=resolved_epochs,
            batch_size=resolved_batch_size,
            learning_rate=resolved_lr,
            seed=settings.training.random_seed,
            data=data,
            extractor=resolved_extractor,
            applied_vram_fraction=applied_vram_fraction,
            study_name=study_name,
            reset_study=reset_study,
        )

        if active_run:
            try:
                import mlflow

                mlflow.log_params(params)
                mlflow.set_tag("hyperparameter_source", params["hyperparameter_source"])
            except Exception as tag_err:
                logger.debug("Failed to set hyperparameter_source tag: %s", tag_err)

        history = train_head_on_cached_embeddings(
            head=head,
            data=data,
            optimizer=optimizer,
            criterion=criterion,
            config=config,
        )

        target_path = _save_checkpoint(head, resolved_checkpoint)
        duration = time.perf_counter() - t0

        if active_run:
            _log_mlflow_metrics(history, duration)
            val_pr_auc = _extract_metric_last(history, "val_pr_auc", 0.0)

            head.eval()
            with torch.no_grad():
                val_logits_np = head(data.val_embeddings).cpu().numpy()

            calibrated_threshold = EnergyOODDetector.calibrate_threshold_from_logits(val_logits_np, percentile=95.0)
            try:
                import mlflow

                mlflow.log_metric("calibrated_ood_energy_threshold", calibrated_threshold)
            except Exception as metric_err:
                logger.debug("Failed to log calibrated energy threshold: %s", metric_err)

            log_training_visualizations(
                val_logits=val_logits_np,
                val_labels=data.val_labels.cpu().numpy(),
                taxa_catalog=load_taxa_catalog(),
                calibrated_energy_threshold=calibrated_threshold,
            )

            promotion_status = _promote_model_if_better(
                model=head,
                pr_auc=val_pr_auc,
                val_embeddings=data.val_embeddings,
                extractor=resolved_extractor,
            )

    train_loss = _extract_metric_last(history, "train_loss", 0.0)
    val_loss = _extract_metric_last(history, "val_loss", 0.0)
    val_acc = _extract_metric_last(history, "val_accuracy", 0.0)

    logger.info(
        "Training completed in %.3fs: train_loss=%.4f, val_loss=%.4f, val_acc=%.2f%%",
        duration,
        train_loss,
        val_loss,
        val_acc * 100.0,
    )

    return {
        "status": "completed",
        "extractor": resolved_extractor,
        "epochs_trained": resolved_epochs,
        "final_train_loss": round(train_loss, 4),
        "final_val_loss": round(val_loss, 4),
        "final_val_accuracy": round(val_acc, 4),
        "checkpoint_path": str(target_path),
        "duration_seconds": round(duration, 3),
        "promotion_status": promotion_status,
        "hyperparameter_source": params["hyperparameter_source"],
        "mlflow_run_id": mlflow_run_id,
        "history": history,
    }


def setup_logging(level: int = logging.INFO, log_file: Path | str | None = None) -> None:
    """Configure structured logging directed to stdout and an optional log file.

    Why:
        Training pipelines produce heterogeneous telemetry: real-time console feedback for interactive CLI users, persistent file audit logs for debugging distributed failures, and MLflow artifacts for experiment tracking. Standardizing log formats across stdout and file handlers ensures uniform timestamps and severity parsing across all runners.

    How:
        Retrieves the root logger, clears preexisting handlers to avoid duplicate message emission, attaches a formatted StreamHandler targeting stdout, and conditionally mounts a FileHandler with microsecond timestamps and source line numbers if `log_file` is provided.

    Args:
        level: Base logging severity level (default: logging.INFO).
        log_file: Optional path to a file destination for persistent log storage.
    """
    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%H:%M:%S")
    )
    root.addHandler(console_handler)

    if log_file:
        file_path = Path(log_file)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(file_path, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s:%(lineno)d: %(message)s"))
        root.addHandler(file_handler)


def _dispatch_remote_training(api_url: str, extractor: str, epochs: int, batch_size: int, lr: float) -> int:
    """Trigger training remotely on a running FastAPI service.

    Why:
        Enables headless training orchestration from external schedulers (e.g. cron jobs, CI/CD runners, or edge devices) without requiring local PyTorch GPU environments. Dispatching via standard HTTP requests allows decoupling compute infrastructure from the orchestration trigger.

    How:
        Constructs a POST request targeting the `/training` endpoint with serialized TrainingRequest JSON payload. Waits synchronously for completion up to 30 seconds, parses the JSON response, logs validation metrics, and returns the appropriate exit code.

    Args:
        api_url: Host base URL of the FastAPI service.
        extractor: Feature extractor backbone name.
        epochs: Number of training epochs.
        batch_size: Mini-batch size.
        lr: Learning rate.

    Returns:
        Process exit code (0 for success, 1 for error).
    """
    endpoint = f"{api_url.rstrip('/')}/training"
    payload = json.dumps(
        {
            "extractor": extractor,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": lr,
        }
    ).encode("utf-8")
    req = urllib.request.Request(endpoint, data=payload, headers={"Content-Type": "application/json"}, method="POST")
    logger.info("Triggering remote training via API: %s", endpoint)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            logger.info("Remote Training Completed: %s", data.get("message", "Success"))
            logger.info("  Status: %s | Epochs: %s", data.get("status"), data.get("epochs_trained"))
            logger.info("  Val Acc: %s | Loss: %s", data.get("final_val_accuracy"), data.get("final_val_loss"))
            return 0
    except urllib.error.URLError as err:
        logger.error("Failed to communicate with API server at %s: %s", endpoint, err)
        return 1


def train_cli(argv: list[str] | None = None) -> int:
    """CLI entrypoint for training pipeline execution.

    Why:
        Preserves backwards-compatibility for existing tests and callers while delegating
        to the canonical implementation in taxon_vision.pipelines.train.

    Args:
        argv: Optional command line arguments list.

    Returns:
        Exit code (0 for success, 1 for error).
    """
    from taxon_vision.pipelines.train import train_cli as _train_cli

    return _train_cli(argv)


if __name__ == "__main__":
    sys.exit(train_cli())
