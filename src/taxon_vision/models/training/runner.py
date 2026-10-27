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

import logging
import os
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
import yaml
from torch import nn, optim

from taxon_vision.config import get_settings
from taxon_vision.models.loss import ClassBalancedLoss
from taxon_vision.models.training.checkpoint import save_checkpoint
from taxon_vision.models.training.dataset import (
    load_or_create_training_split,
    resolve_feature_dim,
    resolve_synthetic_permission,
)
from taxon_vision.models.training.embeddings import train_head_on_cached_embeddings
from taxon_vision.models.training.promotion import _promote_model_if_better
from taxon_vision.models.training.tracking import (
    _log_calibrated_energy_threshold,
    _log_dataset_lineage,
    _log_mlflow_metrics,
    _log_training_hyperparameters,
    mlflow_run_scope,
)
from taxon_vision.models.training.tuning import tune_hyperparameters
from taxon_vision.models.training.types import (
    EmbeddingSplit,
    HeadTrainingConfig,
    ResolvedTrainingParameters,
    TrainingComponents,
    TrainingHyperparameters as TrainingHyperparameters,
    TrainingPipelineResult,
    TuningConfig,
)
from taxon_vision.models.training.visualizations import log_training_visualizations
from taxon_vision.service.inference import load_taxa_catalog
from taxon_vision.uncertainty.ood_detector import EnergyOODDetector

logger = logging.getLogger(__name__)


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


def build_training_components(
    dim: int,
    num_classes: int,
    samples: Sequence[int],
    hparams: TrainingHyperparameters,
    epochs: int,
    extractor: str,
    optuna_tuned: bool = False,
    applied_vram_fraction: float | None = None,
) -> TrainingComponents:
    """Build and wire classification head, loss criterion, optimizer, config, and params.

    Why:
        Encapsulates neural architecture and optimization setup from pre-resolved hyperparameters,
        eliminating the 15-parameter data clump and returning a strongly-typed container.

    Args:
        dim: Dimensionality of feature vectors.
        num_classes: Total number of classification categories.
        samples: Class sample distribution counts.
        hparams: Resolved training hyperparameters.
        epochs: Number of training epochs.
        extractor: Name of the feature extractor backbone.
        optuna_tuned: Whether hyperparameters originated from Bayesian optimization.
        applied_vram_fraction: Applied GPU memory fraction or None.

    Returns:
        Structured TrainingComponents container.
    """
    settings = get_settings()
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
        "optuna_tuned": optuna_tuned,
        "hyperparameter_source": hparams.hyperparameter_source,
        "cuda_memory_fraction": applied_vram_fraction if applied_vram_fraction is not None else "cpu",
    }
    return TrainingComponents(
        head=head,
        criterion=criterion,
        optimizer=optimizer,
        config=config,
        params=params,
    )


def _resolve_training_hyperparameters(
    extractor: str | None,
    epochs: int | None,
    batch_size: int | None,
    learning_rate: float | None,
    checkpoint_path: Path | str | None,
    settings: Any,
) -> ResolvedTrainingParameters:
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
        Structured ResolvedTrainingParameters container.
    """
    resolved_extractor = extractor or settings.model.default_extractor
    resolved_epochs = epochs if epochs is not None else settings.training.epochs
    resolved_batch_size = batch_size if batch_size is not None else settings.training.batch_size
    resolved_lr = learning_rate if learning_rate is not None else settings.training.learning_rate
    resolved_checkpoint = checkpoint_path or settings.model.head_checkpoint_path
    return ResolvedTrainingParameters(
        extractor=resolved_extractor,
        epochs=resolved_epochs,
        batch_size=resolved_batch_size,
        learning_rate=resolved_lr,
        checkpoint_path=resolved_checkpoint,
    )


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
) -> TrainingPipelineResult:
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
        Structured TrainingPipelineResult containing metrics, checkpoint path, duration, and status.
    """
    settings = get_settings()
    resolved = _resolve_training_hyperparameters(
        extractor=extractor,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        checkpoint_path=checkpoint_path,
        settings=settings,
    )

    applied_vram_fraction = configure_cuda_memory_budget(cuda_memory_fraction)
    t0 = time.perf_counter()
    logger.info(
        "Initiating head training for extractor '%s' across %d epochs (tune=%s)",
        resolved.extractor,
        resolved.epochs,
        tune,
    )

    catalog = load_taxa_catalog()
    if not catalog:
        raise RuntimeError("Authoritative taxa catalog contains 0 taxa.")
    num_classes = len(catalog)
    dim = resolve_feature_dim(resolved.extractor)

    resolved_allow_synthetic = resolve_synthetic_permission(
        require_cache=require_cache,
        allow_synthetic=allow_synthetic,
        default_fallback=settings.model.allow_untrained_fallback,
    )

    data = load_or_create_training_split(
        dim,
        num_classes,
        cache_path=cache_path,
        train_embeddings_path=train_embeddings,
        val_embeddings_path=val_embeddings,
        allow_synthetic=resolved_allow_synthetic,
    )
    samples = data.samples_per_class or [1] * num_classes

    run_name = f"train_{resolved.extractor}_{int(time.time())}"
    promotion_status: str | None = None
    mlflow_run_id: str | None = None

    initial_params = {
        "extractor": resolved.extractor,
        "epochs": resolved.epochs,
        "batch_size": resolved.batch_size,
        "learning_rate": resolved.learning_rate,
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

        hparams = _resolve_tier_hyperparameters(
            extractor=resolved.extractor,
            tune=tune,
            n_trials=n_trials,
            epochs=resolved.epochs,
            batch_size=resolved.batch_size,
            learning_rate=resolved.learning_rate,
            seed=settings.training.random_seed,
            data=data,
            num_classes=num_classes,
            study_name=study_name,
            reset_study=reset_study,
            settings=settings,
        )

        components = build_training_components(
            dim=dim,
            num_classes=num_classes,
            samples=samples,
            hparams=hparams,
            epochs=resolved.epochs,
            extractor=resolved.extractor,
            optuna_tuned=tune,
            applied_vram_fraction=applied_vram_fraction,
        )

        if active_run:
            _log_training_hyperparameters(components.params)

        history = train_head_on_cached_embeddings(
            head=components.head,
            data=data,
            optimizer=components.optimizer,
            criterion=components.criterion,
            config=components.config,
        )

        target_path = save_checkpoint(components.head, resolved.checkpoint_path)
        duration = time.perf_counter() - t0

        if active_run:
            _log_mlflow_metrics(history, duration)
            val_pr_auc = _extract_metric_last(history, "val_pr_auc", 0.0)

            components.head.eval()
            with torch.no_grad():
                val_logits_np = components.head(data.val_embeddings).cpu().numpy()

            calibrated_threshold = EnergyOODDetector.calibrate_threshold_from_logits(val_logits_np, percentile=95.0)
            _log_calibrated_energy_threshold(calibrated_threshold)

            log_training_visualizations(
                val_logits=val_logits_np,
                val_labels=data.val_labels.cpu().numpy(),
                taxa_catalog=load_taxa_catalog(),
                calibrated_energy_threshold=calibrated_threshold,
            )

            promotion_status = _promote_model_if_better(
                model=components.head,
                pr_auc=val_pr_auc,
                val_embeddings=data.val_embeddings,
                extractor=resolved.extractor,
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

    result: TrainingPipelineResult = {
        "status": "completed",
        "extractor": resolved.extractor,
        "epochs_trained": resolved.epochs,
        "final_train_loss": round(train_loss, 4),
        "final_val_loss": round(val_loss, 4),
        "final_val_accuracy": round(val_acc, 4),
        "checkpoint_path": str(target_path),
        "duration_seconds": round(duration, 3),
        "promotion_status": promotion_status,
        "hyperparameter_source": components.params["hyperparameter_source"],
        "mlflow_run_id": mlflow_run_id,
        "history": history,
    }
    return result
