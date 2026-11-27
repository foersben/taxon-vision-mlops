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
from pathlib import Path
from typing import Any

import mlflow
import torch
import yaml
from torch import nn, optim

from taxon_vision.config.loader import get_settings
from taxon_vision.models.loss import ClassBalancedLoss
from taxon_vision.models.training.checkpoint import save_checkpoint
from taxon_vision.models.training.dataset import (
    load_or_create_training_split,
    resolve_feature_dim,
    resolve_synthetic_permission,
)
from taxon_vision.models.training.embeddings import train_head_on_cached_embeddings
from taxon_vision.models.training.tuning import tune_hyperparameters
from taxon_vision.models.training.types import (
    ComponentBuildSpec,
    EmbeddingSplit,
    HeadTrainingConfig,
    ResolvedTrainingParameters,
    TrainingComponents,
    TrainingHyperparameters as TrainingHyperparameters,
    TrainingPipelineResult,
    TrainingRunConfig,
    TuningConfig,
)
from taxon_vision.models.training.visualizations import log_training_visualizations
from taxon_vision.registry.dataset_lineage import (
    log_calibrated_energy_threshold,
    log_dataset_lineage,
    log_mlflow_metrics,
    log_training_hyperparameters,
)
from taxon_vision.registry.run_context import MLflowTracker, mlflow_run_scope
from taxon_vision.registry.types import LoggedDatasets, PyTorchModelLogSpec
from taxon_vision.service.inference import load_taxa_catalog
from taxon_vision.uncertainty.ood_detector import EnergyOODDetector

logger = logging.getLogger(__name__)


def configure_cuda_memory_budget(fraction: float | None = None) -> float | None:
    """Configure per-process CUDA memory fraction for safe multi-tenant GPU time-slicing.

    Why:
        Under NVIDIA Device Plugin time-slicing on shared single-GPU nodes (e.g. RTX 5070 Ti), compute is time-multiplexed across virtual slices, but VRAM is shared globally. Unbounded PyTorch allocations can trigger Out-Of-Memory (OOM) failures in co-located real-time inference services. Capping training VRAM (e.g. to 70% / 11.2 GB) permanently reserves sufficient memory for ONNX inference serving and system displays.

    How:
        Inspects CUDA availability. If available, reads `fraction` or the `TAXON_CUDA_MEMORY_FRACTION` environment variable (defaulting to 0.7), and applies `torch.cuda.set_per_process_memory_fraction`.

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
    config: TrainingRunConfig,
    resolved: ResolvedTrainingParameters,
    data: EmbeddingSplit,
    num_classes: int,
) -> TrainingHyperparameters:
    """Resolve training hyperparameters via 3-tier hierarchy: live study, persisted YAML, or default."""
    settings = get_settings()
    if config.tune:
        logger.info("Tier 1: Executing Optuna hyperparameter optimization across %d trials...", config.n_trials)
        tuning_cfg = TuningConfig(
            n_trials=config.n_trials,
            epochs_per_trial=resolved.epochs,
            batch_size=resolved.batch_size,
            seed=settings.training.random_seed,
            extractor_name=resolved.extractor,
            study_name=config.study_name,
            reset_study=config.reset_study,
            lr_min=settings.optuna.lr_min,
            lr_max=settings.optuna.lr_max,
            weight_decay_min=settings.optuna.weight_decay_min,
            weight_decay_max=settings.optuna.weight_decay_max,
            dropout_min=settings.optuna.dropout_min,
            dropout_max=settings.optuna.dropout_max,
            val_loss_penalty=settings.optuna.val_loss_penalty,
        )
        best_params, _ = tune_hyperparameters(data=data, num_classes=num_classes, config=tuning_cfg)
        return TrainingHyperparameters(
            learning_rate=float(best_params.get("learning_rate", best_params.get("lr", resolved.learning_rate))),
            weight_decay=float(best_params.get("weight_decay", settings.training.weight_decay)),
            dropout=float(best_params.get("dropout", settings.training.dropout)),
            cb_beta=float(best_params.get("cb_beta", best_params.get("beta", settings.training.cb_beta))),
            batch_size=int(best_params.get("batch_size", resolved.batch_size)),
            noise_std=float(best_params.get("noise_std", settings.training.noise_std)),
            hyperparameter_source="optuna_live_study",
        )

    persisted = _load_persisted_best_params(resolved.extractor)
    if persisted:
        logger.info("Tier 2: Loaded persisted best hyperparameters for '%s' from best_params.yaml", resolved.extractor)
        return TrainingHyperparameters(
            learning_rate=float(persisted.get("learning_rate", persisted.get("lr", resolved.learning_rate))),
            weight_decay=float(persisted.get("weight_decay", settings.training.weight_decay)),
            dropout=float(persisted.get("dropout", settings.training.dropout)),
            cb_beta=float(persisted.get("cb_beta", persisted.get("beta", settings.training.cb_beta))),
            batch_size=int(persisted.get("batch_size", resolved.batch_size)),
            noise_std=float(persisted.get("noise_std", settings.training.noise_std)),
            hyperparameter_source="persisted_best_params",
        )

    logger.info("Tier 3: Using baseline default hyperparameters from configuration")
    return TrainingHyperparameters(
        learning_rate=resolved.learning_rate,
        weight_decay=settings.training.weight_decay,
        dropout=settings.training.dropout,
        cb_beta=settings.training.cb_beta,
        batch_size=resolved.batch_size,
        noise_std=settings.training.noise_std,
        hyperparameter_source="params_yaml_default",
    )


def build_training_components(spec: ComponentBuildSpec) -> TrainingComponents:
    """Build and wire classification head, loss criterion, optimiser, config, and params.

    Why:
        Encapsulates neural architecture and optimisation setup from pre-resolved hyperparameters using typed `ComponentBuildSpec` to enforce Rule 08 parameter arity limits.

    Args:
        spec: Typed `ComponentBuildSpec` specifying dimensions, class counts, and hyperparameters.

    Returns:
        Structured `TrainingComponents` container.
    """
    settings = get_settings()
    head = nn.Sequential(nn.Dropout(p=spec.hparams.dropout), nn.Linear(spec.dim, spec.num_classes))
    criterion = ClassBalancedLoss(samples_per_class=spec.samples, beta=spec.hparams.cb_beta)
    optimizer = optim.AdamW(head.parameters(), lr=spec.hparams.learning_rate, weight_decay=spec.hparams.weight_decay)

    config = HeadTrainingConfig(
        epochs=spec.epochs,
        batch_size=spec.hparams.batch_size,
        early_stopping_patience=settings.training.early_stopping_patience,
        reduce_lr_patience=settings.training.reduce_lr_patience,
        noise_std=spec.hparams.noise_std,
    )
    params = {
        "extractor": spec.extractor,
        "epochs": spec.epochs,
        "batch_size": spec.hparams.batch_size,
        "learning_rate": spec.hparams.learning_rate,
        "weight_decay": spec.hparams.weight_decay,
        "dropout": spec.hparams.dropout,
        "cb_beta": spec.hparams.cb_beta,
        "noise_std": spec.hparams.noise_std,
        "num_classes": spec.num_classes,
        "feature_dim": spec.dim,
        "loss": "ClassBalancedLoss",
        "optuna_tuned": spec.optuna_tuned,
        "hyperparameter_source": spec.hparams.hyperparameter_source,
        "cuda_memory_fraction": spec.applied_vram_fraction if spec.applied_vram_fraction is not None else "cpu",
    }
    return TrainingComponents(
        head=head,
        criterion=criterion,
        optimizer=optimizer,
        config=config,
        params=params,
    )


def _resolve_training_hyperparameters(config: TrainingRunConfig, settings: Any) -> ResolvedTrainingParameters:
    """Resolve training baseline defaults from `TrainingRunConfig` and application settings."""
    return ResolvedTrainingParameters(
        extractor=config.extractor or settings.model.default_extractor,
        epochs=config.epochs if config.epochs is not None else settings.training.epochs,
        batch_size=config.batch_size if config.batch_size is not None else settings.training.batch_size,
        learning_rate=config.learning_rate if config.learning_rate is not None else settings.training.learning_rate,
        checkpoint_path=config.checkpoint_path or settings.model.head_checkpoint_path,
    )


def _extract_metric_last(history: dict[str, list[float]], key: str, default: float = 0.0) -> float:
    """Safely extract the final floating-point value from a training history metric list."""
    series = history.get(key)
    if not series:
        return default
    return float(series[-1])


def _record_training_run_artifacts(
    components: TrainingComponents,
    data: EmbeddingSplit,
    history: dict[str, list[float]],
    duration: float,
    *,
    logged_datasets: LoggedDatasets | None = None,
) -> str | None:
    """Serialise model card, trajectory metrics, OOD threshold, and visualisations to active MLflow run."""
    sample_val = data.val_embeddings[: min(5, data.val_embeddings.shape[0])].detach().cpu().numpy()
    model_info = MLflowTracker.log_model(
        PyTorchModelLogSpec(
            model=components.head,
            model_name="taxon_vision_classifier",
            params=components.params,
            input_example=sample_val,
        )
    )
    mlflow_model_id: str | None = None
    if model_info is not None:
        try:
            mlflow_model_id = model_info.model_id
        except AttributeError:
            mlflow_model_id = None

    val_dataset = logged_datasets.val if logged_datasets else None
    log_mlflow_metrics(history, duration, model_id=mlflow_model_id, dataset=val_dataset)

    components.head.eval()
    with torch.no_grad():
        val_logits_np = components.head(data.val_embeddings).cpu().numpy()

    calibrated_threshold = EnergyOODDetector.calibrate_threshold_from_logits(val_logits_np, percentile=95.0)
    log_calibrated_energy_threshold(calibrated_threshold)

    log_training_visualizations(
        val_logits=val_logits_np,
        val_labels=data.val_labels.cpu().numpy(),
        taxa_catalog=load_taxa_catalog(),
        calibrated_energy_threshold=calibrated_threshold,
    )

    mlflow.set_tag("registry.status", "candidate_trained")
    val_pr_auc = _extract_metric_last(history, "val_pr_auc", 0.0)
    mlflow.set_tag("candidate.val_pr_auc", f"{val_pr_auc:.4f}")
    if mlflow_model_id:
        mlflow.set_tag("mlflow.model_id", mlflow_model_id)
    return mlflow_model_id


def run_training_pipeline(config: TrainingRunConfig | None = None) -> TrainingPipelineResult:
    """Execute end-to-end classification head training on cached or synthetic embeddings.

    Why:
        Orchestrates taxonomic target loading, representation hydration, optional Optuna Bayesian search, Class-Balanced Loss training, checkpoint serialisation, and MLflow lineage capture cleanly within Rule 08 parameter arity and cognitive complexity limits.

    Args:
        config: Optional `TrainingRunConfig` overriding defaults from settings YAML.

    Returns:
        Structured `TrainingPipelineResult` containing metrics, checkpoint path, duration, and status.
    """
    run_cfg = config or TrainingRunConfig()
    settings = get_settings()
    resolved = _resolve_training_hyperparameters(run_cfg, settings)

    applied_vram_fraction = configure_cuda_memory_budget(run_cfg.cuda_memory_fraction)
    t0 = time.perf_counter()
    logger.info(
        "Initiating head training for extractor '%s' across %d epochs (tune=%s)",
        resolved.extractor,
        resolved.epochs,
        run_cfg.tune,
    )

    catalog = load_taxa_catalog()
    if not catalog:
        raise RuntimeError("Authoritative taxa catalog contains 0 taxa.")
    num_classes = len(catalog)
    dim = resolve_feature_dim(resolved.extractor)

    resolved_allow_synthetic = resolve_synthetic_permission(
        require_cache=run_cfg.require_cache,
        allow_synthetic=run_cfg.allow_synthetic,
        default_fallback=settings.model.allow_untrained_fallback,
    )

    data = load_or_create_training_split(
        dim,
        num_classes,
        cache_path=run_cfg.cache_path,
        train_embeddings_path=run_cfg.train_embeddings,
        val_embeddings_path=run_cfg.val_embeddings,
        allow_synthetic=resolved_allow_synthetic,
    )
    samples = data.samples_per_class or [1] * num_classes

    run_name = f"train_{resolved.extractor}_{int(time.time())}"
    promotion_status: str | None = None
    mlflow_run_id: str | None = None
    mlflow_model_id: str | None = None
    logged_datasets: LoggedDatasets | None = None

    initial_tags = {
        "extractor": resolved.extractor,
        "num_classes": str(num_classes),
        "tune": str(run_cfg.tune).lower(),
        "cuda_memory_fraction": str(applied_vram_fraction) if applied_vram_fraction is not None else "cpu",
    }
    # Pass params=None so immutable hyperparameter keys are NOT locked before Optuna finishes!
    with mlflow_run_scope(run_name=run_name, params=None, log_file=run_cfg.log_file) as active_run:
        if active_run:
            mlflow_run_id = active_run.info.run_id
            mlflow.set_tags(initial_tags)
            logged_datasets = log_dataset_lineage(data=data)

        hparams = _resolve_tier_hyperparameters(run_cfg, resolved, data, num_classes)
        components = build_training_components(
            ComponentBuildSpec(
                dim=dim,
                num_classes=num_classes,
                samples=samples,
                hparams=hparams,
                epochs=resolved.epochs,
                extractor=resolved.extractor,
                optuna_tuned=run_cfg.tune,
                applied_vram_fraction=applied_vram_fraction,
            )
        )

        if active_run:
            log_training_hyperparameters(components.params)

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
            mlflow_model_id = _record_training_run_artifacts(
                components,
                data,
                history,
                duration,
                logged_datasets=logged_datasets,
            )
            promotion_status = "candidate_trained"

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
        "mlflow_model_id": mlflow_model_id,
        "history": history,
    }
