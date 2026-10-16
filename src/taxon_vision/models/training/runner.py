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

import argparse
import json
import logging
import os
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
from torch import nn, optim

from taxon_vision.config import get_settings
from taxon_vision.models.factory import create_feature_extractor, get_feature_dimension
from taxon_vision.models.loss import ClassBalancedLoss
from taxon_vision.models.training.checkpoint import _save_checkpoint
from taxon_vision.models.training.embeddings import train_head_on_cached_embeddings
from taxon_vision.models.training.tracking import (
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


def _prepare_embedding_split(
    dim: int,
    num_classes: int,
    cache_path: Path | str | None = None,
    allow_synthetic: bool = False,
) -> EmbeddingSplit:
    """Construct or restore deterministic embedding splits for rapid head training.

    Why:
        During rapid iteration, CI verification, and local benchmarking, loading pre-computed
        representations avoids incurring hours of redundant forward passes through the frozen
        vision foundation backbone. Restoring cached representations from disk enables training
        in seconds. Synthetic data generation is strictly opt-in to avoid masking data pipeline bugs.

    How:
        1. Checks whether an embedding cache exists at `cache_path` (or standard `data/embeddings/cached_features.pt`).
        2. If found, deserializes the payload, validates that `split.feature_dim == dim`, and returns the cached split.
        3. If absent or shape-incompatible and allow_synthetic is False, logs an error and raises FileNotFoundError.
        4. If allow_synthetic is True, generates a deterministic synthetic Gaussian split.

    Complexity:
        O(N * D) where N is the number of samples and D is the feature dimension.

    Args:
        dim: Dimensionality of feature vectors.
        num_classes: Total number of classification categories.
        cache_path: Optional custom filesystem path to a serialized `.pt` embedding cache.
        allow_synthetic: Whether to fall back to synthetic Gaussian embeddings if cache is missing.

    Returns:
        Pre-allocated and validated EmbeddingSplit data container.

    Raises:
        FileNotFoundError: If cache is absent and allow_synthetic is False.
    """
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
    applied_vram_fraction: float | None,
) -> tuple[nn.Module, nn.Module, optim.Optimizer, HeadTrainingConfig, dict[str, Any]]:
    """Construct model head, loss, optimizer, and training parameters.

    Why:
        Decouples parameter resolution, Bayesian tuning with Optuna, and neural network
        instantiation from execution lifecycle orchestration in run_training_pipeline.

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

    Returns:
        Tuple containing head module, loss criterion, optimizer, training config, and params dict.
    """
    if tune:
        logger.info("Executing Optuna hyperparameter optimization across %d trials...", n_trials)
        tuning_cfg = TuningConfig(
            n_trials=n_trials,
            epochs_per_trial=epochs,
            batch_size=batch_size,
            seed=seed,
        )
        best_params, _ = tune_hyperparameters(data=data, num_classes=num_classes, config=tuning_cfg)
        dropout_p = float(best_params.get("dropout", 0.2))
        beta = float(best_params.get("beta", 0.99))
        lr = float(best_params.get("lr", learning_rate))
        weight_decay = float(best_params.get("weight_decay", 1e-4))
        resolved_batch_size = int(best_params.get("batch_size", batch_size))
        noise_std = float(best_params.get("noise_std", 0.0))
        head = nn.Sequential(nn.Dropout(p=dropout_p), nn.Linear(dim, num_classes))
        criterion = ClassBalancedLoss(samples_per_class=samples, beta=beta)
        optimizer = optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        dropout_p = 0.2
        beta = 0.99
        lr = learning_rate
        weight_decay = 0.0
        resolved_batch_size = batch_size
        noise_std = 0.0
        head = nn.Sequential(nn.Dropout(p=0.2), nn.Linear(dim, num_classes))
        criterion = ClassBalancedLoss(samples_per_class=samples, beta=0.99)
        optimizer = optim.AdamW(head.parameters(), lr=lr)

    config = HeadTrainingConfig(epochs=epochs, batch_size=resolved_batch_size, noise_std=noise_std)
    params = {
        "extractor": extractor,
        "epochs": epochs,
        "batch_size": resolved_batch_size,
        "learning_rate": lr,
        "weight_decay": weight_decay,
        "dropout": dropout_p,
        "beta": beta,
        "noise_std": noise_std,
        "num_classes": num_classes,
        "feature_dim": dim,
        "loss": "ClassBalancedLoss",
        "optuna_tuned": tune,
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
    require_cache: bool = False,
    allow_synthetic: bool | None = None,
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
        require_cache: Whether to strictly require real embedding cache and disable synthetic fallback.
        allow_synthetic: Explicit override to permit synthetic embeddings when cache is absent.

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
        allow_synthetic=resolved_allow_synthetic,
    )
    samples = data.samples_per_class or [1] * num_classes

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
    )

    run_name = f"train_{extractor}_{int(time.time())}"
    promotion_status: str | None = None

    with mlflow_run_scope(run_name=run_name, params=params, log_file=log_file) as active_run:
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
    """Entrypoint for the CLI training script.

    Why:
        Provides a unified command-line interface for human developers, automated Pixi tasks (`pixi run train`), and shell scripts to launch local or remote training pipelines with custom hyperparameters, backbones, and logging paths without modifying source code.

    How:
        Parses CLI flags using standard argparse, configures structured logging with optional file output, and routes execution either to `_dispatch_remote_training` (if `--api-url` is passed) or `run_training_pipeline` (for local execution), logging summary metrics upon completion.

    Args:
        argv: Optional command line arguments list.

    Returns:
        Exit code (0 for success, 1 for error).
    """
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
    parser.add_argument("--api-url", type=str, default=None, help="Optional running API server URL to trigger remotely")
    parser.add_argument("--log-file", type=str, default=None, help="Optional log file destination")
    parser.add_argument(
        "--cuda-memory-fraction",
        type=float,
        default=None,
        help="Optional maximum fraction of GPU VRAM (0.0 to 1.0) for training under time-slicing (defaults to TAXON_CUDA_MEMORY_FRACTION or 0.7)",
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
        "--require-cache",
        action="store_true",
        help="Disallow fallback to synthetic data if embedding cache is absent",
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
    args = parser.parse_args(argv)

    setup_logging(level=logging.INFO, log_file=args.log_file)

    if args.api_url:
        return _dispatch_remote_training(args.api_url, args.extractor, args.epochs, args.batch_size, args.lr)

    logger.info("Starting TaxonVision Training Pipeline [%s]", args.extractor)
    logger.info(
        "  Epochs: %d | Batch Size: %d | Learning Rate: %s | Tune: %s",
        args.epochs,
        args.batch_size,
        args.lr,
        args.tune,
    )

    result = run_training_pipeline(
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
        require_cache=args.require_cache,
        allow_synthetic=args.allow_synthetic,
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


def _save_metrics_json(metrics_out: str | None, result: dict[str, Any]) -> None:
    """Save execution summary metrics to target JSON destination if requested.

    Args:
        metrics_out: Optional filesystem path string.
        result: Dictionary of pipeline training execution metrics.
    """
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


if __name__ == "__main__":
    sys.exit(train_cli())
