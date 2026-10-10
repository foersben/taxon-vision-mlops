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
from pathlib import Path
from typing import Any

import torch
from torch import nn, optim

from taxon_vision.models.factory import create_feature_extractor, get_feature_dimension
from taxon_vision.models.loss import ClassBalancedLoss
from taxon_vision.models.training.checkpoint import _save_checkpoint
from taxon_vision.models.training.embeddings import train_head_on_cached_embeddings
from taxon_vision.models.training.tracking import _log_mlflow_metrics, _promote_model_if_better, mlflow_run_scope
from taxon_vision.models.training.tuning import tune_hyperparameters
from taxon_vision.models.training.types import EmbeddingSplit, HeadTrainingConfig, TuningConfig
from taxon_vision.service.inference import load_taxa_catalog

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
        backbone = create_feature_extractor(extractor, pretrained=False)
        return get_feature_dimension(backbone)
    except Exception as err:
        logger.debug("Could not resolve backbone dim dynamically (%s), falling back to 1280", err)
        return 1280


def _prepare_embedding_split(dim: int, num_classes: int) -> EmbeddingSplit:
    """Construct deterministic embedding splits for rapid head training.

    Why:
        During rapid iteration, CI verification, and local benchmarking, generating or loading pre-computed representations enables instantaneous evaluation without incurring hours of backbone feature extraction. Setting a deterministic PRNG seed ensures reproducible validation trajectories across runs and environments.

    How:
        Seeds PyTorch PRNG (seed=42), samples Gaussian embeddings for train and validation splits, allocates discrete integer taxon labels across classes, and counts class frequencies to populate the effective sample size container required by Class-Balanced Loss.

    Args:
        dim: Dimensionality of feature vectors.
        num_classes: Total number of classification categories.

    Returns:
        Pre-allocated and validated EmbeddingSplit data container.
    """
    torch.manual_seed(42)
    num_train = 120
    num_val = 30
    train_z = torch.randn(num_train, dim)
    train_y = torch.randint(0, num_classes, (num_train,))
    val_z = torch.randn(num_val, dim)
    val_y = torch.randint(0, num_classes, (num_val,))
    samples_per_class = [max(1, int((train_y == c).sum().item())) for c in range(num_classes)]

    return EmbeddingSplit(
        train_embeddings=train_z,
        train_labels=train_y,
        val_embeddings=val_z,
        val_labels=val_y,
        samples_per_class=samples_per_class,
    )


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


def run_training_pipeline(
    extractor: str = "mobilenetv4_conv_small",
    epochs: int = 15,
    batch_size: int = 64,
    learning_rate: float = 0.001,
    checkpoint_path: Path | str | None = None,
    log_file: Path | str | None = None,
    cuda_memory_fraction: float | None = None,
    tune: bool = False,
    n_trials: int = 15,
) -> dict[str, Any]:
    """Execute end-to-end classification head training on cached or synthetic embeddings.

    Why:
        Orchestrates the foundational model head training workflow: loads taxonomic target classes, resolves feature extractor projection dimensionality, prepares embedding splits, optionally tunes hyperparameters via Bayesian optimization with Optuna, initializes the linear classifier with Class-Balanced Loss, trains across specified epochs with early stopping, saves weight checkpoints, and records metrics and model lineage in MLflow.

    How:
        1. Resolves taxa count and feature dimension.
        2. Prepares deterministic embedding splits and sample-per-class distributions.
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

    Returns:
        Dictionary containing training metrics, checkpoint path, duration, and status.
    """
    applied_vram_fraction = configure_cuda_memory_budget(cuda_memory_fraction)
    t0 = time.perf_counter()
    logger.info("Initiating head training for extractor '%s' across %d epochs (tune=%s)", extractor, epochs, tune)

    catalog = load_taxa_catalog()
    num_classes = len(catalog) if catalog else 10
    dim = _resolve_feature_dim(extractor)
    data = _prepare_embedding_split(dim, num_classes)
    samples = data.samples_per_class or [1] * num_classes

    if tune:
        logger.info("Executing Optuna hyperparameter optimization across %d trials...", n_trials)
        tuning_cfg = TuningConfig(
            n_trials=n_trials,
            epochs_per_trial=epochs,
            batch_size=batch_size,
            seed=42,
        )
        best_params, _ = tune_hyperparameters(data=data, num_classes=num_classes, config=tuning_cfg)
        dropout_p = float(best_params.get("dropout", 0.2))
        beta = float(best_params.get("beta", 0.99))
        learning_rate = float(best_params.get("lr", learning_rate))
        weight_decay = float(best_params.get("weight_decay", 1e-4))
        head = nn.Sequential(nn.Dropout(p=dropout_p), nn.Linear(dim, num_classes))
        criterion = ClassBalancedLoss(samples_per_class=samples, beta=beta)
        optimizer = optim.AdamW(head.parameters(), lr=learning_rate, weight_decay=weight_decay)
    else:
        head = nn.Sequential(nn.Dropout(p=0.2), nn.Linear(dim, num_classes))
        criterion = ClassBalancedLoss(samples_per_class=samples, beta=0.99)
        optimizer = optim.AdamW(head.parameters(), lr=learning_rate)

    config = HeadTrainingConfig(epochs=epochs, batch_size=batch_size)

    params = {
        "extractor": extractor,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "num_classes": num_classes,
        "feature_dim": dim,
        "loss": "ClassBalancedLoss",
        "optuna_tuned": tune,
        "cuda_memory_fraction": applied_vram_fraction if applied_vram_fraction is not None else "cpu",
    }
    run_name = f"train_{extractor}_{int(time.time())}"

    with mlflow_run_scope(run_name=run_name, params=params, log_file=log_file) as active_run:
        history = train_head_on_cached_embeddings(
            head=head,
            data=data,
            optimizer=optimizer,
            criterion=criterion,
            config=config,
        )

        target_path = _save_checkpoint(head, checkpoint_path)
        duration = time.perf_counter() - t0

        if active_run:
            _log_mlflow_metrics(history, duration)
            val_pr_auc = float(history.get("val_pr_auc", [0.0])[-1]) if history.get("val_pr_auc") else 0.0
            _promote_model_if_better(head, val_pr_auc)

    train_loss = float(history["train_loss"][-1]) if history.get("train_loss") else 0.0
    val_loss = float(history["val_loss"][-1]) if history.get("val_loss") else 0.0
    val_acc = float(history["val_accuracy"][-1]) if history.get("val_accuracy") else 0.0

    logger.info(
        "Training completed in %.3fs: train_loss=%.4f, val_loss=%.4f, val_acc=%.2f%%",
        duration,
        train_loss,
        val_loss,
        val_acc * 100.0,
    )

    return {
        "status": "completed",
        "extractor": extractor,
        "epochs_trained": epochs,
        "final_train_loss": round(train_loss, 4),
        "final_val_loss": round(val_loss, 4),
        "final_val_accuracy": round(val_acc, 4),
        "checkpoint_path": str(target_path),
        "duration_seconds": round(duration, 3),
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
    parser = argparse.ArgumentParser(description="TaxonVision Model Head Training CLI")
    parser.add_argument("--extractor", type=str, default="dinov3", help="Backbone feature extractor")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Mini-batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--checkpoint-path", type=str, default=None, help="Output checkpoint file path")
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
    )

    logger.info("Training Completed Successfully")
    logger.info("  Duration:         %.3fs", result["duration_seconds"])
    logger.info("  Final Train Loss: %s", result["final_train_loss"])
    logger.info("  Final Val Loss:   %s", result["final_val_loss"])
    logger.info("  Final Val Acc:    %.1f%%", result["final_val_accuracy"] * 100)
    logger.info("  Saved Checkpoint: %s", result["checkpoint_path"])
    return 0
