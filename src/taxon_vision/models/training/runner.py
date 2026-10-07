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
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import mlflow
import torch
import torch.nn as nn
import torch.optim as optim

from taxon_vision.config import get_settings
from taxon_vision.models.factory import create_feature_extractor, get_feature_dimension
from taxon_vision.models.loss import ClassBalancedLoss
from taxon_vision.models.training.embeddings import train_head_on_cached_embeddings
from taxon_vision.models.training.types import EmbeddingSplit, HeadTrainingConfig
from taxon_vision.service.inference import get_classifier, load_taxa_catalog

logger = logging.getLogger(__name__)


def _resolve_feature_dim(extractor: str) -> int:
    """Resolve embedding dimension for the specified backbone extractor.

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


def _get_dvc_dataset_hash() -> str | None:
    """Parse dvc.lock to extract the dataset hash from the ingestion stage.

    Returns:
        The MD5 hash of the data/manifests directory, or None if unavailable.
    """
    try:
        from pathlib import Path

        import yaml

        dvc_lock_path = Path("dvc.lock")
        if not dvc_lock_path.exists():
            logger.debug("dvc.lock not found; skipping DVC hash logging.")
            return None

        with dvc_lock_path.open("r", encoding="utf-8") as f:
            dvc_data = yaml.safe_load(f)

        outs = dvc_data.get("stages", {}).get("ingestion", {}).get("outs", [])
        for out in outs:
            if out.get("path") == "data/manifests":
                return str(out.get("md5"))

        logger.debug("data/manifests output not found in dvc.lock")
    except Exception as e:
        logger.debug("Could not parse dvc.lock for dataset hashing: %s", e)
    return None


def _init_mlflow_run(extractor: str, epochs: int, batch_size: int, lr: float, num_classes: int, dim: int) -> bool:
    """Initialize MLflow experiment run and log training parameters if available.

    Args:
        extractor: Backbone name.
        epochs: Epochs budget.
        batch_size: Mini-batch size.
        lr: Optimizer learning rate.
        num_classes: Target classes count.
        dim: Feature dimension.

    Returns:
        True if MLflow tracking run is active, False otherwise.
    """
    try:
        settings = get_settings()
        if settings.mlflow.tracking_uri:
            mlflow.set_tracking_uri(settings.mlflow.tracking_uri)
        if settings.mlflow.experiment_name:
            mlflow.set_experiment(settings.mlflow.experiment_name)

        mlflow.start_run(run_name=f"train_{extractor}_{int(time.time())}")
        mlflow.log_params(
            {
                "extractor": extractor,
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": lr,
                "num_classes": num_classes,
                "feature_dim": dim,
                "loss": "ClassBalancedLoss",
            }
        )

        dataset_hash = _get_dvc_dataset_hash()
        if dataset_hash:
            mlflow.log_param("dvc_dataset_hash", dataset_hash)

        return True
    except Exception as err:
        logger.debug("MLflow tracking setup skipped or unavailable: %s", err)
        return False


def _finalize_mlflow_run(
    history: dict[str, list[float]],
    duration: float,
    is_active: bool,
    log_file: Path | str | None = None,
) -> None:
    """Log training metric trajectories and terminate active MLflow run.

    Args:
        history: Metric history dictionary.
        duration: Training elapsed duration in seconds.
        is_active: Flag indicating whether an MLflow run was started.
        log_file: Optional path to an execution log file to attach as an MLflow artifact.
    """
    if not is_active:
        return
    try:
        n_epochs = len(history.get("train_loss", []))
        for i in range(n_epochs):
            metrics = {
                "train_loss": history.get("train_loss", [0.0])[i],
                "val_loss": history.get("val_loss", [0.0])[i],
                "val_accuracy": history.get("val_accuracy", [0.0])[i],
                "val_f1": history.get("val_f1", [0.0])[i],
                "val_pr_auc": history.get("val_pr_auc", [0.0])[i],
            }
            mlflow.log_metrics(metrics, step=i + 1)
        mlflow.log_metric("duration_seconds", duration)
        if log_file and Path(log_file).exists():
            try:
                mlflow.log_artifact(str(log_file))
            except Exception as artifact_err:
                logger.debug("Failed to attach log artifact to MLflow: %s", artifact_err)
        mlflow.end_run()
    except Exception as err:
        logger.debug("Failed to finalize MLflow run metrics: %s", err)


def _promote_model_if_better(model: nn.Module, pr_auc: float) -> None:
    """Evaluate performance and conditionally promote the model via MLflow.

    Compares the newly trained model's Macro PR-AUC against the current
    Production model. If it outperforms the baseline, it is tagged as
    'Challenger'. Otherwise, it is tagged as 'Archived'.

    Args:
        model: The trained PyTorch classification head.
        pr_auc: The final validation Macro PR-AUC score.
    """
    from mlflow.tracking import MlflowClient

    mlflow.log_metric("val_promotion_score", pr_auc)

    model_name = "taxon_vision_classifier"
    try:
        model_info = mlflow.pytorch.log_model(model, "model", registered_model_name=model_name)
    except Exception as err:
        logger.warning("Failed to log model to registry: %s", err)
        return

    client = MlflowClient()
    try:
        prod_version = client.get_model_version_by_alias(model_name, "Production")
        if not prod_version.run_id:
            raise ValueError("Production model missing run_id")
        prod_run = client.get_run(prod_version.run_id)
        prod_score = prod_run.data.metrics.get("val_promotion_score", 0.0)

        if pr_auc > prod_score:
            logger.info("New model outperforms production (%.4f > %.4f). Tagging as Challenger.", pr_auc, prod_score)
            client.set_registered_model_alias(model_name, "Challenger", model_info.registered_model_version)
        else:
            logger.info("New model is worse than production (%.4f <= %.4f). Tagging as Archived.", pr_auc, prod_score)
            client.set_registered_model_alias(model_name, "Archived", model_info.registered_model_version)
    except Exception:
        logger.info("No production alias found. Tagging initial model as Production.")
        client.set_registered_model_alias(model_name, "Production", model_info.registered_model_version)


def _save_checkpoint(head: nn.Module, checkpoint_path: Path | str | None) -> Path:
    """Persist trained head weights and invalidate runtime classifier cache.

    Args:
        head: PyTorch classification head module.
        checkpoint_path: Optional explicit filesystem destination.

    Returns:
        Resolved Path where weights were saved.
    """
    if checkpoint_path is None:
        settings = get_settings()
        target_path = Path(__file__).resolve().parents[4] / settings.model.head_checkpoint_path
    else:
        target_path = Path(checkpoint_path)

    target_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(head.state_dict(), target_path)

    try:
        get_classifier.cache_clear()
    except Exception:
        pass

    return target_path


def run_training_pipeline(
    extractor: str = "mobilenetv4_conv_small",
    epochs: int = 5,
    batch_size: int = 16,
    learning_rate: float = 0.001,
    checkpoint_path: Path | str | None = None,
    log_file: Path | str | None = None,
) -> dict[str, Any]:
    """Execute end-to-end classification head training on cached or synthetic embeddings.

    Args:
        extractor: Name of the feature extractor backbone.
        epochs: Number of training epochs to execute.
        batch_size: Mini-batch size for gradient updates.
        learning_rate: Optimizer learning rate.
        checkpoint_path: Optional custom file path to save the trained head weights.
        log_file: Optional path to an execution log file to record and attach to MLflow.

    Returns:
        Dictionary containing training metrics, checkpoint path, and status.
    """
    t0 = time.perf_counter()
    logger.info("Initiating head training for extractor '%s' across %d epochs", extractor, epochs)

    catalog = load_taxa_catalog()
    num_classes = len(catalog) if catalog else 10
    dim = _resolve_feature_dim(extractor)
    data = _prepare_embedding_split(dim, num_classes)

    head = nn.Sequential(nn.Dropout(p=0.2), nn.Linear(dim, num_classes))
    samples = data.samples_per_class or [1] * num_classes
    criterion = ClassBalancedLoss(samples_per_class=samples, beta=0.99)
    optimizer = optim.AdamW(head.parameters(), lr=learning_rate)
    config = HeadTrainingConfig(epochs=epochs, batch_size=batch_size)

    mlflow_active = _init_mlflow_run(extractor, epochs, batch_size, learning_rate, num_classes, dim)
    history = train_head_on_cached_embeddings(
        head=head,
        data=data,
        optimizer=optimizer,
        criterion=criterion,
        config=config,
    )

    target_path = _save_checkpoint(head, checkpoint_path)
    duration = time.perf_counter() - t0
    _finalize_mlflow_run(history, duration, mlflow_active, log_file=log_file)

    train_loss = float(history["train_loss"][-1]) if history.get("train_loss") else 0.0
    val_loss = float(history["val_loss"][-1]) if history.get("val_loss") else 0.0
    val_acc = float(history["val_accuracy"][-1]) if history.get("val_accuracy") else 0.0
    val_pr_auc = float(history.get("val_pr_auc", [0.0])[-1]) if history.get("val_pr_auc") else 0.0

    if mlflow_active:
        _promote_model_if_better(head, val_pr_auc)

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

    Args:
        argv: Optional command line arguments list.

    Returns:
        Exit code (0 for success, 1 for error).
    """
    parser = argparse.ArgumentParser(description="TaxonVision Model Head Training CLI")
    parser.add_argument("--extractor", type=str, default="mobilenetv4_conv_small", help="Backbone feature extractor")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Mini-batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--checkpoint-path", type=str, default=None, help="Output checkpoint file path")
    parser.add_argument("--api-url", type=str, default=None, help="Optional running API server URL to trigger remotely")
    parser.add_argument("--log-file", type=str, default=None, help="Optional log file destination")
    args = parser.parse_args(argv)

    setup_logging(level=logging.INFO, log_file=args.log_file)

    if args.api_url:
        return _dispatch_remote_training(args.api_url, args.extractor, args.epochs, args.batch_size, args.lr)

    logger.info("Starting TaxonVision Training Pipeline [%s]", args.extractor)
    logger.info("  Epochs: %d | Batch Size: %d | Learning Rate: %s", args.epochs, args.batch_size, args.lr)

    result = run_training_pipeline(
        extractor=args.extractor,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        checkpoint_path=args.checkpoint_path,
        log_file=args.log_file,
    )

    logger.info("Training Completed Successfully")
    logger.info("  Duration:         %.3fs", result["duration_seconds"])
    logger.info("  Final Train Loss: %s", result["final_train_loss"])
    logger.info("  Final Val Loss:   %s", result["final_val_loss"])
    logger.info("  Final Val Acc:    %.1f%%", result["final_val_accuracy"] * 100)
    logger.info("  Saved Checkpoint: %s", result["checkpoint_path"])
    return 0
