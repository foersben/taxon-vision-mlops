# SPDX-FileCopyrightText: 2026 Benjamin Förster
# SPDX-License-Identifier: MIT
"""Training routines for cached feature embeddings."""

from __future__ import annotations

import copy
import logging

import torch
from sklearn.metrics import average_precision_score, f1_score
from sklearn.preprocessing import label_binarize
from torch import nn, optim

from taxon_vision.models.training.types import EmbeddingSplit, HeadTrainingConfig

logger = logging.getLogger(__name__)


def _compute_additional_metrics(val_logits: torch.Tensor, val_lbl: torch.Tensor) -> tuple[float, float]:
    """Compute Macro F1-Score and Macro PR-AUC from validation logits and labels.

    Why:
        In severe power-law biological classification, raw top-1 accuracy is misleading: a model can predict only the dominant 10% of species and still achieve 90% accuracy while failing on the remaining 90% of species. Macro F1 and Macro Precision-Recall AUC weight each taxon category equally, providing a sensitive evaluation of performance across rare and minority taxa.

    How:
        Converts PyTorch validation logits to NumPy softmax probability distributions, binarizes ground-truth labels across classes via one-hot encoding, and calculates unweighted macro-averaged F1 and average precision (PR-AUC) scores via scikit-learn metrics. Gracefully returns (0.0, 0.0) if evaluation encounters numerical anomalies on empty classes.

    Args:
        val_logits: Raw model predictions of shape (batch_size, num_classes).
        val_lbl: Ground truth class indices of shape (batch_size,).

    Returns:
        A tuple containing (macro_f1, macro_pr_auc). If computation fails,
        returns (0.0, 0.0) to prevent crashing the training loop.
    """
    try:
        import numpy as np

        y_true = val_lbl.cpu().numpy()
        preds = torch.argmax(val_logits, dim=1).cpu().numpy()
        y_score = torch.softmax(val_logits, dim=1).cpu().numpy()

        num_classes = val_logits.size(1)
        classes = list(range(num_classes))
        y_true_bin = label_binarize(y_true, classes=classes)

        if num_classes == 2:
            y_true_bin = np.hstack((1 - y_true_bin, y_true_bin))

        val_f1 = float(f1_score(y_true, preds, average="macro", zero_division=0))
        val_pr_auc = float(average_precision_score(y_true_bin, y_score, average="macro"))
        return val_f1, val_pr_auc
    except Exception:
        return 0.0, 0.0


def _train_cached_epoch(
    head: nn.Module,
    train_emb: torch.Tensor,
    train_lbl: torch.Tensor,
    permutation: torch.Tensor,
    batch_size: int,
    noise_std: float,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
) -> float:
    """Execute a single training epoch over cached embeddings in mini-batches.

    Args:
        head: Differentiable classification head module.
        train_emb: Device tensor of training representations.
        train_lbl: Device tensor of training integer targets.
        permutation: Pseudo-random permutation of indices.
        batch_size: Mini-batch size.
        noise_std: Standard deviation of Gaussian noise injected into representations.
        optimizer: Optimizer targeting head parameters.
        criterion: Loss criterion.

    Returns:
        Mean training loss across batches.
    """
    head.train()
    epoch_loss = 0.0
    num_batches = 0
    num_samples = train_emb.shape[0]

    for i in range(0, num_samples, batch_size):
        indices = permutation[i : i + batch_size]
        batch_x, batch_y = train_emb[indices], train_lbl[indices]
        if noise_std > 0.0:
            batch_x = batch_x + torch.randn_like(batch_x) * noise_std

        optimizer.zero_grad()
        logits = head(batch_x)
        loss = criterion(logits, batch_y)
        loss.backward()
        optimizer.step()

        epoch_loss += float(loss.item())
        num_batches += 1

    return epoch_loss / max(1, num_batches)


def _evaluate_cached_split(
    head: nn.Module,
    val_emb: torch.Tensor,
    val_lbl: torch.Tensor,
    criterion: nn.Module,
) -> tuple[float, float, float, float]:
    """Evaluate validation loss, accuracy, Macro F1, and PR-AUC on cached embeddings.

    Args:
        head: Classification head module in evaluation mode.
        val_emb: Device tensor of validation feature vectors.
        val_lbl: Device tensor of validation ground-truth labels.
        criterion: Differentiable loss function.

    Returns:
        Tuple of (val_loss, val_accuracy, val_f1, val_pr_auc).
    """
    head.eval()
    with torch.no_grad():
        val_logits = head(val_emb)
        val_loss = float(criterion(val_logits, val_lbl).item())
        preds = torch.argmax(val_logits, dim=1)
        val_acc = float((preds == val_lbl).sum().item()) / max(1, val_lbl.size(0))
        val_f1, val_pr_auc = _compute_additional_metrics(val_logits, val_lbl)
    return val_loss, val_acc, val_f1, val_pr_auc


def _track_checkpoint(
    head: nn.Module,
    val_pr_auc: float,
    best_pr_auc: float,
    best_weights: dict[str, torch.Tensor] | None,
    patience: int,
) -> tuple[float, dict[str, torch.Tensor] | None, int]:
    """Update best validation score and checkpoint state dict.

    Args:
        head: Model head module.
        val_pr_auc: Validation PR-AUC score for current epoch.
        best_pr_auc: Peak validation PR-AUC observed so far.
        best_weights: State dict of best model observed so far.
        patience: Current counter of stagnant epochs.

    Returns:
        Tuple of (updated_best_pr_auc, updated_best_weights, updated_patience).
    """
    if val_pr_auc > best_pr_auc:
        return val_pr_auc, copy.deepcopy(head.state_dict()), 0
    return best_pr_auc, best_weights, patience + 1


def _should_stop_early(
    epoch: int,
    val_pr_auc: float,
    patience_counter: int,
    cfg: HeadTrainingConfig,
) -> bool:
    """Evaluate whether training should terminate early via pruning or patience limits.

    Args:
        epoch: Current epoch index.
        val_pr_auc: Validation PR-AUC achieved.
        patience_counter: Number of epochs since last improvement.
        cfg: Head training configuration.

    Returns:
        True if training should stop immediately, False otherwise.
    """
    if cfg.pruner_callback is not None and cfg.pruner_callback(epoch, val_pr_auc):
        return True
    if cfg.early_stopping_patience is not None and patience_counter >= cfg.early_stopping_patience:
        logger.info(
            "Early stopping triggered at epoch %d: val_pr_auc did not improve for %d epochs.",
            epoch,
            cfg.early_stopping_patience,
        )
        return True
    return False


def train_head_on_cached_embeddings(
    head: nn.Module,
    data: EmbeddingSplit,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    config: HeadTrainingConfig | None = None,
) -> dict[str, list[float]]:
    """Train a linear classification head directly on pre-computed feature embeddings.

    Why:
        Linear probing on cached embeddings decouples representation extraction from classifier head optimization. Running forward and backward passes across deep 100M+ parameter backbones on every epoch wastes orders of magnitude of GPU compute and electricity. Pre-computing feature representations once and optimizing the lightweight head on static vectors reduces training time from hours to seconds while preventing gradient instability and memory bloat.

    How:
        1. Moves cached embedding and label tensors onto the active compute device.
        2. In each epoch, generates a pseudo-random permutation over training indices.
        3. Iterates mini-batches: computes head logits, evaluates Class-Balanced Loss, executes gradient backpropagation (`loss.backward()`), and steps the optimizer.
        4. Switches head to evaluation mode (`eval()`), computes validation loss, top-1 accuracy, macro F1, and PR-AUC.
        5. Calls the pruning callback (if provided by Optuna hyperparameter tuning) to terminate unpromising trials early.

    Complexity:
        O(E * N * D * K) where E is epochs, N is sample count, D is feature dimension, and K is class count. Near-zero dynamic heap allocations because training arrays are pre-allocated in device memory.

    Args:
        head: Differentiable classification head module (e.g. Dropout followed by Linear).
        data: Pre-computed feature representations and ground-truth labels for training and validation splits.
        optimizer: PyTorch optimizer configured for the head parameters.
        criterion: Differentiable loss function (e.g. ClassBalancedLoss).
        config: Hyperparameters governing training epochs, batch size, and pruning callbacks.

    Returns:
        Dictionary containing recorded metric histories per epoch:
            - `"train_loss"`: Training loss history across epochs.
            - `"val_loss"`: Validation loss history across epochs.
            - `"val_accuracy"`: Top-1 validation accuracy history across epochs.
            - `"val_f1"`: Macro F1-Score history across epochs.
            - `"val_pr_auc"`: Macro PR-AUC history across epochs.
    """
    cfg = config or HeadTrainingConfig()
    history: dict[str, list[float]] = {
        "train_loss": [],
        "val_loss": [],
        "val_accuracy": [],
        "val_f1": [],
        "val_pr_auc": [],
    }

    num_samples = data.train_embeddings.shape[0]
    device = next(head.parameters()).device

    train_emb = data.train_embeddings.to(device)
    train_lbl = data.train_labels.to(device)
    val_emb = data.val_embeddings.to(device)
    val_lbl = data.val_labels.to(device)

    scheduler = (
        optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=cfg.reduce_lr_patience)
        if cfg.reduce_lr_patience is not None
        else None
    )

    best_pr_auc = -1.0
    best_weights: dict[str, torch.Tensor] | None = None
    patience_counter = 0

    for epoch in range(cfg.epochs):
        permutation = torch.randperm(num_samples, device=device)
        avg_train_loss = _train_cached_epoch(
            head=head,
            train_emb=train_emb,
            train_lbl=train_lbl,
            permutation=permutation,
            batch_size=cfg.batch_size,
            noise_std=cfg.noise_std,
            optimizer=optimizer,
            criterion=criterion,
        )
        val_loss, val_acc, val_f1, val_pr_auc = _evaluate_cached_split(
            head=head,
            val_emb=val_emb,
            val_lbl=val_lbl,
            criterion=criterion,
        )

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_acc)
        history["val_f1"].append(val_f1)
        history["val_pr_auc"].append(val_pr_auc)

        if scheduler is not None:
            scheduler.step(val_pr_auc)

        best_pr_auc, best_weights, patience_counter = _track_checkpoint(
            head, val_pr_auc, best_pr_auc, best_weights, patience_counter
        )

        if _should_stop_early(epoch, val_pr_auc, patience_counter, cfg):
            break

    if cfg.restore_best_weights and best_weights is not None:
        head.load_state_dict(best_weights)
        logger.debug("Restored best model weights with val_pr_auc: %.4f", best_pr_auc)

    return history
