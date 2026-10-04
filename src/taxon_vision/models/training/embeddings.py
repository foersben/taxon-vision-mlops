# SPDX-FileCopyrightText: 2026 Benjamin Förster
# SPDX-License-Identifier: MIT
"""Training routines for cached feature embeddings."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim

from taxon_vision.models.training.types import EmbeddingSplit, HeadTrainingConfig


def train_head_on_cached_embeddings(
    head: nn.Module,
    data: EmbeddingSplit,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    config: HeadTrainingConfig | None = None,
) -> dict[str, list[float]]:
    """Train a linear classification head directly on pre-computed feature embeddings.

    Executes near-instant training iterations on CPU or GPU because expensive forward
    passes through deep vision foundation backbones are eliminated.

    Args:
        head: Differentiable classification head module (e.g. Dropout followed by Linear).
        data: Pre-computed feature representations and ground-truth labels for training
            and validation splits.
        optimizer: PyTorch optimizer configured for the head parameters.
        criterion: Differentiable loss function (e.g. ClassBalancedLoss).
        config: Hyperparameters governing training epochs, batch size, and pruning callbacks.
            If None, default configuration settings are applied.

    Returns:
        Dictionary containing recorded metric histories per epoch:
            - `"train_loss"`: Training loss history across epochs.
            - `"val_loss"`: Validation loss history across epochs.
            - `"val_accuracy"`: Top-1 validation accuracy history across epochs.
    """
    cfg = config or HeadTrainingConfig()
    history: dict[str, list[float]] = {
        "train_loss": [],
        "val_loss": [],
        "val_accuracy": [],
    }

    num_samples = data.train_embeddings.shape[0]
    device = next(head.parameters()).device

    train_emb = data.train_embeddings.to(device)
    train_lbl = data.train_labels.to(device)
    val_emb = data.val_embeddings.to(device)
    val_lbl = data.val_labels.to(device)

    for epoch in range(cfg.epochs):
        head.train()
        permutation = torch.randperm(num_samples, device=device)
        epoch_loss = 0.0
        num_batches = 0

        for i in range(0, num_samples, cfg.batch_size):
            indices = permutation[i : i + cfg.batch_size]
            batch_x, batch_y = train_emb[indices], train_lbl[indices]

            optimizer.zero_grad()
            logits = head(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

            epoch_loss += float(loss.item())
            num_batches += 1

        avg_train_loss = epoch_loss / max(1, num_batches)
        history["train_loss"].append(avg_train_loss)

        # Validation evaluation
        head.eval()
        with torch.no_grad():
            val_logits = head(val_emb)
            val_loss_val = float(criterion(val_logits, val_lbl).item())
            preds = torch.argmax(val_logits, dim=1)
            val_acc = float((preds == val_lbl).sum().item()) / max(1, val_lbl.size(0))

        history["val_loss"].append(val_loss_val)
        history["val_accuracy"].append(val_acc)

        if cfg.pruner_callback is not None:
            should_stop = cfg.pruner_callback(epoch, val_acc)
            if should_stop:
                break

    return history
