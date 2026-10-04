# SPDX-FileCopyrightText: 2026 Benjamin Förster
# SPDX-License-Identifier: MIT
"""Standard training loops for taxonomic classification heads."""

from __future__ import annotations

from collections.abc import Iterable

import torch
import torch.nn as nn
import torch.optim as optim

from taxon_vision.models.head import TaxonClassifier


def train_head_epoch(
    model: TaxonClassifier,
    dataloader: Iterable[tuple[torch.Tensor, torch.Tensor]],
    optimizer: optim.Optimizer,
    criterion: nn.Module,
) -> float:
    """Execute a single training epoch across batches of raw image inputs.

    Optimizes the parameters of `model.head` while the underlying backbone remains
    frozen and evaluated without gradient tracking.

    Args:
        model: The taxonomic classifier containing frozen backbone and head.
        dataloader: Iterable yielding `(inputs, labels)` where `inputs` is a 4D batch
            tensor of images and `labels` is a 1D tensor of targets.
        optimizer: PyTorch optimizer targeting the head parameters.
        criterion: Differentiable loss function (e.g. ClassBalancedLoss).

    Returns:
        The mean training loss across all processed batches in the epoch.
    """
    model.train()
    model.backbone.eval()
    total_loss = 0.0
    num_batches = 0

    for inputs, labels in dataloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item())
        num_batches += 1

    return total_loss / max(1, num_batches)


def evaluate_head(
    model: TaxonClassifier,
    dataloader: Iterable[tuple[torch.Tensor, torch.Tensor]],
    criterion: nn.Module,
) -> tuple[float, float]:
    """Evaluate classifier performance on a validation or test dataloader.

    Args:
        model: The taxonomic classifier to evaluate.
        dataloader: Batches of validation data yielding `(inputs, labels)`.
        criterion: Loss function to compute validation loss.

    Returns:
        A pair `(val_loss, val_accuracy)` where `val_loss` is the average batch
        loss and `val_accuracy` is top-1 accuracy in $[0.0, 1.0]$.
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total_samples = 0
    num_batches = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += float(loss.item())
            preds = torch.argmax(outputs, dim=1)
            correct += int((preds == labels).sum().item())
            total_samples += labels.size(0)
            num_batches += 1

    avg_loss = total_loss / max(1, num_batches)
    accuracy = correct / max(1, total_samples)
    return avg_loss, accuracy
