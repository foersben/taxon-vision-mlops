# SPDX-FileCopyrightText: 2026 Benjamin Förster
# SPDX-License-Identifier: MIT
"""Standard training loops for taxonomic classification heads."""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import nn, optim

from taxon_vision.models.head import TaxonClassifier


def train_head_epoch(
    model: TaxonClassifier,
    dataloader: Iterable[tuple[torch.Tensor, torch.Tensor]],
    optimizer: optim.Optimizer,
    criterion: nn.Module,
) -> float:
    """Execute a single training epoch across batches of raw image inputs.

    Optimizes the parameters of `model.head` while the underlying backbone remains frozen and evaluated without gradient tracking.

    Why:
        During the transfer learning phase, the backbone (pre-trained on a large general dataset) captures robust low-level visual features (edges, textures, shapes). Freezing these parameters prevents the high-frequency noise and class imbalance of the specific target domain from corrupting the learned representations. Training only the head allows the model to specialize in mapping these existing features to the new taxonomic labels efficiently, avoiding catastrophic forgetting and accelerating convergence.

    How:
        Sets the `requires_grad` attribute to `False` for all parameters in `model.backbone` prior to the optimization step. During the forward pass, only the head's parameters are updated via standard backpropagation. The `torch.no_grad()` context manager or explicit parameter freezing ensures that gradient computations do not propagate into the backbone layers, saving memory and computation time.

    Complexity:
        - time complexity: O(T * B * C_feat), where T is the number of training steps, B is the batch size, and C_feat is the fixed computational cost of the backbone forward pass. The complexity is independent of the number of trainable parameters in the head, as the backbone computation is constant.
        - space complexity: O(M_head + B * C_feat), where M_head is the number of parameters in the head (typically small) and B * C_feat is the memory required for backbone activations. The space complexity is dominated by the backbone's feature map storage.

    Args:
        model: The taxonomic classifier containing frozen backbone and head.
        dataloader: Iterable yielding `(inputs, labels)` where `inputs` is a 4D batch tensor of images and `labels` is a 1D tensor of targets.
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
