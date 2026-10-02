# SPDX-FileCopyrightText: 2026 Benjamin Förster
# SPDX-License-Identifier: MIT
"""Rapid CPU/GPU head training loop."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim

from taxon_vision.models.head import TaxonClassifier


def train_head_epoch(
    model: TaxonClassifier,
    dataloader: list[tuple[torch.Tensor, torch.Tensor]],
    optimizer: optim.Optimizer,
    criterion: nn.Module,
) -> float:
    """Run single training epoch."""
    model.train()
    total_loss = 0.0
    for inputs, labels in dataloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item())
    return total_loss / max(1, len(dataloader))
