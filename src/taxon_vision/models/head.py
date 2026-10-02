# SPDX-FileCopyrightText: 2026 Benjamin Förster
# SPDX-License-Identifier: MIT
"""Trainable classification head on frozen extractor."""

from __future__ import annotations

import torch
import torch.nn as nn


class TaxonClassifier(nn.Module):
    """Frozen backbone coupled with linear classification head."""

    def __init__(self, backbone: nn.Module, feature_dim: int, num_classes: int, dropout: float = 0.2) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(feature_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            features = self.backbone(x)
        logits: torch.Tensor = self.head(features)
        return logits
