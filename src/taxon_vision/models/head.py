# SPDX-FileCopyrightText: 2026 Benjamin Förster
# SPDX-License-Identifier: MIT
"""Trainable linear classification head on frozen Vision Foundation Model backbones.

This module decouples feature representation learning from taxonomic classification.
By freezing the deep vision transformer or CNN backbone and training only a lightweight
linear projection head with dropout regularization, model convergence is accelerated
to seconds or minutes on cached embeddings without risk of catastrophic forgetting
or gradient instability.

Typical usage example:
    backbone = create_feature_extractor("bioclip-2", pretrained=False)
    classifier = TaxonClassifier(backbone=backbone, feature_dim=512, num_classes=10)
    # Forward on raw images:
    logits = classifier(images)
    # Forward directly on pre-extracted embeddings:
    logits = classifier.forward_head(embeddings)
"""

from __future__ import annotations

import torch
import torch.nn as nn


class TaxonClassifier(nn.Module):
    r"""Frozen vision foundation backbone coupled with a trainable linear classification head.

    This architecture keeps all backbone weights static while optimizing a single linear
    transformation matrix $\\theta_{\\text{last}} \\in \\mathbb{R}^{D \\times K}$ with dropout
    regularization.

    Attributes:
        backbone: Frozen vision backbone producing $D$-dimensional representations.
        feature_dim: Dimensionality of the input feature vector ($D$).
        num_classes: Number of target taxonomic classes ($K$).
        head: Sequential head comprising Dropout followed by Linear projection.
    """

    def __init__(
        self,
        backbone: nn.Module,
        feature_dim: int,
        num_classes: int,
        dropout: float = 0.2,
    ) -> None:
        """Initialize the TaxonClassifier model.

        Args:
            backbone: Pre-trained and frozen feature extractor backbone.
            feature_dim: Number of input features yielded by the backbone extractor.
            num_classes: Number of target classification categories.
            dropout: Dropout probability applied prior to the final linear layer.
        """
        super().__init__()
        self.backbone = backbone
        self.feature_dim = feature_dim
        self.num_classes = num_classes

        # Ensure backbone parameters remain frozen
        for param in self.backbone.parameters():
            param.requires_grad = False

        self.head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(feature_dim, num_classes),
        )

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract feature embeddings from raw image tensors using the frozen backbone.

        Args:
            x: 4D image tensor of shape `(batch_size, 3, height, width)`.

        Returns:
            Feature embeddings of shape `(batch_size, feature_dim)`.
        """
        with torch.no_grad():
            features: torch.Tensor = self.backbone(x)
        return features

    def forward_head(self, features: torch.Tensor) -> torch.Tensor:
        """Compute taxonomic class logits directly from pre-computed feature embeddings.

        Allows near-instant training and hyperparameter search without repeatedly
        passing images through the deep vision backbone.

        Args:
            features: 2D embedding tensor of shape `(batch_size, feature_dim)`.

        Returns:
            Raw unnormalized class logits of shape `(batch_size, num_classes)`.
        """
        logits: torch.Tensor = self.head(features)
        return logits

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Execute end-to-end forward pass from input images to class logits.

        Extracts features via the frozen backbone and projects them through the linear head.

        Args:
            x: 4D input image tensor of shape `(batch_size, channels, height, width)`.

        Returns:
            Unnormalized class logits of shape `(batch_size, num_classes)`.
        """
        features = self.forward_features(x)
        return self.forward_head(features)
