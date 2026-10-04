#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 Benjamin Förster
# SPDX-License-Identifier: MIT
"""Mock script to run linear head training."""

import argparse
import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from taxon_vision.config import get_settings
from taxon_vision.models.training import (
    EmbeddingSplit,
    HeadTrainingConfig,
    train_head_on_cached_embeddings,
)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(description="Train TaxonClassifier Head")
    parser.add_argument("--extractor", type=str, default="mobilenetv4_conv_small")
    parser.add_argument("--epochs", type=int, default=5)
    args = parser.parse_args()

    settings = get_settings()
    num_classes = 10
    feature_dim = 1280  # Approximate generic feature dim for mobilenet

    # 1. Create a fresh classification head
    head = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(feature_dim, num_classes),
    )

    # 2. Generate Synthetic Training Data (Mocking extracted embeddings)
    N_SAMPLES = 500
    train_z = torch.randn(N_SAMPLES, feature_dim)
    train_y = torch.randint(0, num_classes, (N_SAMPLES,))

    optimizer = optim.AdamW(head.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    logger.info(f"Training linear head for {args.epochs} epochs on {N_SAMPLES} synthetic samples...")

    data = EmbeddingSplit(
        train_embeddings=train_z,
        train_labels=train_y,
        val_embeddings=train_z,
        val_labels=train_y,
    )
    training_config = HeadTrainingConfig(epochs=args.epochs, batch_size=32)

    history = train_head_on_cached_embeddings(
        head=head,
        data=data,
        optimizer=optimizer,
        criterion=criterion,
        config=training_config,
    )

    logger.info(f"Final Training Loss: {history['train_loss'][-1]:.4f}")

    # 3. Save the trained head weights
    out_path = Path(settings.model.head_checkpoint_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(head.state_dict(), out_path)
    logger.info(f"Successfully saved trained head to {out_path}")
