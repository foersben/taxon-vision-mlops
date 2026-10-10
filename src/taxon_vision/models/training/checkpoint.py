# SPDX-FileCopyrightText: 2026 Benjamin Förster
# SPDX-License-Identifier: MIT
"""Checkpoint persistence utilities for model training."""

from __future__ import annotations

import contextlib
from pathlib import Path

import torch
from torch import nn

from taxon_vision.config import get_settings
from taxon_vision.service.inference import get_classifier


def _save_checkpoint(head: nn.Module, checkpoint_path: Path | str | None) -> Path:
    """Persist trained head weights and invalidate runtime classifier cache.

    Why:
        Persisting model weights to disk ensures deployment permanence and enables fast restoration during service cold starts. Clearing the runtime inference classifier cache (`get_classifier.cache_clear()`) ensures subsequent inference requests immediately consume the updated weights without requiring a service restart.

    How:
        Resolves target filesystem destination, creates parent directories if missing, serializes state dictionary via `torch.save`, and calls `cache_clear()` on `get_classifier`.

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

    with contextlib.suppress(Exception):
        get_classifier.cache_clear()

    return target_path
