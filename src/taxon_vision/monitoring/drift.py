# SPDX-FileCopyrightText: 2026 Benjamin Förster
# SPDX-License-Identifier: MIT
"""Embedding and concept drift detection."""

from __future__ import annotations

import numpy as np


def compute_wasserstein_drift(reference_embeddings: np.ndarray, current_embeddings: np.ndarray) -> float:
    """1D Wasserstein distance approximation on embedding norm.

    The Wasserstein distance is a measure of the distance between two probability distributions.
    In this case, we are comparing the distribution of embedding norms of the reference embeddings
    with the distribution of embedding norms of the current embeddings.

    Args:
        reference_embeddings: The reference embeddings to compare against.
        current_embeddings: The current embeddings to compare against.

    Returns:
        The Wasserstein drift distance.
    """
    ref_norm = np.linalg.norm(reference_embeddings, axis=-1)
    curr_norm = np.linalg.norm(current_embeddings, axis=-1)
    return float(np.abs(np.mean(ref_norm) - np.mean(curr_norm)))
