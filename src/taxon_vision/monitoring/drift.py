# SPDX-FileCopyrightText: 2026 Benjamin Förster
# SPDX-License-Identifier: MIT
"""Embedding and concept drift detection."""

from __future__ import annotations

import numpy as np


def compute_wasserstein_drift(reference_embeddings: np.ndarray, current_embeddings: np.ndarray) -> float:
    """1D Wasserstein distance approximation on embedding norm.

    Args:
        reference_embeddings: The reference embeddings parameter.
        current_embeddings: The current embeddings parameter.

    Returns:
        The resulting value from the operation.
    """
    ref_norm = np.linalg.norm(reference_embeddings, axis=-1)
    curr_norm = np.linalg.norm(current_embeddings, axis=-1)
    return float(np.abs(np.mean(ref_norm) - np.mean(curr_norm)))
