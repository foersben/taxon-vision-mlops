# SPDX-FileCopyrightText: 2026 Benjamin Förster
# SPDX-License-Identifier: MIT
"""Active Learning Query Strategies (Margin, Entropy, BADGE)."""

from __future__ import annotations

import numpy as np


def margin_sampling(probs: np.ndarray) -> np.ndarray:
    """Computes margin score: difference between top 2 highest probabilities."""
    sorted_probs = np.sort(probs, axis=-1)
    margin = sorted_probs[:, -1] - sorted_probs[:, -2]
    return 1.0 - margin  # Higher score = higher uncertainty
