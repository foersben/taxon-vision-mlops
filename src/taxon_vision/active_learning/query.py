# SPDX-FileCopyrightText: 2026 Benjamin Förster
# SPDX-License-Identifier: MIT
"""Active Learning Query Strategies (Margin, Entropy, BADGE)."""

from __future__ import annotations

import numpy as np


def margin_sampling(probs: np.ndarray) -> np.ndarray:
    """Compute active learning uncertainty scores via multi-class margin sampling.

    Why:
        In ecological observation streams, labeling budgets are severely constrained.
        Random sampling over-indexes on canonical, easy-to-identify species (e.g. domestic cats,
        common dandelions) while starving difficult sister taxa with subtle phenotypic differences.
        Margin sampling identifies samples where the decision boundary is most contested
        between the top two candidate classes, prioritizing observations that provide maximal
        gradient signal when reviewed and annotated by human taxonomists.

    How:
        For each row of predicted class probabilities p, sorts class probabilities along the
        last axis in ascending order, calculates the separation margin delta = p_(top1) - p_(top2),
        and returns the inverted uncertainty score 1.0 - delta. A score close to 1.0 indicates
        maximum ambiguity (p_(top1) approx p_(top2)), while a score close to 0.0 indicates decisive
        confidence (p_(top1) >> p_(top2)).

    Complexity:
        O(N * K log K) time complexity where N is the batch size and K is the number of classes.
        O(N * K) space complexity for array sorting.

    Args:
        probs: 2D NumPy array of shape (N, K) containing normalized class probabilities.

    Returns:
        1D NumPy array of length N containing scalar uncertainty scores in the interval [0.0, 1.0].
    """
    sorted_probs = np.sort(probs, axis=-1)
    margin = sorted_probs[:, -1] - sorted_probs[:, -2]
    result: np.ndarray = np.asarray(1.0 - margin)
    return result
