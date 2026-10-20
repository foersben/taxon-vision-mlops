# SPDX-FileCopyrightText: 2026 Benjamin Förster
# SPDX-License-Identifier: MIT
"""Embedding and concept drift detection."""

from __future__ import annotations

import numpy as np


def compute_wasserstein_drift(reference_embeddings: np.ndarray, current_embeddings: np.ndarray) -> float:
    r"""Compute 1D Wasserstein distance approximation on pre-normalized representation embedding norms.

    Why:
        In continuous ecological vision streams, visual domain shifts (seasonal foliage variations, differing camera sensors, regional lighting) and concept drift (shifts in species relative abundances) degrade model reliability over time. High-dimensional multivariate statistical tests (e.g. MMD, KS tests over 1024 dimensions) suffer from the curse of dimensionality and high computational overhead. Projecting unnormalized penultimate feature activations onto their L2 vector norms and measuring the Wasserstein-1 (earth mover's) distance yields an efficient, sensitive scalar metric for detecting representation drift across production inference windows.

        Critically, this metric requires unnormalized penultimate feature activations. If embeddings are unit-normalized ($L_2$ normalized to unit length $\|z\|_2 = 1.0$) prior to drift evaluation, all vector norms collapse identically to 1.0, rendering $W_1$ completely blind to domain shift ($|1.0 - 1.0| = 0.0$). Upstream feature extraction pipelines and ONNX exports therefore emit pre-normalized activations, leaving unit-normalization to downstream consumers (such as LanceDB cosine indexers) at point of ingestion.

    How:
        1. Evaluates L2 Euclidean vector norms along the feature dimension for both reference and current observation batches:
           \|z\|_2 = \sqrt{\sum_d z_d^2}
        2. Computes the absolute difference in mean norm distributions between reference and current sets:
           W_1(P_ref, P_curr) \approx | E[\|z_ref\|_2] - E[\|z_curr\|_2] |
        3. Returns the scalar distance as a drift alert indicator.

    Complexity:
        O(N * D) where N is the number of embeddings and D is feature dimensionality.

    Args:
        reference_embeddings: 2D NumPy array (N_ref, D) of baseline reference pre-normalized representations.
        current_embeddings: 2D NumPy array (N_curr, D) of newly observed production pre-normalized representations.

    Returns:
        Scalar float representing the estimated Wasserstein drift distance.
    """
    ref_norm = np.linalg.norm(reference_embeddings, axis=-1)
    curr_norm = np.linalg.norm(current_embeddings, axis=-1)
    return float(np.abs(np.mean(ref_norm) - np.mean(curr_norm)))
