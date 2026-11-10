# SPDX-FileCopyrightText: 2026 Benjamin Förster
# SPDX-License-Identifier: MIT
"""Embedding and concept drift detection algorithms.

Summary:
    Implements a two-tier statistical drift detection framework combining 1D Wasserstein-1 (Earth Mover's) distance on representation norms and multivariate Maximum Mean Discrepancy (MMD) with RBF kernels and permutation testing.

Why (Architectural Rationale & Trade-offs):
    Online biodiversity vision systems experience continuous distribution shifts across both visual domains (illumination, blur, weather) and taxonomic populations (seasonal emergence, regional migrations). A single scalar projection like 1D Wasserstein-1 on feature norms is computationally fast (sub-millisecond) for continuous telemetry, but remains invariant to orthogonal rotations and angular manifold shifts. Multivariate kernel Maximum Mean Discrepancy (MMD) operates in the full feature space to capture subtle covariance and semantic shifts, providing a statistically sound permutation p-value for automated retraining triggers and candidate model exchangeability gating.

How (Algorithmic Mechanism & Invariants):
    Tier 1 projects pre-normalised activations to Euclidean norms and computes exact 1D Wasserstein-1 distance via sorted empirical CDF differences. Tier 2 evaluates an unbiased two-sample MMD squared estimator using an RBF kernel with bandwidth determined by the median heuristic across pairwise distances. Exact finite-sample p-values are calculated through Monte Carlo permutation of the pooled Gram matrix, avoiding asymptotic distribution approximations.

Standard Fields:
    References: Gretton et al. (2012) "A Kernel Two-Sample Test", JMLR; Davison & Hinkley (1997) "Bootstrap Methods and Their Application".
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy.stats import wasserstein_distance


@dataclass(frozen=True)
class MMDResult:
    """Outcome of a multivariate two-sample Maximum Mean Discrepancy squared permutation test.

    Attributes:
        mmd2: Unbiased sample estimate of MMD squared between reference and current populations.
        p_value: Finite-sample permutation test p-value under the null hypothesis of distribution equality.
        bandwidth: RBF kernel bandwidth parameter (sigma) selected via the median heuristic.
        n_reference: Number of reference observations evaluated after subsampling.
        n_current: Number of current observations evaluated after subsampling.
    """

    mmd2: float
    p_value: float
    bandwidth: float
    n_reference: int
    n_current: int


def compute_wasserstein_drift(reference_embeddings: np.ndarray, current_embeddings: np.ndarray) -> float:
    r"""Compute exact 1D Wasserstein-1 distance on pre-normalized representation embedding norms.

    Why:
        In continuous ecological vision streams, visual domain shifts (seasonal foliage variations, differing camera sensors, regional lighting) and concept drift (shifts in species relative abundances) degrade model reliability over time. High-dimensional multivariate statistical tests (e.g. MMD, KS tests over 1024 dimensions) suffer from the curse of dimensionality and high computational overhead. Projecting unnormalized penultimate feature activations onto their L2 vector norms and measuring the exact Wasserstein-1 (earth mover's) distance yields an efficient, distribution-sensitive scalar metric for detecting representation drift across production inference windows, capturing variance changes and multi-modal splits that first-moment mean differences miss.

        Critically, this metric requires unnormalized penultimate feature activations. If embeddings are unit-normalized ($L_2$ normalized to unit length $\|z\|_2 = 1.0$) prior to drift evaluation, all vector norms collapse identically to 1.0, rendering $W_1$ completely blind to domain shift ($|1.0 - 1.0| = 0.0$). Upstream feature extraction pipelines and ONNX exports therefore emit pre-normalized activations, leaving unit-normalization to downstream consumers (such as LanceDB cosine indexers) at point of ingestion.

    How:
        1. Evaluates L2 Euclidean vector norms along the feature dimension for both reference and current observation batches:
           \|z\|_2 = \sqrt{\sum_d z_d^2}
        2. Computes the exact 1D Kantorovich-Rubinstein Wasserstein-1 distance between the empirical CDFs of reference and current norms via `scipy.stats.wasserstein_distance`:
           W_1(P_{ref}, P_{curr}) = \int_{-\infty}^{\infty} |F_{ref}(u) - F_{curr}(u)| \, du
        3. Returns the scalar distance as a drift alert indicator.

    Complexity:
        O(N * D + N log N) where N is the number of embeddings and D is feature dimensionality.

    Args:
        reference_embeddings: 2D NumPy array (N_ref, D) of baseline reference pre-normalized representations.
        current_embeddings: 2D NumPy array (N_curr, D) of newly observed production pre-normalized representations.

    Returns:
        Scalar float representing the Wasserstein-1 drift distance.
    """
    ref_norm = np.linalg.norm(reference_embeddings, axis=-1).ravel()
    curr_norm = np.linalg.norm(current_embeddings, axis=-1).ravel()
    if ref_norm.size == 0 or curr_norm.size == 0:
        return 0.0
    return float(wasserstein_distance(ref_norm, curr_norm))


def _subsample(x: npt.NDArray[np.float64], max_samples: int, rng: np.random.Generator) -> npt.NDArray[np.float64]:
    """Subsample an embedding matrix to at most max_samples without replacement.

    Args:
        x: Embedding matrix of shape (N, D).
        max_samples: Maximum number of samples to subsample to.
        rng: Random number generator.

    Returns:
        Subsampled embedding matrix of shape (M, D) where M <= max_samples.
    """
    if x.shape[0] <= max_samples:
        return x
    return x[rng.choice(x.shape[0], size=max_samples, replace=False)]


def _mmd2_unbiased(kernel: npt.NDArray[np.float64], n_ref: int) -> float:
    """Compute the unbiased sample estimate of MMD squared from a precomputed Gram kernel matrix.

    Args:
        kernel: Gram kernel matrix of shape (N_ref + N_curr, N_ref + N_curr).
        n_ref: Number of reference samples.

    Returns:
        Unbiased estimate of MMD squared.
    """
    k_xx = kernel[:n_ref, :n_ref]
    k_yy = kernel[n_ref:, n_ref:]
    k_xy = kernel[:n_ref, n_ref:]
    m, n = k_xx.shape[0], k_yy.shape[0]
    term_xx = (k_xx.sum() - np.trace(k_xx)) / (m * (m - 1))
    term_yy = (k_yy.sum() - np.trace(k_yy)) / (n * (n - 1))
    return float(term_xx + term_yy - 2.0 * k_xy.mean())


def compute_mmd2_permutation(
    reference: np.ndarray,
    current: np.ndarray,
    *,
    max_samples: int = 500,
    n_permutations: int = 200,
    seed: int = 42,
) -> MMDResult:
    """Compute multivariate Maximum Mean Discrepancy squared with RBF kernel and permutation testing.

    Summary:
        Evaluates the distance between two multivariate embedding distributions in an RKHS and determines statistical significance via Monte Carlo permutations.

    Why (Architectural Rationale & Trade-offs):
        Single-sample metrics and scalar norm projections cannot detect angular manifold shifts or class-conditional composition changes. Multivariate MMD in a reproducing kernel Hilbert space with a characteristic RBF kernel detects any divergence between probability distributions P and Q. Coupling MMD with permutation testing yields an exact, distribution-free p-value without requiring asymptotic normality assumptions.

    How (Algorithmic Mechanism & Invariants):
        1. Subsamples reference and current embedding matrices to max_samples using a deterministic PRNG generator.
        2. Asserts at least 2 samples per group, raising ValueError otherwise.
        3. Stacks populations into a pooled matrix Z = [X; Y] and computes vectorised pairwise squared Euclidean distances:
           D_ij^2 = ||z_i - z_j||_2^2
        4. Calculates the RBF kernel bandwidth sigma using the median heuristic over off-diagonal distances:
           sigma = sqrt(0.5 * median(D_ij^2))
        5. Evaluates the Gram kernel matrix K = exp(-D^2 / (2 * sigma^2)).
        6. Computes the unbiased observed MMD^2 statistic excluding diagonal self-similarity.
        7. Evaluates n_permutations index shuffles of the Gram matrix slices, accumulating test statistics that exceed the observed value.
        8. Returns MMDResult with exact finite-sample p-value: (exceed + 1) / (n_permutations + 1).

    Complexity:
        O((M + N)^2 * D + B * (M + N)^2) where M, N <= max_samples, D is embedding dimension, and B is n_permutations.

    Args:
        reference: 2D array of baseline reference feature embeddings of shape (N_ref, D).
        current: 2D array of newly observed feature embeddings of shape (N_curr, D).
        max_samples: Maximum number of observations per group to evaluate (default: 500).
        n_permutations: Number of random index shuffles for empirical p-value estimation (default: 200).
        seed: Random seed for deterministic subsampling and permutation generation (default: 42).

    Returns:
        MMDResult containing observed MMD squared, empirical p-value, bandwidth, and sample counts.

    Raises:
        ValueError: If either group contains fewer than 2 samples after subsampling.
    """
    rng = np.random.default_rng(seed)
    x = _subsample(np.asarray(reference, dtype=np.float64), max_samples, rng)
    y = _subsample(np.asarray(current, dtype=np.float64), max_samples, rng)
    if x.shape[0] < 2 or y.shape[0] < 2:
        raise ValueError("MMD requires at least two samples per group")

    pooled = np.vstack([x, y])
    sq = np.einsum("ij,ij->i", pooled, pooled)
    dist2 = np.maximum(sq[:, None] + sq[None, :] - 2.0 * (pooled @ pooled.T), 0.0)
    off_diag = dist2[~np.eye(pooled.shape[0], dtype=bool)]

    med = float(np.median(off_diag)) if off_diag.size > 0 else 1.0
    bandwidth = float(np.sqrt(0.5 * med)) if med > 0.0 else 1.0
    kernel = np.exp(-dist2 / (2.0 * bandwidth**2))

    observed = _mmd2_unbiased(kernel, x.shape[0])
    exceed = 0
    for _ in range(n_permutations):
        perm = rng.permutation(pooled.shape[0])
        if _mmd2_unbiased(kernel[np.ix_(perm, perm)], x.shape[0]) >= observed:
            exceed += 1

    p_value = (exceed + 1) / (n_permutations + 1)
    return MMDResult(
        mmd2=observed,
        p_value=p_value,
        bandwidth=bandwidth,
        n_reference=x.shape[0],
        n_current=y.shape[0],
    )
