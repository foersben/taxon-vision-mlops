# SPDX-FileCopyrightText: 2026 Benjamin Förster
# SPDX-License-Identifier: MIT
"""Calibration pipeline for Tier 1 representation drift threshold.

Summary:
    Calibrates the 1D Wasserstein-1 drift threshold (tau_W1) using bootstrap null-distribution simulation across in-distribution holdout embedding partitions.

Why (Architectural Rationale & Trade-offs):
    Hardcoded drift thresholds (such as an arbitrary tau = 0.35) fail across different vision backbones whose activation magnitudes vary by orders of magnitude. Setting a threshold too low causes severe false positive churn on healthy in-distribution traffic; setting it too high masks genuine domain shifts. Bootstrapping null distributions from pristine in-distribution partitions (cal_embeddings.pt vs test_embeddings.pt) establishes an empirical, mathematically grounded threshold corresponding to a defensible false-alarm rate (e.g. 1% under the 99th percentile).

How (Algorithmic Mechanism & Invariants):
    1. Deserializes reference calibration embeddings (data/embeddings/cal_embeddings.pt) and in-distribution holdout embeddings (data/embeddings/test_embeddings.pt).
    2. Computes the pre-normalized Euclidean vector norms for both populations.
    3. Simulates B bootstrap draws (default: 1000) matching the operational production sliding window size, evaluating exact 1D Wasserstein-1 distances against the reference norm distribution.
    4. Computes the empirical (1 - alpha_drift) quantile (e.g. 99th percentile) as the calibrated threshold tau_W1.
    5. Calculates the cryptographic SHA256 digest of the reference embedding artifact to guarantee end-to-end data lineage tracking.
    6. Persists calibrated thresholds and metadata to config/drift_config.yaml.

Complexity:
    O(B * (N * log N)) where B is the number of bootstrap iterations and N is the sliding window size.

Standard Fields:
    References: Davison & Hinkley (1997) "Bootstrap Methods and Their Application", Cambridge University Press.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from scipy.stats import wasserstein_distance

from taxon_vision.config import get_settings

logger = logging.getLogger(__name__)


def _compute_file_sha256(path: Path) -> str:
    """Calculate the cryptographic SHA256 hex digest of a file on disk.

    Args:
        path: Path to the file to compute the SHA256 hex digest of.

    Returns:
        SHA256 hex digest of the file.
    """
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def _load_embedding_norms(path: Path) -> np.ndarray:
    """Load an embedding partition from disk and extract 1D Euclidean activation norms.

    Args:
        path: Path to the embedding partition file.

    Returns:
        1D array of embedding norms.
    """
    if not path.exists():
        raise FileNotFoundError(f"Embedding partition file not found: {path}")

    data = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(data, dict) and "embeddings" in data:
        raw_tensor = data["embeddings"]
    elif isinstance(data, torch.Tensor):
        raw_tensor = data
    else:
        raise ValueError(f"Unsupported embedding format in {path}")

    if isinstance(raw_tensor, torch.Tensor):
        embs = raw_tensor.detach().cpu().numpy()
    else:
        embs = np.asarray(raw_tensor)

    norms = np.linalg.norm(embs, axis=-1).ravel()
    if norms.size == 0:
        raise ValueError(f"Embedding partition in {path} contains zero records")
    return norms.astype(np.float64)


def _bootstrap_null_w1_distribution(
    ref_norms: np.ndarray,
    test_norms: np.ndarray,
    *,
    window_size: int,
    n_bootstraps: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Simulate null-hypothesis W1 distances between reference and random in-distribution subsets.

    Args:
        ref_norms: 1D array of reference embedding norms.
        test_norms: 1D array of test embedding norms.
        window_size: Nominal sample count of production sliding window.
        n_bootstraps: Number of bootstrap iterations for empirical null distribution simulation.
        rng: Random number generator.

    Returns:
        1D array of W1 distances between reference and bootstrapped test samples.
    """
    sample_size = min(window_size, test_norms.shape[0])
    replace = test_norms.shape[0] < window_size

    distances = np.empty(n_bootstraps, dtype=np.float64)
    for i in range(n_bootstraps):
        bootstrap_sample = rng.choice(test_norms, size=sample_size, replace=replace)
        distances[i] = wasserstein_distance(ref_norms, bootstrap_sample)

    return distances


def calibrate_drift_threshold(
    cal_path: Path,
    test_path: Path,
    *,
    window_size: int = 500,
    n_bootstraps: int = 1000,
    percentile: float = 99.0,
    alpha: float = 0.01,
    seed: int = 42,
) -> dict[str, Any]:
    """Execute bootstrap null-distribution calibration and emit structured drift configuration parameters.

    Args:
        cal_path: Path to reference calibration embeddings.
        test_path: Path to holdout in-distribution test embeddings.
        window_size: Nominal sample count of production sliding window.
        n_bootstraps: Number of bootstrap iterations for empirical null distribution simulation.
        percentile: Statistical percentile for threshold selection (e.g. 99.0 for 1% false alarm rate).
        alpha: Maximum significance level for MMD permutation hypothesis test.
        seed: Random seed for deterministic bootstrap sampling.

    Returns:
        Dictionary containing calibrated drift threshold, sample window sizes, and lineage hashes.
    """
    rng = np.random.default_rng(seed)
    ref_norms = _load_embedding_norms(cal_path)
    test_norms = _load_embedding_norms(test_path)

    boot_distances = _bootstrap_null_w1_distribution(
        ref_norms,
        test_norms,
        window_size=window_size,
        n_bootstraps=n_bootstraps,
        rng=rng,
    )

    tau_w1 = float(np.percentile(boot_distances, percentile))
    null_median = float(np.median(boot_distances))
    null_max = float(np.max(boot_distances))
    ref_sha256 = _compute_file_sha256(cal_path)

    effective_window = min(window_size, test_norms.shape[0])

    return {
        "drift": {
            "tau_w1": round(tau_w1, 6),
            "alpha": float(alpha),
            "window_size": effective_window,
            "n_bootstraps": n_bootstraps,
            "percentile": float(percentile),
            "null_median_w1": round(null_median, 6),
            "null_max_w1": round(null_max, 6),
            "reference_source": str(cal_path),
            "reference_sha256": ref_sha256,
            "test_source": str(test_path),
            "calibrated_at": datetime.now(UTC).isoformat(),
        }
    }


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for drift threshold calibration pipeline.

    Examples:
        >>> python -m taxon_vision.pipelines.calibrate_drift \
        ...     --cal-path data/embeddings/cal_embeddings.pt \
        ...     --test-path data/embeddings/test_embeddings.pt \
        ...     --output-yaml config/drift_config.yaml \
        ...     --window-size 500 \
        ...     --n-bootstraps 1000 \
        ...     --percentile 99.0 \
        ...     --alpha 0.01 \
        ...     --seed 42

    Args:
        argv: List of command-line arguments.

    Returns:
        Exit code (0 on success, non-zero on failure).
    """
    parser = argparse.ArgumentParser(description="Calibrate Tier 1 Wasserstein-1 drift threshold.")
    parser.add_argument(
        "--cal-path",
        type=Path,
        default=Path("data/embeddings/cal_embeddings.pt"),
        help="Path to baseline reference calibration embeddings.",
    )
    parser.add_argument(
        "--test-path",
        type=Path,
        default=Path("data/embeddings/test_embeddings.pt"),
        help="Path to in-distribution holdout test embeddings.",
    )
    parser.add_argument(
        "--output-yaml",
        type=Path,
        default=Path("config/drift_config.yaml"),
        help="Path to destination drift configuration YAML file.",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=None,
        help="Sliding window size (defaults to configured drift.window_size).",
    )
    parser.add_argument(
        "--n-bootstraps",
        type=int,
        default=1000,
        help="Number of bootstrap iterations (default: 1000).",
    )
    parser.add_argument(
        "--percentile",
        type=float,
        default=99.0,
        help="Percentile for threshold cut-off (default: 99.0).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Deterministic random seed (default: 42).",
    )

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", force=True)

    settings = get_settings()
    window_size = args.window_size if args.window_size is not None else settings.drift.window_size

    logger.info("Starting drift threshold calibration (reference: %s, test: %s)", args.cal_path, args.test_path)
    try:
        config_payload = calibrate_drift_threshold(
            args.cal_path,
            args.test_path,
            window_size=window_size,
            n_bootstraps=args.n_bootstraps,
            percentile=args.percentile,
            alpha=settings.drift.alpha,
            seed=args.seed,
        )
    except Exception as exc:
        logger.exception("Drift threshold calibration failed: %s", exc)
        return 1

    args.output_yaml.parent.mkdir(parents=True, exist_ok=True)
    with args.output_yaml.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config_payload, f, sort_keys=False)

    drift_meta = config_payload["drift"]
    logger.info(
        "Calibration successful: tau_W1=%.6f (null median=%.6f, max=%.6f) -> written to %s",
        drift_meta["tau_w1"],
        drift_meta["null_median_w1"],
        drift_meta["null_max_w1"],
        args.output_yaml,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
