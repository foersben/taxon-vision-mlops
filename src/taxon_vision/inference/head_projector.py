# SPDX-FileCopyrightText: 2026 Benjamin Förster
# SPDX-License-Identifier: MIT
"""ONNX linear classification head weight extraction and embedding projection."""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import numpy as np


class HeadWeights(NamedTuple):
    """Extracted linear classification head weight matrix and bias vector."""

    weight: np.ndarray
    bias: np.ndarray


def _extract_weight_from_inits(inits: dict[str, np.ndarray]) -> np.ndarray | None:
    """Extract and dequantize linear classification projection weight from ONNX initializers.

    Args:
        inits: Dictionary mapping ONNX initializer names to NumPy arrays.

    Returns:
        Dequantized or float32 weight matrix, or None if not found.
    """
    for k, v in inits.items():
        if "head" not in k or "weight" not in k:
            continue
        if "quantized" in k:
            scale = inits.get(k.replace("weight_quantized", "weight_scale"), np.array(1.0))
            zp = inits.get(k.replace("weight_quantized", "weight_zero_point"), np.array(0))
            dequantized: np.ndarray = (v.astype(np.float32) - zp) * scale
            return dequantized
        if "scale" not in k and "zero_point" not in k:
            float_weight: np.ndarray = v.astype(np.float32)
            return float_weight
    return None


def _extract_bias_from_inits(inits: dict[str, np.ndarray]) -> np.ndarray | None:
    """Extract linear classification projection bias from ONNX initializers.

    Args:
        inits: Dictionary mapping ONNX initializer names to NumPy arrays.

    Returns:
        Float32 bias vector, or None if not found.
    """
    for k, v in inits.items():
        if "head" in k and "bias" in k:
            float_bias: np.ndarray = v.astype(np.float32)
            return float_bias
    return None


def extract_onnx_head_weights(onnx_path: str | Path) -> HeadWeights:
    """Extract linear classification head projection weight and bias from an ONNX model.

    Why:
        Calibration (`calibrate.py`) and evaluation (`evaluate.py`) stages operate on pre-computed cached feature embeddings (`cal_embeddings.pt`, `test_embeddings.pt`) rather than re-running raw images through the full vision backbone. Extracting the terminal linear projection weights directly from the quantized ONNX graph ensures calibration and holdout evaluation reflect the exact serialized ONNX head parameters without instantiating PyTorch or repeating backbone inference.

    How:
        Parses ONNX initializers via `onnx.numpy_helper.to_array`, locates the classification head weight (dequantizing INT8 weights via scale and zero-point if quantized) and float32 bias vector, and returns them as a typed `HeadWeights` tuple.

    Complexity:
        O(I + D * K) where I is the number of ONNX graph initializers, D is feature dimension, and K is class count.

    Args:
        onnx_path: Path to serialized ONNX graph.

    Returns:
        HeadWeights of (weight, bias) as float32 numpy arrays.

    Raises:
        FileNotFoundError: If the ONNX file does not exist.
        ValueError: If head weight or bias initializers cannot be found in the graph.
    """
    path = Path(onnx_path)
    if not path.exists():
        raise FileNotFoundError(f"ONNX model file not found: {path}")

    import onnx
    from onnx import numpy_helper

    model = onnx.load(str(path))
    inits = {i.name: numpy_helper.to_array(i) for i in model.graph.initializer}

    weight = _extract_weight_from_inits(inits)
    bias = _extract_bias_from_inits(inits)

    if weight is None or bias is None:
        raise ValueError(f"Could not locate linear classification head weights/bias in ONNX graph: {path}")

    return HeadWeights(weight=weight, bias=bias)


def compute_onnx_logits(embeddings: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
    """Project dense embeddings to classification logits using extracted ONNX linear projection.

    Args:
        embeddings: Feature embedding matrix of shape (N, D).
        weight: Linear projection weight matrix.
        bias: Linear projection bias vector.

    Returns:
        Logits matrix of shape (N, K).

    Raises:
        ValueError: If embedding dimension does not match weight matrix input dimension.
    """
    feat_dim = embeddings.shape[1]
    if weight.shape[0] != feat_dim and weight.shape[1] == feat_dim:
        weight = weight.T
    if weight.shape[0] != feat_dim:
        raise ValueError(f"Embedding dimension {feat_dim} does not match weight dimension {weight.shape[0]}")
    logits: np.ndarray = embeddings @ weight + bias
    return logits
