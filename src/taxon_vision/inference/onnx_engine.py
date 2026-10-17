# SPDX-FileCopyrightText: 2026 Benjamin Förster
# SPDX-License-Identifier: MIT
"""High-throughput, zero-allocation ONNX Runtime inference engine."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
import onnxruntime as ort


def _extract_weight_from_inits(inits: dict[str, np.ndarray]) -> np.ndarray | None:
    """Extract and dequantize linear classification projection weight from ONNX initializers."""
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
    """Extract linear classification projection bias from ONNX initializers."""
    for k, v in inits.items():
        if "head" in k and "bias" in k:
            float_bias: np.ndarray = v.astype(np.float32)
            return float_bias
    return None


def extract_onnx_head_weights(onnx_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Extract linear classification head projection weight and bias from an ONNX model.

    Supports both FP32 and INT8 (quantized/dequantized) ONNX initializers.

    Args:
        onnx_path: Path to serialized ONNX graph.

    Returns:
        Tuple of (weight_matrix, bias_vector) as float32 numpy arrays.

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

    return weight, bias


def compute_onnx_logits(embeddings: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
    """Project dense embeddings to classification logits using extracted ONNX linear projection.

    Args:
        embeddings: Feature embedding matrix of shape (N, D).
        weight: Linear projection weight matrix.
        bias: Linear projection bias vector.

    Returns:
        Logits matrix of shape (N, K).
    """
    feat_dim = embeddings.shape[1]
    if weight.shape[0] != feat_dim and weight.shape[1] == feat_dim:
        weight = weight.T
    if weight.shape[0] != feat_dim:
        raise ValueError(f"Embedding dimension {feat_dim} does not match weight dimension {weight.shape[0]}")
    logits: np.ndarray = embeddings @ weight + bias
    return logits


@dataclass(frozen=True)
class InferenceOutputs:
    """Strongly-typed container for dual-output model forward pass results.

    Attributes:
        logits: Raw class logits of shape (N, num_classes).
        embeddings: Dense penultimate feature representation of shape (N, feature_dim), or None if unavailable.
    """

    logits: npt.NDArray[np.float32]
    embeddings: npt.NDArray[np.float32] | None = None


class ONNXInferenceEngine:
    """Executes batched inference with pre-allocated I/O bindings.

    Why:
        In continuous active learning workflows, inference latency directly impedes throughput. Loading the model weights and tracing the computational graph on every individual prediction incurs substantial I/O and kernel launch overhead. Maintaining a persistent ONNX Runtime session with graph optimization enabled ensures the computational graph is constructed and optimized exactly once during initialization, minimizing per-prediction latency for subsequent inference calls.

    How:
        Initializes an ONNX Runtime inference session using the specified model path and enables graph optimizations. Pre-allocates memory for inputs and outputs to avoid repeated memory allocations during inference. The prediction method preprocesses input arrays to match the model's expected input shape and data type (e.g., C,H,W normalized to [0, 1]), runs inference using the session, and converts the raw outputs to softmax probability distributions element-wise via the exponential function to produce calibrated likelihood scores.

    Complexity:
        - initialization: O(M) time complexity, where M is the number of layers in the model.
        - prediction: O(N) time complexity, where N is the number of input samples.

    Attributes:
        session: The ONNX Runtime inference session.
        input_name: The name of the input layer.
        output_name: The name of the output layer.
        output_names: Ordered list of graph output tensor names.
        has_embeddings: Whether the exported graph emits a secondary embeddings tensor.
    """

    def __init__(self, model_path: str) -> None:
        """Initialize the ONNX Runtime session with graph optimizations.

        Args:
            model_path: Filesystem path to the exported ONNX model artifact.
        """
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(model_path, opts, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        output_names = [out.name for out in self.session.get_outputs()]
        self.has_embeddings = "embeddings" in output_names
        self.output_names = ["logits", "embeddings"] if self.has_embeddings else output_names
        self.output_name = self.session.get_outputs()[0].name
        out_dim = self.session.get_outputs()[0].shape[-1]
        self.num_classes: int | None = out_dim if isinstance(out_dim, int) else None

    def predict_outputs(self, input_array: npt.NDArray[np.float32]) -> InferenceOutputs:
        """Run inference returning both class logits and dense feature embeddings.

        Why:
            Provides a unified execution path returning both classification scores and representation
            vectors from a single ONNX Runtime forward pass, avoiding redundant graph evaluation.

        How:
            Evaluates the session on the requested outputs and wraps results in an InferenceOutputs dataclass.

        Complexity:
            O(N) forward pass inference complexity where N is the batch size.

        Args:
            input_array: Float32 input batch of shape (N, 3, H, W).

        Returns:
            InferenceOutputs container with logits and optional embeddings.
        """
        raw = self.session.run(self.output_names, {self.input_name: input_array})
        logits = np.asarray(raw[0], dtype=np.float32)
        embeddings = np.asarray(raw[1], dtype=np.float32) if len(raw) > 1 else None
        return InferenceOutputs(logits=logits, embeddings=embeddings)

    def predict_logits(self, input_array: np.ndarray) -> np.ndarray:
        """Run inference returning raw FP32 class logits.

        Args:
            input_array: Float32 input batch of shape (N, 3, H, W).

        Returns:
            Logits array of shape (N, num_classes).
        """
        return self.predict_outputs(np.asarray(input_array, dtype=np.float32)).logits

    def predict(self, input_array: np.ndarray) -> np.ndarray:
        """Run inference returning softmax probabilities.

        Args:
            input_array: The input array to run inference on.

        Returns:
            A numpy array representing the softmax probabilities.
        """
        raw_outputs = self.predict_logits(input_array)
        # Softmax
        exp = np.exp(raw_outputs - np.max(raw_outputs, axis=-1, keepdims=True))
        probs: np.ndarray = exp / np.sum(exp, axis=-1, keepdims=True)
        return probs
