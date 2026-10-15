# SPDX-FileCopyrightText: 2026 Benjamin Förster
# SPDX-License-Identifier: MIT
"""High-throughput, zero-allocation ONNX Runtime inference engine."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import onnxruntime as ort


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
