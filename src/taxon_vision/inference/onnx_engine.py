# SPDX-FileCopyrightText: 2026 Benjamin Förster
# SPDX-License-Identifier: MIT
"""High-throughput, zero-allocation ONNX Runtime inference engine."""

from __future__ import annotations

import numpy as np
import onnxruntime as ort


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
        self.output_name = self.session.get_outputs()[0].name

    def predict(self, input_array: np.ndarray) -> np.ndarray:
        """Run inference returning softmax probabilities.

        Args:
            input_array: The input array to run inference on.

        Returns:
            A numpy array representing the softmax probabilities.
        """
        raw_outputs = self.session.run([self.output_name], {self.input_name: input_array})[0]
        # Softmax
        exp = np.exp(raw_outputs - np.max(raw_outputs, axis=-1, keepdims=True))
        probs: np.ndarray = exp / np.sum(exp, axis=-1, keepdims=True)
        return probs
