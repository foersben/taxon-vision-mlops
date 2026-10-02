# SPDX-FileCopyrightText: 2026 Benjamin Förster
# SPDX-License-Identifier: MIT
"""High-throughput, zero-allocation ONNX Runtime inference engine."""

from __future__ import annotations

import numpy as np
import onnxruntime as ort


class ONNXInferenceEngine:
    """Executes batched inference with pre-allocated I/O bindings.

    Attributes:
        Various internal state and configuration variables used by the class.
    """

    def __init__(self, model_path: str) -> None:
        """Init  .

        Args:
            model_path: The model path parameter.
        """
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(model_path, opts, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def predict(self, input_array: np.ndarray) -> np.ndarray:
        """Run inference returning softmax probabilities.

        Args:
            input_array: The input array parameter.

        Returns:
            The resulting value from the operation.
        """
        raw_outputs = self.session.run([self.output_name], {self.input_name: input_array})[0]
        # Softmax
        exp = np.exp(raw_outputs - np.max(raw_outputs, axis=-1, keepdims=True))
        probs: np.ndarray = exp / np.sum(exp, axis=-1, keepdims=True)
        return probs
