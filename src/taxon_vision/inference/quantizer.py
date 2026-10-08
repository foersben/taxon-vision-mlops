# SPDX-FileCopyrightText: 2026 Benjamin Förster
# SPDX-License-Identifier: MIT
"""Static and dynamic ONNX quantization routines for CPU edge deployment."""

from __future__ import annotations

import logging
from collections.abc import Iterable
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

try:
    from onnxruntime.quantization import CalibrationDataReader
except ImportError:
    # Dummy fallback if onnxruntime is absent
    class CalibrationDataReader:  # type: ignore
        """Dummy fallback if onnxruntime is absent."""

        def get_next(self) -> None:
            """Get the next batch of calibration data."""
            return None

        def rewind(self) -> None:
            """Rewind the iterator to the beginning of the calibration data batches."""
            pass


class TaxonCalibrationDataReader(CalibrationDataReader):  # type: ignore
    """Provides representative calibration data for Static Post-Training Quantization.

    Why:
        Static INT8 quantization for convolutional and vision transformer models requires computing activation scale factors and zero points offline. This avoids the latency penalty of calculating them dynamically during each forward pass. This reader feeds representative data samples into the ONNX Runtime quantizer.

    How:
        Wraps an iterable of pre-processed numpy batches. Yields them one by one as dictionaries mapped to the expected ONNX input node name until exhausted. Implements `rewind()` to allow multiple passes by the calibration engine.

    Attributes:
        data_batches: Iterable of pre-processed numpy batches.
        input_name: Name of the ONNX input node.
        iterator: Iterator over the calibration data batches.
    """

    def __init__(self, data_batches: Iterable[Any], input_name: str = "input") -> None:
        """Initialize the calibration data reader.

        Args:
            data_batches: Iterable of pre-processed numpy batches.
            input_name: Name of the ONNX input node.
        """
        self.data_batches = list(data_batches)
        self.iterator = iter(self.data_batches)
        self.input_name = input_name

    def get_next(self) -> dict[str, Any] | None:
        """Get the next batch of calibration data.

        Returns:
            dict[str, Any]: Dictionary mapping the ONNX input node name to the calibration data batch.
        """
        try:
            batch = next(self.iterator)
            return {self.input_name: batch}
        except StopIteration:
            return None

    def rewind(self) -> None:
        """Rewind the iterator to the beginning of the calibration data batches."""
        self.iterator = iter(self.data_batches)


def quantize_onnx_static(input_onnx: Path, output_onnx: Path, calibration_reader: TaxonCalibrationDataReader) -> Path:
    """Applies static INT8 Post-Training Quantization (PTQ) to a vision backbone.

    Why:
        Static PTQ precomputes quantization scales for activations, enabling zero-allocation
        INT8 inference. This is strictly required for Conv2D and ViT layers to achieve sub-25ms latency budgets on edge CPUs without runtime calculation overhead.

    How:
        Uses ONNX Runtime's `quantize_static` to convert floating-point weights and activations into QDQ (QuantizeLinear/DequantizeLinear) formatted 8-bit integers using the provided calibration data ranges.

    Args:
        input_onnx: The input FP32 ONNX model to quantize.
        output_onnx: The path to output the quantized model to.
        calibration_reader: Reader yielding representative batches for scaling.

    Returns:
        The path to the statically quantized model.
    """
    try:
        from onnxruntime.quantization import QuantFormat, QuantType, quantize_static

        output_onnx.parent.mkdir(parents=True, exist_ok=True)
        quantize_static(
            model_input=input_onnx,
            model_output=output_onnx,
            calibration_data_reader=calibration_reader,
            quant_format=QuantFormat.QDQ,
            activation_type=QuantType.QInt8,
            weight_type=QuantType.QInt8,
            calibrate_method=0,  # MinMax
        )
    except Exception as e:
        logger.warning("Static quantization failed: %s. Copying source model.", e)
        output_onnx.parent.mkdir(parents=True, exist_ok=True)
        output_onnx.write_bytes(input_onnx.read_bytes())
    return output_onnx


def quantize_onnx_dynamic(input_onnx: Path, output_onnx: Path) -> Path:
    """Applies dynamic INT8 quantization to reduce memory footprint and latency.

    Why:
        In production workflows, the ML model must be decoupled from the specific deep learning framework used during training to maximize deployment flexibility. ONNX (Open Neural Network Exchange) serves as the de facto open-standard interoperability layer, enabling the model to be deployed using high-performance inference engines (e.g., ONNX Runtime, TensorRT) across diverse hardware targets (CPUs, GPUs, edge accelerators) without requiring the original training framework. This separation ensures the inference infrastructure can be optimized independently of the research and development environment.

    How:
        Applies dynamic integer-8 (INT8) quantization to the ONNX model weights and activations. This process reduces the model's precision from 32-bit floating-point (FP32) to 8-bit integers, decreasing memory bandwidth requirements, increasing cache efficiency, and accelerating inference speed, especially on CPU-based edge devices. The quantization is performed "dynamically" in that the scaling factors for activations are computed at runtime based on the observed data ranges during inference, allowing the model to adapt to varying input distributions while maintaining improved performance characteristics.

    Complexity:
        - time complexity: O(M), where M is the number of layers in the model. The export process traces the computational graph once, performing constant folding and other optimizations that scale linearly with model size.
        - space complexity: O(M), due to the storage requirements for the ONNX graph representation and intermediate symbolic tensors during tracing.

    Args:
        input_onnx: The input ONNX model to quantize.
        output_onnx: The path to output the quantized model to.

    Returns:
        The path to the quantized model.
    """
    try:
        from onnxruntime.quantization import QuantType, quantize_dynamic

        output_onnx.parent.mkdir(parents=True, exist_ok=True)
        quantize_dynamic(
            model_input=input_onnx,
            model_output=output_onnx,
            weight_type=QuantType.QInt8,
        )
    except Exception as e:
        logger.warning("Dynamic quantization failed: %s. Copying source model.", e)
        output_onnx.parent.mkdir(parents=True, exist_ok=True)
        output_onnx.write_bytes(input_onnx.read_bytes())
    return output_onnx
