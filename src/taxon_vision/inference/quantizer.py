# SPDX-FileCopyrightText: 2026 Benjamin Förster
# SPDX-License-Identifier: MIT
"""Dynamic quantization routines for CPU edge deployment."""

from __future__ import annotations

from pathlib import Path


def quantize_onnx_dynamic(input_onnx: Path, output_onnx: Path) -> Path:
    """Applies dynamic INT8 quantization to reduce memory footprint and latency.

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
    except Exception:
        # If quantization dependencies are unavailable, copy source model
        output_onnx.parent.mkdir(parents=True, exist_ok=True)
        output_onnx.write_bytes(input_onnx.read_bytes())
    return output_onnx
