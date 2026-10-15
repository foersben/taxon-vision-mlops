# SPDX-FileCopyrightText: 2026 Benjamin Förster
# SPDX-License-Identifier: MIT
"""Static (QDQ) and dynamic INT8 ONNX quantization with explicit operator-aware mode selection.

Mode selection rule (Strategy Report Section 8.1):

* **Static PTQ** (`quantize_static`, `QuantFormat.QDQ`, MinMax calibration) is mandatory for graphs that contain
  convolutions or vision-transformer blocks (MobileNetV4, EfficientNet, DINOv3, BioCLIP-2, YOLO-World). Activation
  scales and zero points are calibrated offline on representative images, so no runtime min/max reduction is needed
  and `Conv` becomes a true INT8 kernel. `quantize_dynamic` does not quantize `Conv` usefully.
* **Dynamic quantization** (`quantize_dynamic`) is reserved for purely linear graphs (a standalone classification
  head, VLM linear projections) and for data-free CI smoke tests. Only `MatMul` and `Gemm` are quantized.

In both modes the node(s) producing the graph outputs (the logits) are excluded from quantization. Energy-based
OOD scoring and split-conformal non-conformity scores consume the logits directly; an 8-bit logit grid would
introduce staircase artefacts into both. Conformal quantiles and OOD thresholds must be re-calibrated on the
quantized graph.
"""

from __future__ import annotations

import logging
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Literal

import onnx
import onnxruntime.quantization as ort_quant
from onnxruntime.quantization import (
    CalibrationDataReader,
    CalibrationMethod,
    QuantFormat,
    QuantType,
)
from onnxruntime.quantization.shape_inference import quant_pre_process

logger = logging.getLogger(__name__)

QuantMode = Literal["auto", "static", "dynamic"]
"""Quantization mode: `auto` selects static when calibration data is provided, otherwise dynamic."""

STATIC_OP_TYPES: tuple[str, ...] = ("Conv", "MatMul", "Gemm")
"""Operators quantized by static PTQ. Normalisation, softmax and activations stay in FP32 for ViT accuracy."""

DYNAMIC_OP_TYPES: tuple[str, ...] = ("MatMul", "Gemm")
"""Operators quantized by dynamic quantization. `Conv` is deliberately excluded (no useful INT8 kernel)."""


class TaxonCalibrationDataReader(CalibrationDataReader):  # type: ignore[misc]
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


def _find_upstream_matmul(add_node: Any, producers: dict[str, Any]) -> str | None:
    """Find upstream MatMul producer feeding an Add bias operation.

    Why:
        Linear heads in PyTorch/ONNX typically decompose into MatMul + Add (bias).
        Locating the upstream MatMul ensures the entire projection head remains unquantized.

    Args:
        add_node: ONNX NodeProto representing the bias Add operation.
        producers: Mapping of output tensor names to producing node protos.

    Returns:
        Name of the upstream MatMul node if found, otherwise None.
    """
    for inp in add_node.input:
        upstream = producers.get(inp)
        if upstream is not None and upstream.op_type == "MatMul":
            return str(upstream.name)
    return None


def find_logit_nodes(model_path: Path) -> list[str]:
    """Locate the nodes that produce the graph outputs (the classification logits).

    Why:
        Quantizing the final linear layer discretises the logits onto an 8-bit grid. Both the Helmholtz free
        energy score and the conformal non-conformity scores are smooth functions of the logits, so a staircase
        logit distribution biases OOD thresholds and conformal quantiles.

    How:
        Collects the producers of every graph output. When a producer is an `Add` (bias added after a `MatMul`),
        the `MatMul` feeding it is included so the whole affine head stays in floating point.

    Complexity:
        O(N) where N is the number of nodes in the ONNX graph.

    Args:
        model_path: Path to the FP32 ONNX model.

    Returns:
        Names of the nodes to exclude from quantization.
    """
    graph = onnx.load(str(model_path), load_external_data=False).graph
    producers = {out: node for node in graph.node for out in node.output}
    excluded: list[str] = []

    for graph_output in graph.output:
        node = producers.get(graph_output.name)
        if node is None:
            continue
        excluded.append(node.name)
        if node.op_type == "Add":
            upstream_name = _find_upstream_matmul(node, producers)
            if upstream_name is not None:
                excluded.append(upstream_name)
    return excluded


def _preprocess(input_onnx: Path, workdir: Path) -> Path:
    """Run ORT's quantization pre-processing (shape inference, constant folding) and return the model path.

    Args:
        input_onnx: FP32 model to prepare.
        workdir: Temporary directory for the pre-processed model.

    Returns:
        Path to the pre-processed model, or the original path if pre-processing is not applicable.
    """
    prepared = workdir / "preprocessed.onnx"
    try:
        quant_pre_process(str(input_onnx), str(prepared), skip_symbolic_shape=True)
    except Exception as exc:
        # Pre-processing is an optimisation only; the numerical result is unaffected when it is skipped.
        logger.warning("Quantization pre-processing skipped (%s); quantizing the model as exported.", exc)
        return input_onnx
    return prepared


def quantize_onnx_static(
    input_onnx: Path,
    output_onnx: Path,
    calibration_reader: CalibrationDataReader,
    per_channel: bool = True,
    nodes_to_exclude: list[str] | None = None,
) -> Path:
    """Apply static INT8 Post-Training Quantization (QDQ, MinMax) to a convolutional or ViT graph.

    Why:
        Static PTQ precomputes quantization scales for activations, enabling INT8 execution without runtime min/max scanning. This is required for Conv2D and ViT layers to meet sub-25ms latency budgets, and keeps the activation scales tied to the exact data distribution used for conformal calibration.

    How:
        Runs ONNX Runtime's `quantize_static` with `QuantFormat.QDQ`, unsigned 8-bit activations and signed 8-bit per-channel weights, MinMax calibration, and only `Conv`/`MatMul`/`Gemm` quantized. The logit-producing nodes are excluded automatically so the output stays FP32.

    Args:
        input_onnx: The input FP32 ONNX model to quantize.
        output_onnx: The path to output the quantized model to.
        calibration_reader: Reader yielding representative batches for scaling.
        per_channel: Whether to use per-channel weight scales.
        nodes_to_exclude: Additional node names to keep in FP32 (the logit nodes are always excluded).

    Returns:
        The path to the statically quantized model.

    Raises:
        RuntimeError: If ONNX Runtime fails to quantize the model. The FP32 source is never copied silently.
    """
    output_onnx.parent.mkdir(parents=True, exist_ok=True)
    excluded = sorted(set(find_logit_nodes(input_onnx)) | set(nodes_to_exclude or []))
    try:
        with tempfile.TemporaryDirectory() as tmp:
            prepared = _preprocess(input_onnx, Path(tmp))
            ort_quant.quantize_static(
                model_input=str(prepared),
                model_output=str(output_onnx),
                calibration_data_reader=calibration_reader,
                quant_format=QuantFormat.QDQ,
                activation_type=QuantType.QUInt8,
                weight_type=QuantType.QInt8,
                per_channel=per_channel,
                calibrate_method=CalibrationMethod.MinMax,
                op_types_to_quantize=list(STATIC_OP_TYPES),
                nodes_to_exclude=excluded,
            )
    except Exception as exc:
        raise RuntimeError(f"Static INT8 quantization of {input_onnx} failed: {exc}") from exc
    logger.info("Static INT8 (QDQ, MinMax) model written to %s (FP32 nodes kept: %s)", output_onnx, excluded)
    return output_onnx


def quantize_onnx_dynamic(
    input_onnx: Path,
    output_onnx: Path,
    per_channel: bool = True,
    nodes_to_exclude: list[str] | None = None,
) -> Path:
    """Apply dynamic INT8 quantization to the linear layers of an ONNX graph.

    Why:
        Dynamic quantization needs no calibration data. It suits purely linear graphs (standalone head, VLM projections) whose activation ranges vary per token or per input, and data-free CI smoke tests. It does not quantize `Conv` operators, so it must not be used to accelerate convolutional backbones.

    How:
        Runs ONNX Runtime's `quantize_dynamic` with signed 8-bit weights restricted to `MatMul` and `Gemm`. Activation scales are computed at runtime by `DynamicQuantizeLinear`. The logit-producing nodes are excluded so the output head stays FP32.

    Complexity:
        - time complexity: O(M), where M is the number of nodes in the graph.
        - space complexity: O(M), for the in-memory ONNX graph representation.

    Args:
        input_onnx: The input FP32 ONNX model to quantize.
        output_onnx: The path to output the quantized model to.
        per_channel: Whether to use per-channel weight scales.
        nodes_to_exclude: Additional node names to keep in FP32 (the logit nodes are always excluded).

    Returns:
        The path to the dynamically quantized model.

    Raises:
        RuntimeError: If ONNX Runtime fails to quantize the model. The FP32 source is never copied silently.
    """
    output_onnx.parent.mkdir(parents=True, exist_ok=True)
    excluded = sorted(set(find_logit_nodes(input_onnx)) | set(nodes_to_exclude or []))
    try:
        ort_quant.quantize_dynamic(
            model_input=input_onnx,
            model_output=output_onnx,
            weight_type=QuantType.QInt8,
            per_channel=per_channel,
            op_types_to_quantize=list(DYNAMIC_OP_TYPES),
            nodes_to_exclude=excluded,
        )
    except Exception as exc:
        raise RuntimeError(f"Dynamic INT8 quantization of {input_onnx} failed: {exc}") from exc
    logger.info("Dynamic INT8 model written to %s (FP32 nodes kept: %s)", output_onnx, excluded)
    return output_onnx


def quantize_onnx_model(
    input_onnx: Path,
    output_onnx: Path,
    calibration_tensors: Iterable[Any] | None = None,
    mode: QuantMode = "auto",
    per_channel: bool = True,
    input_name: str = "input",
) -> Path:
    """Quantize an ONNX model, choosing static or dynamic INT8 explicitly.

    Why:
        Static and dynamic quantization are not interchangeable: static is needed for Conv and ViT graphs, dynamic only suits linear graphs and data-free smoke tests. A single entry point with an explicit mode makes the choice auditable and prevents silently producing a poorly accelerated model.

    How:
        `auto` resolves to static when calibration tensors are supplied and to dynamic otherwise. `static` requires calibration tensors. `dynamic` ignores them.

    Args:
        input_onnx: The input FP32 ONNX model.
        output_onnx: Destination of the quantized model.
        calibration_tensors: Pre-processed `(1, 3, H, W)` float32 arrays for static calibration.
        mode: One of `auto`, `static` or `dynamic`.
        per_channel: Whether to use per-channel weight scales.
        input_name: Name of the ONNX input node used to key calibration batches.

    Returns:
        Path to the quantized model.

    Raises:
        ValueError: If `static` is requested without calibration data, or the mode is unknown.
    """
    if mode not in ("auto", "static", "dynamic"):
        raise ValueError(f"Unknown quantization mode '{mode}'. Expected 'auto', 'static' or 'dynamic'.")

    batches: list[Any] = list(calibration_tensors) if calibration_tensors is not None else []
    resolved: Literal["static", "dynamic"]
    if mode == "auto":
        resolved = "static" if batches else "dynamic"
    else:
        resolved = mode

    if resolved == "static":
        if not batches:
            raise ValueError("Static quantization requires calibration tensors; pass data or use mode='dynamic'.")
        reader = TaxonCalibrationDataReader(batches, input_name=input_name)
        return quantize_onnx_static(input_onnx, output_onnx, reader, per_channel=per_channel)

    return quantize_onnx_dynamic(input_onnx, output_onnx, per_channel=per_channel)
