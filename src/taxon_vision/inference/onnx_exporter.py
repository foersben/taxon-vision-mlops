# SPDX-FileCopyrightText: 2026 Benjamin Förster
# SPDX-License-Identifier: MIT
"""Export PyTorch model to ONNX format."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn


def export_to_onnx(model: nn.Module, output_path: Path, input_shape: tuple[int, ...] = (1, 3, 224, 224)) -> Path:
    """Export model to ONNX graph with dynamic batch sizing.

    Why:
        In production workflows, the ML model must be decoupled from the specific deep learning framework used during training to maximize deployment flexibility. ONNX (Open Neural Network Exchange) serves as the de facto open-standard interoperability layer, enabling the model to be deployed using high-performance inference engines (e.g., ONNX Runtime, TensorRT) across diverse hardware targets (CPUs, GPUs, edge accelerators) without requiring the original training framework. This separation ensures the inference infrastructure can be optimized independently of the research and development environment.

    How:
        Exports the trained PyTorch model to a self-contained ONNX graph using torch.onnx.export. The export process traces the computational graph using a dummy input tensor, generating a serialized graph representation. Key optimization arguments include enabling constant folding to fuse operations (e.g., scaling, bias addition) into single nodes, specifying dynamic axes to permit variable batch sizes during inference, and selecting an opset version (e.g., 17) that balances feature availability with broad runtime compatibility.

    Complexity:
        - time complexity: O(M), where M is the number of layers in the model. The export process traces the computational graph once, performing constant folding and other optimizations that scale linearly with model size.
        - space complexity: O(M), due to the storage requirements for the ONNX graph representation and intermediate symbolic tensors during tracing.

    Args:
        model: The model to export.
        output_path: The path to export the model to.
        input_shape: The shape of the input to the model.

    Returns:
        The path to the exported model.
    """
    model.eval()
    dummy_input = torch.randn(*input_shape)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        (dummy_input,),
        str(output_path),
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch_size"}, "logits": {0: "batch_size"}},
    )
    return output_path
