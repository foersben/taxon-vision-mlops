# SPDX-FileCopyrightText: 2026 Benjamin Förster
# SPDX-License-Identifier: MIT
"""Export PyTorch model to ONNX format."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn


def export_to_onnx(model: nn.Module, output_path: Path, input_shape: tuple[int, ...] = (1, 3, 224, 224)) -> Path:
    """Export model to ONNX graph with dynamic batch sizing.

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
