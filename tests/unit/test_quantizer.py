# SPDX-FileCopyrightText: 2026 Benjamin Förster
# SPDX-License-Identifier: MIT
"""Unit tests for static and dynamic ONNX quantization routines."""

import importlib.util
import tempfile
from pathlib import Path

import numpy as np
import pytest

from taxon_vision.inference.quantizer import (
    TaxonCalibrationDataReader,
    quantize_onnx_dynamic,
    quantize_onnx_static,
)

try:
    import onnxruntime
    import torch
    import torch.nn as nn

    HAS_DEPS = importlib.util.find_spec("onnx") is not None
except ImportError:
    HAS_DEPS = False


class DummyVisionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(8 * 4 * 4, 2)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


@pytest.fixture
def dummy_onnx_model():
    if not HAS_DEPS:
        pytest.skip("Requires torch, onnx, onnxruntime")

    model = DummyVisionModel()
    model.eval()

    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = Path(tmpdir) / "dummy.onnx"
        dummy_input = torch.randn(1, 3, 4, 4)

        torch.onnx.export(
            model,
            dummy_input,
            str(input_path),
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        )
        yield input_path


def test_taxon_calibration_data_reader():
    batches = [np.random.randn(2, 3, 4, 4).astype(np.float32) for _ in range(3)]
    reader = TaxonCalibrationDataReader(batches, input_name="input")

    # Check first pass
    for i in range(3):
        batch_dict = reader.get_next()
        assert batch_dict is not None
        assert "input" in batch_dict
        np.testing.assert_array_equal(batch_dict["input"], batches[i])

    assert reader.get_next() is None

    # Check rewind
    reader.rewind()
    batch_dict = reader.get_next()
    assert batch_dict is not None
    np.testing.assert_array_equal(batch_dict["input"], batches[0])


@pytest.mark.skipif(not HAS_DEPS, reason="Requires torch, onnx, onnxruntime")
def test_quantize_onnx_dynamic(dummy_onnx_model, tmp_path):
    output_path = tmp_path / "quantized_dynamic.onnx"

    result_path = quantize_onnx_dynamic(dummy_onnx_model, output_path)

    assert result_path.exists()

    # Verify it can be loaded
    session = onnxruntime.InferenceSession(str(result_path), providers=["CPUExecutionProvider"])
    assert session is not None

    # Run a dummy pass
    dummy_input = np.random.randn(1, 3, 4, 4).astype(np.float32)
    outputs = session.run(None, {"input": dummy_input})
    assert outputs[0].shape == (1, 2)


@pytest.mark.skipif(not HAS_DEPS, reason="Requires torch, onnx, onnxruntime")
def test_quantize_onnx_static(dummy_onnx_model, tmp_path):
    output_path = tmp_path / "quantized_static.onnx"

    # Create calibration data
    batches = [np.random.randn(1, 3, 4, 4).astype(np.float32) for _ in range(5)]
    reader = TaxonCalibrationDataReader(batches, input_name="input")

    result_path = quantize_onnx_static(dummy_onnx_model, output_path, reader)

    assert result_path.exists()

    # Verify it can be loaded
    session = onnxruntime.InferenceSession(str(result_path), providers=["CPUExecutionProvider"])
    assert session is not None

    # Run a dummy pass
    dummy_input = np.random.randn(1, 3, 4, 4).astype(np.float32)
    outputs = session.run(None, {"input": dummy_input})
    assert outputs[0].shape == (1, 2)
