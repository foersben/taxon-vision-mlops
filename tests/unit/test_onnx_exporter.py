# SPDX-FileCopyrightText: 2026 Benjamin Förster
# SPDX-License-Identifier: MIT
"""Unit tests for the ONNX computational graph exporter."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import onnxruntime
import torch.nn as nn

from taxon_vision.inference.onnx_exporter import export_classifier_cli, export_to_onnx


def test_export_to_onnx_dummy_module(tmp_path: Path) -> None:
    """Test exporting a dummy PyTorch model to ONNX."""
    model = nn.Sequential(
        nn.Linear(16, 32),
        nn.ReLU(),
        nn.Linear(32, 5),
    )
    output_path = tmp_path / "dummy.onnx"
    result = export_to_onnx(model, output_path, input_shape=(1, 16))

    assert result == output_path
    assert output_path.exists()
    assert output_path.stat().st_size > 0

    session = onnxruntime.InferenceSession(str(output_path), providers=["CPUExecutionProvider"])
    assert len(session.get_inputs()) == 1
    assert len(session.get_outputs()) == 1


def test_export_classifier_cli_execution(tmp_path: Path) -> None:
    """Test CLI execution of export_classifier_cli."""
    out_file = tmp_path / "cli_model.onnx"
    ret = export_classifier_cli(["--output", str(out_file), "--extractor", "mobilenetv4_conv_small"])
    assert ret == 0
    assert out_file.exists()
    assert out_file.stat().st_size > 0


def test_export_classifier_module_subprocess(tmp_path: Path) -> None:
    """Test executing onnx_exporter via python -m subprocess."""
    out_file = tmp_path / "subproc_model.onnx"
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "taxon_vision.inference.onnx_exporter",
            "--output",
            str(out_file),
            "--extractor",
            "mobilenetv4_conv_small",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0
    assert "ONNX export succeeded" in (proc.stdout + proc.stderr)
    assert out_file.exists()
