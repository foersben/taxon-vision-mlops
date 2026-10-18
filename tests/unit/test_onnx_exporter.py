# SPDX-FileCopyrightText: 2026 Benjamin Förster
# SPDX-License-Identifier: MIT
"""Unit tests for the ONNX computational graph exporter."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime
import pytest
import torch
import torch.nn as nn

from taxon_vision.inference.onnx_engine import ONNXInferenceEngine
from taxon_vision.inference.onnx_exporter import (
    _load_head_checkpoint,
    export_classifier_cli,
    export_to_onnx,
)


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


def test_export_classifier_cli_allow_untrained(tmp_path: Path) -> None:
    """Test CLI execution of export_classifier_cli with --allow-untrained flag and dual outputs."""
    out_file = tmp_path / "cli_model.onnx"
    ret = export_classifier_cli(
        [
            "--output",
            str(out_file),
            "--extractor",
            "mobilenetv4_conv_small",
            "--allow-untrained",
        ]
    )
    assert ret == 0
    assert out_file.exists()
    assert out_file.stat().st_size > 0

    session = onnxruntime.InferenceSession(str(out_file), providers=["CPUExecutionProvider"])
    out_names = [o.name for o in session.get_outputs()]
    assert "logits" in out_names
    assert "embeddings" in out_names


def test_export_classifier_cli_fails_when_checkpoint_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that CLI strictly fails (exit code 1) when checkpoint is missing and --allow-untrained is absent."""
    monkeypatch.setenv("TAXON_ALLOW_UNTRAINED_FALLBACK", "false")
    from taxon_vision.config import get_settings

    get_settings.cache_clear()

    out_file = tmp_path / "should_not_exist.onnx"
    missing_ckpt = tmp_path / "nonexistent_head.pt"

    ret = export_classifier_cli(
        [
            "--output",
            str(out_file),
            "--checkpoint",
            str(missing_ckpt),
            "--extractor",
            "mobilenetv4_conv_small",
        ]
    )
    assert ret == 1
    assert not out_file.exists()


def test_export_classifier_cli_strict_failure_on_mismatch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that CLI strictly fails (exit code 1) when checkpoint is mismatched and --allow-untrained is absent."""
    monkeypatch.setenv("TAXON_ALLOW_UNTRAINED_FALLBACK", "false")
    from taxon_vision.config import get_settings

    get_settings.cache_clear()

    mismatched_ckpt = tmp_path / "mismatched_head.pt"
    torch.save({"1.weight": torch.randn(10, 64), "1.bias": torch.randn(10)}, mismatched_ckpt)
    out_file = tmp_path / "should_fail.onnx"

    ret = export_classifier_cli(
        [
            "--output",
            str(out_file),
            "--checkpoint",
            str(mismatched_ckpt),
            "--extractor",
            "mobilenetv4_conv_small",
        ]
    )
    assert ret == 1
    assert not out_file.exists()


def test_export_classifier_cli_with_valid_checkpoint_and_quantization(tmp_path: Path) -> None:
    """Test CLI export with valid trained checkpoint and optional INT8 quantization."""
    ckpt_file = tmp_path / "valid_head.pt"
    # mobilenetv4_conv_small has feat_dim=1280, num_classes=10
    torch.save({"1.weight": torch.randn(10, 1280), "1.bias": torch.randn(10)}, ckpt_file)
    out_file = tmp_path / "trained_model.onnx"

    ret = export_classifier_cli(
        [
            "--output",
            str(out_file),
            "--checkpoint",
            str(ckpt_file),
            "--extractor",
            "mobilenetv4_conv_small",
            "--quantize-int8",
            "--quantize-mode",
            "dynamic",
        ]
    )
    assert ret == 0
    assert out_file.exists()
    assert out_file.stat().st_size > 0

    quantized_file = tmp_path / "trained_model.int8.onnx"
    assert quantized_file.exists()
    assert quantized_file.stat().st_size > 0


def test_load_head_checkpoint_formats(tmp_path: Path) -> None:
    """Test _load_head_checkpoint with raw, wrapped, and prefixed state dictionaries."""
    dummy_classifier: Any = nn.Module()
    dummy_classifier.head = nn.Sequential(nn.Dropout(0.2), nn.Linear(32, 5))

    weight = torch.randn(5, 32)
    bias = torch.randn(5)

    # 1. Raw head state dict
    ckpt1 = tmp_path / "raw.pt"
    torch.save({"1.weight": weight, "1.bias": bias}, ckpt1)
    _load_head_checkpoint(dummy_classifier, ckpt1)
    assert torch.allclose(dummy_classifier.head[1].weight, weight)

    # 2. Wrapped state dict
    ckpt2 = tmp_path / "wrapped.pt"
    torch.save({"model_state_dict": {"1.weight": weight * 2, "1.bias": bias * 2}}, ckpt2)
    _load_head_checkpoint(dummy_classifier, ckpt2)
    assert torch.allclose(dummy_classifier.head[1].weight, weight * 2)

    # 3. Prefixed state dict (from full classifier.state_dict())
    ckpt3 = tmp_path / "prefixed.pt"
    torch.save({"head.1.weight": weight * 3, "head.1.bias": bias * 3}, ckpt3)
    _load_head_checkpoint(dummy_classifier, ckpt3)
    assert torch.allclose(dummy_classifier.head[1].weight, weight * 3)


def test_export_classifier_module_subprocess(tmp_path: Path) -> None:
    """Test executing onnx_exporter via python -m subprocess with --allow-untrained."""
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
            "--allow-untrained",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0
    assert "ONNX export succeeded" in (proc.stdout + proc.stderr)
    assert out_file.exists()


def test_onnx_inference_engine(tmp_path: Path) -> None:
    """Test ONNXInferenceEngine execution and probability computation."""
    model = nn.Sequential(nn.Linear(8, 3))
    out_file = tmp_path / "test_engine.onnx"
    export_to_onnx(model, out_file, input_shape=(1, 8))

    engine = ONNXInferenceEngine(str(out_file))
    inputs = np.random.randn(2, 8).astype(np.float32)
    probs = engine.predict(inputs)
    assert probs.shape == (2, 3)
    assert np.allclose(probs.sum(axis=-1), 1.0)


def test_export_classifier_cli_quantize_int8(tmp_path: Path) -> None:
    """Test CLI execution with dynamic INT8 quantization enabled."""
    out_file = tmp_path / "model.onnx"
    int8_file = tmp_path / "model_int8.onnx"
    ret = export_classifier_cli(
        [
            "--output",
            str(out_file),
            "--quantize-int8",
            "--quantize-mode",
            "dynamic",
            "--quantized-output",
            str(int8_file),
            "--extractor",
            "mobilenetv4_conv_small",
            "--allow-untrained",
        ]
    )
    assert ret == 0
    assert out_file.exists()
    assert int8_file.exists()
    assert int8_file.stat().st_size > 0

    session = onnxruntime.InferenceSession(str(int8_file), providers=["CPUExecutionProvider"])
    assert len(session.get_inputs()) == 1
    assert len(session.get_outputs()) == 2


def test_onnx_inference_engine_embedding_projections(tmp_path: Path) -> None:
    """Test project_embeddings and predict_from_embeddings directly on ONNX linear head."""
    int8_file = tmp_path / "model_int8.onnx"
    ret = export_classifier_cli(
        [
            "--quantize-int8",
            "--quantize-mode",
            "dynamic",
            "--quantized-output",
            str(int8_file),
            "--extractor",
            "mobilenetv4_conv_small",
            "--allow-untrained",
        ]
    )
    assert ret == 0

    engine = ONNXInferenceEngine(int8_file)
    dummy_embs = np.random.randn(4, 1280).astype(np.float32)
    logits = engine.project_embeddings(dummy_embs)
    assert logits.shape[0] == 4
    assert logits.shape[1] == 10

    probs = engine.predict_from_embeddings(dummy_embs)
    assert probs.shape == (4, 10)
    assert np.allclose(probs.sum(axis=-1), 1.0, atol=1e-5)
