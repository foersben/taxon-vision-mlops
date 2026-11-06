# SPDX-FileCopyrightText: 2026 Benjamin Förster
# SPDX-License-Identifier: MIT
"""Unit test suite for TaxonVisionPyFuncModel and end-to-end MLflow pyfunc serving plane."""

from __future__ import annotations

import base64
import io
from pathlib import Path
from unittest.mock import MagicMock

import mlflow.pyfunc
import numpy as np
import pandas as pd
import pytest
import yaml
from mlflow.pyfunc.model import PythonModelContext
from PIL import Image

from taxon_vision.inference.onnx_engine import InferenceOutputs
from taxon_vision.inference.pyfunc import (
    TaxonVisionPyFuncModel,
    _decode_image_payload,
    _load_catalog_entries,
    _load_conformal_parameters,
    _preprocess_input_batch,
)
from taxon_vision.service.inference import get_image_transform


def test_load_conformal_parameters_valid_and_defaults(tmp_path: Path) -> None:
    """Verify deserialisation of conformal and OOD parameters with fallback guarantees."""
    config_file = tmp_path / "conformal_config.yaml"
    payload = {
        "calibration": {
            "q_hat": 0.1234,
            "alpha": 0.05,
            "tau_ood": -15.5,
            "min_confidence_threshold": 0.75,
            "max_set_size_for_auto_accept": 4,
        }
    }
    with config_file.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f)

    q_hat, alpha, tau_ood, min_conf, k_max = _load_conformal_parameters(config_file)
    assert q_hat == pytest.approx(0.1234)
    assert alpha == pytest.approx(0.05)
    assert tau_ood == pytest.approx(-15.5)
    assert min_conf == pytest.approx(0.75)
    assert k_max == 4

    # Missing file fallback
    q_def, a_def, t_def, m_def, k_def = _load_conformal_parameters(tmp_path / "non_existent.yaml")
    assert isinstance(q_def, float)
    assert isinstance(a_def, float)
    assert isinstance(t_def, (float, type(None)))
    assert isinstance(m_def, float)
    assert isinstance(k_def, int)


def test_load_catalog_entries_valid_and_corrupt(tmp_path: Path) -> None:
    """Verify loading taxonomic catalog entries with schema validation and error handling."""
    catalog_file = tmp_path / "taxa_catalog.yaml"
    catalog_data = {
        "taxa": [
            {"id": 1, "scientific_name": "Turdus merula", "common_name": "Common Blackbird"},
            {"id": 2, "scientific_name": "Passer domesticus", "common_name": "House Sparrow"},
        ]
    }
    with catalog_file.open("w", encoding="utf-8") as f:
        yaml.safe_dump(catalog_data, f)

    entries = _load_catalog_entries(catalog_file)
    assert len(entries) == 2
    assert entries[0]["scientific_name"] == "Turdus merula"

    # Corrupt YAML falls back gracefully
    corrupt_file = tmp_path / "corrupt.yaml"
    corrupt_file.write_text("invalid: [yaml: broken", encoding="utf-8")
    fallback_entries = _load_catalog_entries(corrupt_file)
    assert isinstance(fallback_entries, list)


def test_decode_image_payload_variants() -> None:
    """Verify decoding and normalising of PIL Images, raw bytes, and base64 payloads."""
    transform = get_image_transform()
    img = Image.new("RGB", (224, 224), color=(100, 150, 200))

    # 1. PIL Image
    arr_pil = _decode_image_payload(img, transform)
    assert arr_pil.shape == (3, 224, 224)
    assert arr_pil.dtype == np.float32

    # 2. Binary bytes
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    raw_bytes = buf.getvalue()
    arr_bytes = _decode_image_payload(raw_bytes, transform)
    assert arr_bytes.shape == (3, 224, 224)

    # 3. Base64 string with and without data URI header
    b64_str = base64.b64encode(raw_bytes).decode("ascii")
    arr_b64 = _decode_image_payload(b64_str, transform)
    assert arr_b64.shape == (3, 224, 224)

    arr_uri = _decode_image_payload(f"data:image/jpeg;base64,{b64_str}", transform)
    assert arr_uri.shape == (3, 224, 224)

    # 4. Invalid item raises ValueError
    with pytest.raises(ValueError, match="Payload is not a valid decodable image"):
        _decode_image_payload(b"not_an_image_data", transform)


def test_preprocess_input_batch() -> None:
    """Verify batch ingestion across DataFrames, raw arrays, and embedding matrices."""
    transform = get_image_transform()
    img = Image.new("RGB", (224, 224), color=(50, 60, 70))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    b64_str = base64.b64encode(buf.getvalue()).decode("ascii")

    # DataFrame with base64 string
    df_b64 = pd.DataFrame({"image": [b64_str, b64_str]})
    arr, is_emb = _preprocess_input_batch(df_b64, transform)
    assert arr.shape == (2, 3, 224, 224)
    assert not is_emb

    # DataFrame with numeric embeddings
    df_num = pd.DataFrame(np.random.randn(5, 384).astype(np.float32))
    arr_num, is_emb_num = _preprocess_input_batch(df_num, transform)
    assert arr_num.shape == (5, 384)
    assert is_emb_num

    # List of images
    arr_list, is_emb_list = _preprocess_input_batch([img], transform)
    assert arr_list.shape == (1, 3, 224, 224)
    assert not is_emb_list

    # NumPy 4D array
    raw_arr = np.random.randn(3, 3, 224, 224).astype(np.float32)
    arr_np, is_emb_np = _preprocess_input_batch(raw_arr, transform)
    assert arr_np.shape == (3, 3, 224, 224)
    assert not is_emb_np


def test_pyfunc_lifecycle_and_predict(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Verify initialisation, load_context, and predict on synthetic inputs."""
    # Create mock ONNX file and config
    onnx_file = tmp_path / "model.onnx"
    onnx_file.write_bytes(b"dummy_onnx")
    calib_file = tmp_path / "conformal_config.yaml"
    calib_file.write_text(
        "calibration:\n  q_hat: 0.1\n  alpha: 0.05\n  tau_ood: -10.0\n  min_confidence_threshold: 0.5\n",
        encoding="utf-8",
    )
    catalog_file = tmp_path / "taxa_catalog.yaml"
    catalog_file.write_text(
        "taxa:\n  - id: 1\n    scientific_name: Taxon Alpha\n    common_name: Alpha\n  - id: 2\n    scientific_name: Taxon Beta\n    common_name: Beta\n",
        encoding="utf-8",
    )

    mock_engine = MagicMock()
    mock_engine.predict_outputs.return_value = InferenceOutputs(
        logits=np.array([[2.5, 0.5], [0.2, 3.1]], dtype=np.float32),
        embeddings=np.random.randn(2, 384).astype(np.float32),
    )
    mock_engine.project_embeddings.return_value = np.array([[2.5, 0.5], [0.2, 3.1]], dtype=np.float32)

    monkeypatch.setattr("taxon_vision.inference.pyfunc.ONNXInferenceEngine", lambda _p: mock_engine)

    pyfunc_model = TaxonVisionPyFuncModel()
    with pytest.raises(RuntimeError, match="Model engines are uninitialised"):
        pyfunc_model.predict(None, np.zeros((1, 3, 224, 224), dtype=np.float32))

    context = PythonModelContext(  # type: ignore[no-untyped-call]
        artifacts={
            "onnx_model": str(onnx_file),
            "conformal_config": str(calib_file),
            "taxa_catalog": str(catalog_file),
        },
        model_config={},
    )
    pyfunc_model.load_context(context)

    # Run inference on 4D array
    input_batch = np.zeros((2, 3, 224, 224), dtype=np.float32)
    predictions_df = pyfunc_model.predict(None, input_batch)

    assert isinstance(predictions_df, pd.DataFrame)
    assert len(predictions_df) == 2
    assert "top1_taxon" in predictions_df.columns
    assert "top1_confidence" in predictions_df.columns
    assert "conformal_set" in predictions_df.columns
    assert "set_size" in predictions_df.columns
    assert "ood_energy" in predictions_df.columns
    assert "is_ood" in predictions_df.columns
    assert "requires_review" in predictions_df.columns

    assert predictions_df.iloc[0]["top1_taxon"] == "Taxon Alpha"
    assert predictions_df.iloc[1]["top1_taxon"] == "Taxon Beta"
    assert predictions_df.iloc[0]["top1_confidence"] > 0.8
    assert isinstance(predictions_df.iloc[0]["conformal_set"], list)

    # Run inference on embedding batch
    emb_batch = np.random.randn(2, 384).astype(np.float32)
    emb_df = pyfunc_model.predict(None, emb_batch)
    assert len(emb_df) == 2
    assert emb_df.iloc[0]["top1_taxon"] == "Taxon Alpha"


def test_mlflow_pyfunc_save_and_load(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Verify saving and loading TaxonVisionPyFuncModel via MLflow pyfunc module."""
    onnx_file = tmp_path / "model.onnx"
    onnx_file.write_bytes(b"dummy_onnx")
    calib_file = tmp_path / "conformal_config.yaml"
    calib_file.write_text("calibration:\n  q_hat: 0.1\n  alpha: 0.05\n", encoding="utf-8")
    catalog_file = tmp_path / "taxa_catalog.yaml"
    catalog_file.write_text("taxa:\n  - scientific_name: Taxon Alpha\n", encoding="utf-8")

    mock_engine = MagicMock()
    mock_engine.predict_outputs.return_value = InferenceOutputs(
        logits=np.array([[3.0, 0.1]], dtype=np.float32),
        embeddings=np.random.randn(1, 384).astype(np.float32),
    )
    monkeypatch.setattr("taxon_vision.inference.pyfunc.ONNXInferenceEngine", lambda _p: mock_engine)

    model_dir = tmp_path / "saved_model"
    mlflow.pyfunc.save_model(
        path=str(model_dir),
        python_model=TaxonVisionPyFuncModel(),
        artifacts={
            "onnx_model": str(onnx_file),
            "conformal_config": str(calib_file),
            "taxa_catalog": str(catalog_file),
        },
    )

    loaded = mlflow.pyfunc.load_model(str(model_dir))
    preds = loaded.predict(np.zeros((1, 3, 224, 224), dtype=np.float32))
    assert isinstance(preds, pd.DataFrame)
    assert len(preds) == 1
    assert preds.iloc[0]["top1_taxon"] == "Taxon Alpha"
