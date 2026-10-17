# SPDX-FileCopyrightText: 2026 Benjamin Förster
# SPDX-License-Identifier: MIT
"""Unit tests for MLOps tracking lifecycle, DVC dataset hashing, and model promotion."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import mlflow
import mlflow.pytorch
import numpy as np
import pytest
import torch
import torch.nn as nn
from mlflow.models import ModelSignature
from mlflow.types.schema import Schema, TensorSpec

from taxon_vision.models.training.tracking import (
    MLflowTracker,
    _attach_mlflow_artifact_safely,
    _determine_promotion_alias,
    _end_mlflow_run_safely,
    _get_dvc_dataset_hash,
    _log_mlflow_metrics,
    _plot_confusion_matrix,
    _prepare_model_signature_and_example,
    _promote_model_if_better,
    log_training_visualizations,
    mlflow_run_scope,
)


def test_get_dvc_dataset_hash_missing_lock(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify None is returned when dvc.lock does not exist."""
    monkeypatch.chdir(tmp_path)
    assert _get_dvc_dataset_hash() is None


def test_get_dvc_dataset_hash_success(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify manifest MD5 hash is extracted from valid dvc.lock stage outputs."""
    lock_content = """
schema: "2.0"
stages:
  ingestion:
    cmd: python scripts/ingest.py
    outs:
      - path: data/manifests
        md5: 8f9b2d4e6a1c
"""
    lock_file = tmp_path / "dvc.lock"
    lock_file.write_text(lock_content, encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    assert _get_dvc_dataset_hash() == "8f9b2d4e6a1c"


def test_get_dvc_dataset_hash_no_manifest_out(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify None is returned when ingestion stage does not output data/manifests."""
    lock_content = """
stages:
  ingestion:
    outs:
      - path: other/path
        md5: 123456
"""
    lock_file = tmp_path / "dvc.lock"
    lock_file.write_text(lock_content, encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    assert _get_dvc_dataset_hash() is None


def test_get_dvc_dataset_hash_malformed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify None is returned gracefully when dvc.lock cannot be parsed."""
    lock_file = tmp_path / "dvc.lock"
    lock_file.write_text("invalid: [yaml: broken", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    assert _get_dvc_dataset_hash() is None


def test_end_mlflow_run_safely(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify _end_mlflow_run_safely delegates to mlflow.end_run and swallows exceptions."""
    mock_end_run = MagicMock()
    monkeypatch.setattr(mlflow, "end_run", mock_end_run)

    _end_mlflow_run_safely("FINISHED")
    mock_end_run.assert_called_once_with(status="FINISHED")

    # When mlflow raises an exception, it should not propagate
    mock_end_run.side_effect = RuntimeError("Tracking server down")
    _end_mlflow_run_safely("FAILED")


def test_attach_mlflow_artifact_safely(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify _attach_mlflow_artifact_safely attaches existing files and handles missing files."""
    mock_log_artifact = MagicMock()
    monkeypatch.setattr(mlflow, "log_artifact", mock_log_artifact)

    # None path or nonexistent path: no action
    _attach_mlflow_artifact_safely(None)
    _attach_mlflow_artifact_safely(tmp_path / "nonexistent.log")
    assert mock_log_artifact.call_count == 0

    # Existing file: logs artifact
    real_log = tmp_path / "training.log"
    real_log.write_text("epoch 1", encoding="utf-8")
    _attach_mlflow_artifact_safely(real_log)
    mock_log_artifact.assert_called_once_with(str(real_log))

    # Exception during artifact logging: swallowed safely
    mock_log_artifact.side_effect = OSError("Disk error")
    _attach_mlflow_artifact_safely(real_log)


def test_mlflow_run_scope_success(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify mlflow_run_scope sets tags, logs params, and marks run FINISHED on success."""
    mock_active_run = MagicMock()
    mock_start_run = MagicMock(return_value=mock_active_run)
    mock_end_run = MagicMock()
    mock_log_params = MagicMock()
    mock_log_param = MagicMock()
    mock_set_tag = MagicMock()

    monkeypatch.setattr(mlflow, "set_tracking_uri", MagicMock())
    monkeypatch.setattr(mlflow, "set_experiment", MagicMock())
    monkeypatch.setattr(mlflow, "start_run", mock_start_run)
    monkeypatch.setattr(mlflow, "end_run", mock_end_run)
    monkeypatch.setattr(mlflow, "log_params", mock_log_params)
    monkeypatch.setattr(mlflow, "log_param", mock_log_param)
    monkeypatch.setattr(mlflow, "set_tag", mock_set_tag)
    monkeypatch.setattr("taxon_vision.models.training.tracking._get_dvc_dataset_hash", lambda: "mock_dvc_hash")

    log_file = tmp_path / "run.log"
    log_file.write_text("test logs", encoding="utf-8")

    params = {"lr": 1e-4, "batch_size": 32}
    with mlflow_run_scope("test_run", params=params, log_file=log_file) as run:
        assert run == mock_active_run

    mock_start_run.assert_called_once_with(run_name="test_run")
    mock_log_params.assert_called_once_with(params)
    mock_end_run.assert_called_once_with(status="FINISHED")


def test_mlflow_run_scope_failure_propagation(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify mlflow_run_scope marks run FAILED and propagates internal exception."""
    mock_active_run = MagicMock()
    mock_start_run = MagicMock(return_value=mock_active_run)
    mock_end_run = MagicMock()

    monkeypatch.setattr(mlflow, "set_tracking_uri", MagicMock())
    monkeypatch.setattr(mlflow, "set_experiment", MagicMock())
    monkeypatch.setattr(mlflow, "start_run", mock_start_run)
    monkeypatch.setattr(mlflow, "end_run", mock_end_run)
    monkeypatch.setattr(mlflow, "log_params", MagicMock())
    monkeypatch.setattr(mlflow, "log_param", MagicMock())
    monkeypatch.setattr(mlflow, "set_tag", MagicMock())
    monkeypatch.setattr("taxon_vision.models.training.tracking._get_dvc_dataset_hash", lambda: "mock_dvc_hash")

    with pytest.raises(ValueError, match="Training exploded"), mlflow_run_scope("failing_run"):
        raise ValueError("Training exploded")

    mock_end_run.assert_called_once_with(status="FAILED")


def test_mlflow_run_scope_unreachable_server(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify mlflow_run_scope yields None and continues if server cannot be contacted."""
    mock_start_run = MagicMock(side_effect=ConnectionError("Cannot connect to remote MLflow"))
    monkeypatch.setattr(mlflow, "set_tracking_uri", MagicMock())
    monkeypatch.setattr(mlflow, "set_experiment", MagicMock())
    monkeypatch.setattr(mlflow, "start_run", mock_start_run)

    with mlflow_run_scope("offline_run") as run:
        assert run is None


def test_log_mlflow_metrics(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify _log_mlflow_metrics logs per-epoch trajectories and training duration."""
    mock_log_metrics = MagicMock()
    mock_log_metric = MagicMock()
    monkeypatch.setattr(mlflow, "log_metrics", mock_log_metrics)
    monkeypatch.setattr(mlflow, "log_metric", mock_log_metric)

    history = {
        "train_loss": [0.5, 0.3],
        "val_loss": [0.4, 0.25],
        "val_accuracy": [0.8, 0.9],
        "val_f1": [0.75, 0.88],
        "val_pr_auc": [0.70, 0.85],
    }

    _log_mlflow_metrics(history, duration=12.5)

    assert mock_log_metrics.call_count == 2
    mock_log_metric.assert_called_once_with("duration_seconds", 12.5)

    # Safe handling of exceptions
    mock_log_metrics.side_effect = RuntimeError("Tracking failure")
    _log_mlflow_metrics(history, duration=10.0)


def test_promote_model_initial_production(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify first registered model is tagged as Production when no baseline exists."""
    dummy_model = nn.Sequential(nn.Dropout(0.2), nn.Linear(32, 5))

    mock_model_info = MagicMock()
    mock_model_info.registered_model_version = "1"
    monkeypatch.setattr(mlflow.pytorch, "log_model", MagicMock(return_value=mock_model_info))
    monkeypatch.setattr(mlflow, "log_metric", MagicMock())

    mock_client = MagicMock()
    mock_client.get_model_version_by_alias.side_effect = Exception("No alias found")
    monkeypatch.setattr("mlflow.tracking.MlflowClient", lambda: mock_client)

    alias = _promote_model_if_better(dummy_model, pr_auc=0.82)
    assert alias == "Production"
    mock_client.set_registered_model_alias.assert_called_once_with("taxon_vision_classifier", "Production", "1")


def test_promote_model_challenger(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify new superior model is tagged as Challenger for canary evaluation."""
    dummy_model = nn.Sequential(nn.Linear(64, 10))

    mock_model_info = MagicMock()
    mock_model_info.registered_model_version = "2"
    monkeypatch.setattr(mlflow.pytorch, "log_model", MagicMock(return_value=mock_model_info))
    monkeypatch.setattr(mlflow, "log_metric", MagicMock())

    mock_prod_version = MagicMock(run_id="prod_run_123")
    mock_prod_run = MagicMock()
    mock_prod_run.data.metrics = {"val_promotion_score": 0.75}

    mock_client = MagicMock()
    mock_client.get_model_version_by_alias.return_value = mock_prod_version
    mock_client.get_run.return_value = mock_prod_run
    monkeypatch.setattr("mlflow.tracking.MlflowClient", lambda: mock_client)

    alias = _promote_model_if_better(dummy_model, pr_auc=0.88)
    assert alias == "Challenger"
    mock_client.set_registered_model_alias.assert_called_once_with("taxon_vision_classifier", "Challenger", "2")


def test_promote_model_archived(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify underperforming model is tagged as Archived."""
    dummy_model = nn.Sequential(nn.Linear(64, 10))

    mock_model_info = MagicMock()
    mock_model_info.registered_model_version = "3"
    monkeypatch.setattr(mlflow.pytorch, "log_model", MagicMock(return_value=mock_model_info))
    monkeypatch.setattr(mlflow, "log_metric", MagicMock())

    mock_prod_version = MagicMock(run_id="prod_run_123")
    mock_prod_run = MagicMock()
    mock_prod_run.data.metrics = {"val_promotion_score": 0.85}

    mock_client = MagicMock()
    mock_client.get_model_version_by_alias.return_value = mock_prod_version
    mock_client.get_run.return_value = mock_prod_run
    monkeypatch.setattr("mlflow.tracking.MlflowClient", lambda: mock_client)

    alias = _promote_model_if_better(dummy_model, pr_auc=0.80)
    assert alias == "Archived"
    mock_client.set_registered_model_alias.assert_called_once_with("taxon_vision_classifier", "Archived", "3")


def test_promote_model_serialization_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify None is returned when model logging fails completely."""
    dummy_model = nn.Sequential(nn.Linear(16, 2))
    monkeypatch.setattr(mlflow.pytorch, "log_model", MagicMock(side_effect=RuntimeError("Serialization error")))
    monkeypatch.setattr(mlflow, "log_metric", MagicMock())

    alias = _promote_model_if_better(dummy_model, pr_auc=0.75)
    assert alias is None


def test_promote_model_with_real_val_embeddings(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify real validation embeddings infer model signature and populate version metadata."""
    dummy_model = nn.Sequential(nn.Linear(32, 5))
    val_embeddings = torch.randn(10, 32)

    mock_model_info = MagicMock()
    mock_model_info.registered_model_version = "4"
    mock_log_model = MagicMock(return_value=mock_model_info)
    monkeypatch.setattr(mlflow.pytorch, "log_model", mock_log_model)
    monkeypatch.setattr(mlflow, "log_metric", MagicMock())

    mock_client = MagicMock()
    mock_client.get_model_version_by_alias.side_effect = Exception("No alias found")
    monkeypatch.setattr("mlflow.tracking.MlflowClient", lambda: mock_client)

    alias = _promote_model_if_better(
        dummy_model,
        pr_auc=0.92,
        val_embeddings=val_embeddings,
        extractor="dinov3",
    )
    assert alias == "Production"

    # Verify signature and input example were provided
    mock_log_model.assert_called_once()
    kwargs = mock_log_model.call_args.kwargs
    assert kwargs.get("signature") is not None
    example = kwargs.get("input_example")
    assert example is not None
    assert example.shape == (5, 32)

    # Verify model metadata updates
    mock_client.update_model_version.assert_called_once()
    mock_client.set_model_version_tag.assert_any_call("taxon_vision_classifier", "4", "extractor", "dinov3")
    mock_client.set_model_version_tag.assert_any_call("taxon_vision_classifier", "4", "val_pr_auc", "0.9200")


def test_log_training_visualizations(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify log_training_visualizations emits confusion matrix, PR curves, and energy distribution."""
    mock_log_figure = MagicMock()
    monkeypatch.setattr(mlflow, "log_figure", mock_log_figure)
    monkeypatch.setattr(mlflow, "active_run", lambda: True)

    val_logits = np.random.randn(20, 5)
    val_labels = np.random.randint(0, 5, size=20)
    taxa_catalog = [{"common_name": f"Species {i}", "scientific_name": f"Taxon {i}"} for i in range(5)]

    log_training_visualizations(
        val_logits=val_logits,
        val_labels=val_labels,
        taxa_catalog=taxa_catalog,
        calibrated_energy_threshold=-3.50,
    )

    assert mock_log_figure.call_count == 3
    artifact_paths = [call.args[1] for call in mock_log_figure.call_args_list]
    assert "visualizations/confusion_matrix.png" in artifact_paths
    assert "visualizations/precision_recall_curves.png" in artifact_paths
    assert "visualizations/energy_distribution.png" in artifact_paths


def test_determine_promotion_alias_tie_break() -> None:
    """Verify promotion tie-breaking logic favors replacement on equal or better metrics."""
    mock_client = MagicMock()

    # Scenario 1: Initial model when no production exists -> Production
    mock_client.get_model_version_by_alias.side_effect = Exception("No alias")
    alias = _determine_promotion_alias(mock_client, "taxon_vision_classifier", version="1", pr_auc=0.8500)
    assert alias == "Production"

    # Scenario 2: Production exists at 0.85. Candidate achieves 0.85 (exact tie) -> Challenger
    prod_mv = MagicMock()
    prod_mv.run_id = "run_1"
    prod_run = MagicMock()
    prod_run.data.metrics = {"val_promotion_score": 0.8500}

    mock_client.get_model_version_by_alias.side_effect = None
    mock_client.get_model_version_by_alias.return_value = prod_mv
    mock_client.get_run.return_value = prod_run

    alias = _determine_promotion_alias(mock_client, "taxon_vision_classifier", version="2", pr_auc=0.8500)
    assert alias == "Challenger"

    # Scenario 3: Candidate below production (0.75 < 0.85) -> Archived
    alias = _determine_promotion_alias(mock_client, "taxon_vision_classifier", version="3", pr_auc=0.7500)
    assert alias == "Archived"


def test_prepare_model_signature_and_example_cpu_isolation() -> None:
    """Verify model is isolated on CPU and input example is NumPy array to prevent PT2 CUDA errors."""
    model = nn.Sequential(nn.Linear(16, 4))
    val_embeddings = torch.randn(20, 16)

    signature, example = _prepare_model_signature_and_example(model, val_embeddings)

    # Example must be numpy array with at most 5 rows
    assert isinstance(example, np.ndarray)
    assert example.shape == (5, 16)
    assert signature is not None


def test_plot_confusion_matrix_explicit_labels() -> None:
    """Verify _plot_confusion_matrix builds K x K matrix even when validation split has single label."""
    y_true = np.zeros(10, dtype=int)
    y_pred = np.zeros(10, dtype=int)
    class_names = ["Species 0", "Species 1", "Species 2"]

    fig = _plot_confusion_matrix(y_true, y_pred, class_names)
    assert fig is not None


def test_build_embedding_and_onnx_signatures() -> None:
    """Verify MLflowTracker signature builders create valid ModelSignature instances with TensorSpecs."""
    emb_sig = MLflowTracker.build_embedding_signature(input_dim=512, num_classes=15)
    assert isinstance(emb_sig, ModelSignature)
    assert isinstance(emb_sig.inputs, Schema)
    assert isinstance(emb_sig.outputs, Schema)
    assert len(emb_sig.inputs.inputs) == 1
    assert emb_sig.inputs.inputs[0].name == "backbone_embeddings"
    assert emb_sig.inputs.inputs[0].shape == (-1, 512)
    assert emb_sig.inputs.inputs[0].type == np.dtype(np.float32)

    assert len(emb_sig.outputs.inputs) == 1
    assert emb_sig.outputs.inputs[0].name == "taxonomic_logits"
    assert emb_sig.outputs.inputs[0].shape == (-1, 15)
    assert emb_sig.outputs.inputs[0].type == np.dtype(np.float32)

    onnx_sig = MLflowTracker.build_onnx_signature(num_classes=15, image_size=224)
    assert isinstance(onnx_sig, ModelSignature)
    assert isinstance(onnx_sig.inputs, Schema)
    assert isinstance(onnx_sig.outputs, Schema)
    assert len(onnx_sig.inputs.inputs) == 1
    assert onnx_sig.inputs.inputs[0].name == "input_images"
    assert onnx_sig.inputs.inputs[0].shape == (-1, 3, 224, 224)
    assert onnx_sig.inputs.inputs[0].type == np.dtype(np.float32)

    assert len(onnx_sig.outputs.inputs) == 1
    assert onnx_sig.outputs.inputs[0].name == "taxonomic_logits"
    assert onnx_sig.outputs.inputs[0].shape == (-1, 15)
    assert onnx_sig.outputs.inputs[0].type == np.dtype(np.float32)


def test_log_model_with_explicit_schema(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify MLflowTracker.log_model invokes mlflow.pytorch.log_model with explicit ModelSignature and input example."""
    mock_log_model = MagicMock()
    monkeypatch.setattr(mlflow.pytorch, "log_model", mock_log_model)

    model = nn.Sequential(nn.Dropout(0.2), nn.Linear(64, 8))

    MLflowTracker.log_model(model=model, model_name="taxon_vision_classifier")

    mock_log_model.assert_called_once()
    kwargs = mock_log_model.call_args.kwargs
    sig = kwargs.get("signature")
    assert sig is not None
    assert isinstance(sig, ModelSignature)

    # Validate input schema
    assert len(sig.inputs.inputs) == 1
    input_spec = sig.inputs.inputs[0]
    assert isinstance(input_spec, TensorSpec)
    assert input_spec.name == "backbone_embeddings"
    assert input_spec.shape == (-1, 64)
    assert input_spec.type == np.dtype(np.float32)

    # Validate output schema
    assert len(sig.outputs.inputs) == 1
    output_spec = sig.outputs.inputs[0]
    assert isinstance(output_spec, TensorSpec)
    assert output_spec.name == "taxonomic_logits"
    assert output_spec.shape == (-1, 8)
    assert output_spec.type == np.dtype(np.float32)

    # Validate input example
    example = kwargs.get("input_example")
    assert example is not None
    assert isinstance(example, np.ndarray)
    assert example.shape == (1, 64)
    assert example.dtype == np.float32


def test_log_onnx_model_signature(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify MLflowTracker.log_onnx_model constructs valid ONNX image tensor signature."""
    import mlflow.onnx

    mock_log_onnx = MagicMock()
    monkeypatch.setattr(mlflow.onnx, "log_model", mock_log_onnx)

    dummy_onnx = MagicMock()
    MLflowTracker.log_onnx_model(
        onnx_model=dummy_onnx,
        artifact_path="model",
        num_classes=12,
        image_size=224,
    )

    mock_log_onnx.assert_called_once()
    kwargs = mock_log_onnx.call_args.kwargs
    sig = kwargs.get("signature")
    assert sig is not None
    assert isinstance(sig, ModelSignature)

    # Validate input image TensorSpec
    input_spec = sig.inputs.inputs[0]
    assert isinstance(input_spec, TensorSpec)
    assert input_spec.name == "input_images"
    assert input_spec.shape == (-1, 3, 224, 224)
    assert input_spec.type == np.dtype(np.float32)

    # Validate output logits TensorSpec
    output_spec = sig.outputs.inputs[0]
    assert isinstance(output_spec, TensorSpec)
    assert output_spec.name == "taxonomic_logits"
    assert output_spec.shape == (-1, 12)
    assert output_spec.type == np.dtype(np.float32)

    # Validate input example
    example = kwargs.get("input_example")
    assert example is not None
    assert isinstance(example, np.ndarray)
    assert example.shape == (1, 3, 224, 224)
    assert example.dtype == np.float32
