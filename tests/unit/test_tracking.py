# SPDX-FileCopyrightText: 2026 Benjamin Förster
# SPDX-License-Identifier: MIT
"""Unit tests for MLOps tracking lifecycle, DVC dataset hashing, and model promotion."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast
from unittest.mock import MagicMock

import mlflow
import mlflow.pytorch
import numpy as np
import pytest
import torch
import torch.nn as nn
from mlflow.models import ModelSignature
from mlflow.types.schema import Schema, TensorSpec

from taxon_vision.models.training.dataset import get_dvc_dataset_hash
from taxon_vision.models.training.promotion import (
    ModelPromotionTarget,
    PromotionCriteria,
    evaluate_promotion_gates,
    promote_candidate_model,
)
from taxon_vision.models.training.tracking import (
    MLflowTracker,
    _attach_mlflow_artifact_safely,
    _determine_promotion_alias,
    _end_mlflow_run_safely,
    _log_mlflow_metrics,
    _plot_confusion_matrix,
    _prepare_model_signature_and_example,
    _promote_model_if_better,
    _resolve_reachable_tracking_uri,
    log_training_visualizations,
    mlflow_run_scope,
    resume_parent_mlflow_run,
)
from taxon_vision.models.training.types import PromotionAlias
from taxon_vision.models.training.visualizations import (
    HoldoutVisualizationPayload,
    log_holdout_visualizations,
)


def test_get_dvc_dataset_hash_missing_lock(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify None is returned when dvc.lock does not exist."""
    monkeypatch.chdir(tmp_path)
    assert get_dvc_dataset_hash() is None


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

    assert get_dvc_dataset_hash() == "8f9b2d4e6a1c"


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

    assert get_dvc_dataset_hash() is None


def test_get_dvc_dataset_hash_malformed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify None is returned gracefully when dvc.lock cannot be parsed."""
    lock_file = tmp_path / "dvc.lock"
    lock_file.write_text("invalid: [yaml: broken", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    assert get_dvc_dataset_hash() is None


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
    monkeypatch.setattr("taxon_vision.models.training.tracking.get_dvc_dataset_hash", lambda: "mock_dvc_hash")

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
    monkeypatch.setattr("taxon_vision.models.training.tracking.get_dvc_dataset_hash", lambda: "mock_dvc_hash")

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

    mock_model_info = MagicMock(model_uri="runs:/123/model")
    mock_model_info.registered_model_version = "1"
    monkeypatch.setattr(mlflow.pytorch, "log_model", MagicMock(return_value=mock_model_info))
    monkeypatch.setattr(mlflow, "log_metric", MagicMock())
    monkeypatch.setattr(mlflow, "set_tag", MagicMock())
    monkeypatch.setattr(mlflow, "register_model", MagicMock(return_value=MagicMock(version="1")))

    mock_client = MagicMock()
    mock_client.get_model_version_by_alias.side_effect = Exception("No alias found")
    monkeypatch.setattr("mlflow.tracking.MlflowClient", lambda: mock_client)

    alias = _promote_model_if_better(dummy_model, pr_auc=0.82)
    assert alias == "Production"
    mock_client.set_registered_model_alias.assert_called_once_with("taxon_vision_classifier", "Production", "1")


def test_promote_model_challenger(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify new superior model is registered and tagged as Challenger for canary evaluation."""
    dummy_model = nn.Sequential(nn.Linear(64, 10))

    mock_model_info = MagicMock(model_uri="runs:/123/model")
    mock_model_info.registered_model_version = "2"
    monkeypatch.setattr(mlflow.pytorch, "log_model", MagicMock(return_value=mock_model_info))
    monkeypatch.setattr(mlflow, "log_metric", MagicMock())
    monkeypatch.setattr(mlflow, "set_tag", MagicMock())
    monkeypatch.setattr(mlflow, "register_model", MagicMock(return_value=MagicMock(version="2")))

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


def test_promote_model_skipped_below_incumbent(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify underperforming model is not registered and tagged as skipped_below_incumbent."""
    dummy_model = nn.Sequential(nn.Linear(64, 10))

    mock_model_info = MagicMock(model_uri="runs:/123/model")
    monkeypatch.setattr(mlflow.pytorch, "log_model", MagicMock(return_value=mock_model_info))
    monkeypatch.setattr(mlflow, "log_metric", MagicMock())
    mock_set_tag = MagicMock()
    monkeypatch.setattr(mlflow, "set_tag", mock_set_tag)
    mock_register_model = MagicMock()
    monkeypatch.setattr(mlflow, "register_model", mock_register_model)

    mock_prod_version = MagicMock(run_id="prod_run_123")
    mock_prod_run = MagicMock()
    mock_prod_run.data.metrics = {"val_promotion_score": 0.85}

    mock_client = MagicMock()
    mock_client.get_model_version_by_alias.return_value = mock_prod_version
    mock_client.get_run.return_value = mock_prod_run
    monkeypatch.setattr("mlflow.tracking.MlflowClient", lambda: mock_client)

    # 1. Inferior score (0.80 < 0.85) -> skipped_below_incumbent
    alias = _promote_model_if_better(dummy_model, pr_auc=0.80)
    assert alias == "skipped_below_incumbent"
    mock_register_model.assert_not_called()
    mock_set_tag.assert_any_call("registry.status", "skipped_below_incumbent")

    # 2. Equal score (0.85 == 0.85) -> strictly beats required -> skipped_below_incumbent
    alias_tie = _promote_model_if_better(dummy_model, pr_auc=0.85)
    assert alias_tie == "skipped_below_incumbent"
    mock_register_model.assert_not_called()


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

    mock_model_info = MagicMock(model_uri="runs:/123/model")
    mock_model_info.registered_model_version = "4"
    mock_log_model = MagicMock(return_value=mock_model_info)
    monkeypatch.setattr(mlflow.pytorch, "log_model", mock_log_model)
    monkeypatch.setattr(mlflow, "log_metric", MagicMock())
    monkeypatch.setattr(mlflow, "set_tag", MagicMock())
    monkeypatch.setattr(mlflow, "register_model", MagicMock(return_value=MagicMock(version="4")))

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
    mock_log_metric = MagicMock()
    monkeypatch.setattr(mlflow, "log_figure", mock_log_figure)
    monkeypatch.setattr(mlflow, "log_metric", mock_log_metric)
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

    logged_metric_keys = [call.args[0] for call in mock_log_metric.call_args_list]
    assert "worst_class_ap" in logged_metric_keys
    assert "median_class_ap" in logged_metric_keys


def test_log_holdout_visualizations(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify log_holdout_visualizations emits all publication-grade figures to MLflow."""
    mock_log_figure = MagicMock()
    monkeypatch.setattr(mlflow, "log_figure", mock_log_figure)
    monkeypatch.setattr(mlflow, "active_run", lambda: True)

    n_samples, n_classes = 30, 4
    test_logits = np.random.randn(n_samples, n_classes).astype(np.float32)
    shifted = test_logits - np.max(test_logits, axis=1, keepdims=True)
    exp = np.exp(shifted)
    test_probs = (exp / np.sum(exp, axis=1, keepdims=True)).astype(np.float32)
    test_labels = np.random.randint(0, n_classes, size=n_samples)
    set_sizes = np.random.randint(1, 4, size=n_samples)

    class DummyParetoProfile:
        extractor = "dinov3_vits14"
        parameters_m = 22.0
        p95_latency_ms = 13.5
        throughput_fps = 105.0

    payload = HoldoutVisualizationPayload(
        test_labels=test_labels,
        test_probs=test_probs,
        test_logits=test_logits,
        set_sizes=set_sizes,
        q_hat=0.08,
        nominal_coverage=0.95,
        empirical_coverage=0.96,
        calibrated_energy_threshold=-12.5,
        k_max=5,
        taxa_catalog=[{"common_name": f"Taxon {i}", "scientific_name": f"Species {i}"} for i in range(n_classes)],
        pareto_profiles=[DummyParetoProfile()],
        latency_slo_ms=25.0,
    )

    log_holdout_visualizations(payload)

    assert mock_log_figure.call_count == 6
    artifact_paths = [call.args[1] for call in mock_log_figure.call_args_list]
    assert "visualizations/test_conformal_set_sizes.png" in artifact_paths
    assert "visualizations/test_reliability_diagram.png" in artifact_paths
    assert "visualizations/test_energy_distribution.png" in artifact_paths
    assert "visualizations/test_confusion_matrix.png" in artifact_paths
    assert "visualizations/test_precision_recall_curves.png" in artifact_paths
    assert "visualizations/pareto_frontier.png" in artifact_paths


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
    from mlflow.types import TensorSpec

    assert len(emb_sig.inputs.inputs) == 1
    assert emb_sig.inputs.inputs[0].name == "backbone_embeddings"
    assert isinstance(emb_sig.inputs.inputs[0], TensorSpec)
    assert emb_sig.inputs.inputs[0].shape == (-1, 512)
    assert emb_sig.inputs.inputs[0].type == np.dtype(np.float32)

    assert len(emb_sig.outputs.inputs) == 1
    assert emb_sig.outputs.inputs[0].name == "taxonomic_logits"
    assert isinstance(emb_sig.outputs.inputs[0], TensorSpec)
    assert emb_sig.outputs.inputs[0].shape == (-1, 15)
    assert emb_sig.outputs.inputs[0].type == np.dtype(np.float32)

    onnx_sig = MLflowTracker.build_onnx_signature(num_classes=15, image_size=224)
    assert isinstance(onnx_sig, ModelSignature)
    assert isinstance(onnx_sig.inputs, Schema)
    assert isinstance(onnx_sig.outputs, Schema)
    assert len(onnx_sig.inputs.inputs) == 1
    assert onnx_sig.inputs.inputs[0].name == "input_images"
    assert isinstance(onnx_sig.inputs.inputs[0], TensorSpec)
    assert onnx_sig.inputs.inputs[0].shape == (-1, 3, 224, 224)
    assert onnx_sig.inputs.inputs[0].type == np.dtype(np.float32)

    assert len(onnx_sig.outputs.inputs) == 1
    assert onnx_sig.outputs.inputs[0].name == "taxonomic_logits"
    assert isinstance(onnx_sig.outputs.inputs[0], TensorSpec)
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


def test_mlflow_run_scope_nested_and_ci_tags(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify mlflow_run_scope supports nested runs and injects CI/CD lineage tags."""
    mock_active_run = MagicMock()
    mock_start_run = MagicMock(return_value=mock_active_run)
    mock_set_tag = MagicMock()

    monkeypatch.setattr(mlflow, "set_tracking_uri", MagicMock())
    monkeypatch.setattr(mlflow, "set_experiment", MagicMock())
    monkeypatch.setattr(mlflow, "start_run", mock_start_run)
    monkeypatch.setattr(mlflow, "end_run", MagicMock())
    monkeypatch.setattr(mlflow, "set_tag", mock_set_tag)
    monkeypatch.setattr("taxon_vision.models.training.tracking.get_dvc_dataset_hash", lambda: None)

    monkeypatch.setenv("BUILD_TAG", "jenkins-taxon-pipeline-42")
    monkeypatch.setenv("BUILD_URL", "https://ci.taxon.org/job/taxon/42/")
    monkeypatch.setenv("GITHUB_RUN_ID", "99887766")
    monkeypatch.setenv("GITHUB_WORKFLOW", "Vision CI")

    with mlflow_run_scope("nested_trial_run", nested=True) as run:
        assert run == mock_active_run

    mock_start_run.assert_called_once_with(run_name="nested_trial_run", nested=True)
    tag_calls = {call.args[0]: call.args[1] for call in mock_set_tag.call_args_list}
    assert tag_calls.get("ci.jenkins_build_tag") == "jenkins-taxon-pipeline-42"
    assert tag_calls.get("ci.jenkins_build_url") == "https://ci.taxon.org/job/taxon/42/"
    assert tag_calls.get("ci.github_run_id") == "99887766"
    assert tag_calls.get("ci.github_workflow") == "Vision CI"


def test_log_model_strict_signature_enforcement() -> None:
    """Verify MLflowTracker.log_model raises ValueError when signature contract is violated."""
    dummy_model = nn.Linear(16, 4)
    with pytest.raises(ValueError, match="Strict contract violation"):
        MLflowTracker.log_model(
            model=dummy_model,
            signature=cast(Any, "not_a_model_signature"),
        )


def test_log_dataset_lineage(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify _log_dataset_lineage converts arrays via mlflow.data and calls mlflow.log_input."""
    from taxon_vision.models.training.tracking import _log_dataset_lineage
    from taxon_vision.models.training.types import EmbeddingSplit

    mock_log_input = MagicMock()
    mock_from_numpy = MagicMock(return_value="mock_dataset")
    monkeypatch.setattr(mlflow, "active_run", MagicMock(return_value=MagicMock()))
    monkeypatch.setattr(mlflow, "log_input", mock_log_input)
    monkeypatch.setattr(mlflow.data, "from_numpy", mock_from_numpy)
    emb = torch.zeros((10, 16), dtype=torch.float32)
    lbl = torch.ones((10,), dtype=torch.long)
    split = EmbeddingSplit(
        train_embeddings=emb,
        train_labels=lbl,
        val_embeddings=emb,
        val_labels=lbl,
    )
    result = _log_dataset_lineage(data=split, manifest_path=None)
    assert result.train == "mock_dataset"
    assert result.val == "mock_dataset"
    assert mock_log_input.call_count >= 2


def test_log_onnx_model_from_file_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify log_onnx_model loads ONNX model using onnx.load when given a file path."""
    import onnx

    dummy_onnx_path = tmp_path / "model.onnx"
    dummy_onnx_path.write_bytes(b"dummy_onnx_proto")

    mock_onnx_load = MagicMock(return_value="mock_proto")
    mock_log_onnx = MagicMock()
    monkeypatch.setattr(onnx, "load", mock_onnx_load)
    monkeypatch.setattr(mlflow.onnx, "log_model", mock_log_onnx)

    MLflowTracker.log_onnx_model(
        onnx_model=dummy_onnx_path,
        artifact_path="model",
        num_classes=5,
        image_size=224,
    )

    mock_onnx_load.assert_called_once_with(str(dummy_onnx_path))
    mock_log_onnx.assert_called_once()
    assert mock_log_onnx.call_args.kwargs["onnx_model"] == "mock_proto"


def test_resolve_reachable_tracking_uri_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify MLFLOW_TRACKING_URI environment variable takes precedence unconditionally."""
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://custom-runner:5000")
    assert _resolve_reachable_tracking_uri() == "http://custom-runner:5000"


def test_resolve_reachable_tracking_uri_health_success(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify HTTP tracking URI is returned when /health responds with HTTP 200."""
    import urllib.request

    from taxon_vision.config import get_settings

    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    monkeypatch.delenv("TAXON_MLFLOW_TRACKING_URI", raising=False)
    settings = get_settings()
    monkeypatch.setattr(settings.mlflow, "tracking_uri", "http://127.0.0.1:5000")

    mock_resp = MagicMock()
    mock_resp.status = 200
    mock_resp.__enter__.return_value = mock_resp

    mock_urlopen = MagicMock(return_value=mock_resp)
    monkeypatch.setattr(urllib.request, "urlopen", mock_urlopen)

    resolved = _resolve_reachable_tracking_uri(settings)
    assert resolved == "http://127.0.0.1:5000"
    mock_urlopen.assert_called_once()


def test_resolve_reachable_tracking_uri_fallback_sqlite(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify unreachable HTTP tracking server triggers fallback to local SQLite backend."""
    import urllib.request

    from taxon_vision.config import get_settings

    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    monkeypatch.delenv("TAXON_MLFLOW_TRACKING_URI", raising=False)
    settings = get_settings()
    monkeypatch.setattr(settings.mlflow, "tracking_uri", "http://127.0.0.1:5000")
    monkeypatch.setattr(urllib.request, "urlopen", MagicMock(side_effect=OSError("Connection refused")))

    custom_fallback = f"sqlite:///{tmp_path}/mlflow_fallback.db"
    monkeypatch.setattr(settings.mlflow, "fallback_sqlite_uri", custom_fallback)

    resolved = _resolve_reachable_tracking_uri(settings)
    assert resolved == custom_fallback
    assert (tmp_path / "artifacts").exists()


def test_mlflow_run_scope_custom_experiment_name(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify mlflow_run_scope applies explicit experiment_name override when supplied."""
    mock_set_experiment = MagicMock()
    monkeypatch.setattr(mlflow, "set_tracking_uri", MagicMock())
    monkeypatch.setattr(mlflow, "set_experiment", mock_set_experiment)
    monkeypatch.setattr(mlflow, "start_run", MagicMock(return_value=MagicMock()))
    monkeypatch.setattr(mlflow, "end_run", MagicMock())
    monkeypatch.setattr(mlflow, "set_tag", MagicMock())
    with mlflow_run_scope("drift_eval_run", experiment_name="taxon-vision-drift"):
        pass
    mock_set_experiment.assert_called_once_with("taxon-vision-drift")


def test_resume_parent_mlflow_run_success(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify resume_parent_mlflow_run parses metrics JSON, sets tracking URI, and re-enters run."""

    metrics_file = tmp_path / "train_metrics.json"
    metrics_file.write_text(
        json.dumps(
            {
                "status": "completed",
                "extractor": "dinov3",
                "mlflow_run_id": "parent_run_12345",
                "mlflow_model_id": "model_uuid_67890",
            }
        ),
        encoding="utf-8",
    )

    mock_start_run = MagicMock()
    mock_end_run = MagicMock()
    monkeypatch.setattr(mlflow, "set_tracking_uri", MagicMock())
    monkeypatch.setattr(mlflow, "start_run", mock_start_run)
    monkeypatch.setattr(mlflow, "end_run", mock_end_run)
    monkeypatch.setattr(
        "taxon_vision.models.training.tracking._resolve_reachable_tracking_uri",
        lambda _settings=None: "http://mlflow-server:5000",
    )

    with resume_parent_mlflow_run(metrics_file) as ctx:
        assert ctx is not None
        assert ctx.run_id == "parent_run_12345"
        assert ctx.model_id == "model_uuid_67890"
        assert ctx.extractor == "dinov3"

    mock_start_run.assert_called_once_with(run_id="parent_run_12345")
    mock_end_run.assert_called_once_with(status="FINISHED")


def test_resume_parent_mlflow_run_missing_or_corrupt_file(tmp_path: Path) -> None:
    """Verify resume_parent_mlflow_run yields None when file does not exist or lacks run_id."""
    # Scenario 1: Non-existent file
    with resume_parent_mlflow_run(tmp_path / "non_existent.json") as ctx:
        assert ctx is None

    # Scenario 2: Corrupted JSON
    corrupt_file = tmp_path / "corrupt.json"
    corrupt_file.write_text("{malformed: json", encoding="utf-8")
    with resume_parent_mlflow_run(corrupt_file) as ctx:
        assert ctx is None

    # Scenario 3: Missing mlflow_run_id key
    missing_key_file = tmp_path / "no_run_id.json"
    missing_key_file.write_text(json.dumps({"status": "completed"}), encoding="utf-8")
    with resume_parent_mlflow_run(missing_key_file) as ctx:
        assert ctx is None


def test_resume_parent_mlflow_run_connection_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify resume_parent_mlflow_run yields None gracefully when tracking server rejects connection."""
    metrics_file = tmp_path / "train_metrics.json"
    metrics_file.write_text(
        json.dumps({"mlflow_run_id": "parent_run_unreachable"}),
        encoding="utf-8",
    )

    monkeypatch.setattr(mlflow, "set_tracking_uri", MagicMock())
    monkeypatch.setattr(
        mlflow,
        "start_run",
        MagicMock(side_effect=ConnectionError("Server unreachable")),
    )
    monkeypatch.setattr(
        "taxon_vision.models.training.tracking._resolve_reachable_tracking_uri",
        lambda _settings=None: "http://down:5000",
    )

    with resume_parent_mlflow_run(metrics_file) as ctx:
        assert ctx is None


def test_evaluate_promotion_gates_initial_production_bootstrapping() -> None:
    """Verify first candidate model bootstraps to @Production when no incumbent exists."""
    criteria = PromotionCriteria(
        candidate_pr_auc=0.88,
        fp32_to_int8_drop=0.01,
        empirical_coverage=0.96,
        nominal_coverage=0.95,
        p95_latency_ms=45.0,
        exchangeability_p_value=None,
        live_sample_count=0,
    )
    verdict = evaluate_promotion_gates(criteria, incumbent_pr_auc=None)
    assert verdict.alias == PromotionAlias.PRODUCTION
    assert verdict.gate == "initial_production"
    assert "gate_1_predictive_viability" in verdict.passed_gates
    assert "gate_2_conformal_coverage" in verdict.passed_gates
    assert "gate_3_latency_slo" in verdict.passed_gates
    assert "gate_4_bootstrap_production" in verdict.passed_gates
    assert verdict.failed_gate is None


def test_evaluate_promotion_gates_gate1_pr_auc_failure() -> None:
    """Verify candidate is archived if PR-AUC is below incumbent."""
    criteria = PromotionCriteria(
        candidate_pr_auc=0.82,
        fp32_to_int8_drop=0.01,
        empirical_coverage=0.96,
        nominal_coverage=0.95,
        p95_latency_ms=40.0,
    )
    verdict = evaluate_promotion_gates(criteria, incumbent_pr_auc=0.85)
    assert verdict.alias == PromotionAlias.ARCHIVED
    assert verdict.gate == "below_incumbent"
    assert verdict.failed_gate == "gate_1_predictive_viability"


def test_evaluate_promotion_gates_gate1_quantization_drop_failure() -> None:
    """Verify candidate is archived if INT8 accuracy drop exceeds tolerance."""
    criteria = PromotionCriteria(
        candidate_pr_auc=0.90,
        fp32_to_int8_drop=0.045,
        empirical_coverage=0.96,
        nominal_coverage=0.95,
        p95_latency_ms=40.0,
        quantization_drop_tol=0.03,
    )
    verdict = evaluate_promotion_gates(criteria, incumbent_pr_auc=0.85)
    assert verdict.alias == PromotionAlias.ARCHIVED
    assert verdict.gate == "quantization_drop_exceeded"
    assert verdict.failed_gate == "gate_1_quantization_fidelity"


def test_evaluate_promotion_gates_gate2_conformal_coverage_failure() -> None:
    """Verify candidate is archived if empirical coverage violates nominal guarantee minus margin."""
    criteria = PromotionCriteria(
        candidate_pr_auc=0.90,
        fp32_to_int8_drop=0.01,
        empirical_coverage=0.88,
        nominal_coverage=0.95,
        coverage_tolerance=0.05,
        p95_latency_ms=40.0,
    )
    verdict = evaluate_promotion_gates(criteria, incumbent_pr_auc=0.85)
    assert verdict.alias == PromotionAlias.ARCHIVED
    assert verdict.gate == "conformal_coverage_violation"
    assert verdict.failed_gate == "gate_2_conformal_coverage"


def test_evaluate_promotion_gates_gate3_latency_slo_failure() -> None:
    """Verify candidate is archived if quantized inference exceeds physical latency budget."""
    criteria = PromotionCriteria(
        candidate_pr_auc=0.90,
        fp32_to_int8_drop=0.01,
        empirical_coverage=0.95,
        nominal_coverage=0.95,
        p95_latency_ms=135.0,
        latency_slo_ms=120.0,
    )
    verdict = evaluate_promotion_gates(criteria, incumbent_pr_auc=0.85)
    assert verdict.alias == PromotionAlias.ARCHIVED
    assert verdict.gate == "latency_slo_violation"
    assert verdict.failed_gate == "gate_3_latency_slo"


def test_evaluate_promotion_gates_gate4_insufficient_traffic() -> None:
    """Verify candidate is staged as @Challenger when live production traffic is insufficient."""
    criteria = PromotionCriteria(
        candidate_pr_auc=0.90,
        fp32_to_int8_drop=0.01,
        empirical_coverage=0.96,
        nominal_coverage=0.95,
        p95_latency_ms=45.0,
        live_sample_count=20,
        min_telemetry_samples=50,
    )
    verdict = evaluate_promotion_gates(criteria, incumbent_pr_auc=0.85)
    assert verdict.alias == PromotionAlias.CHALLENGER
    assert verdict.gate == "insufficient_traffic"
    assert verdict.failed_gate == "gate_4_traffic_volume"


def test_evaluate_promotion_gates_gate4_drift_detected() -> None:
    """Verify candidate is staged as @Challenger when MMD squared permutation test detects distribution drift."""
    criteria = PromotionCriteria(
        candidate_pr_auc=0.90,
        fp32_to_int8_drop=0.01,
        empirical_coverage=0.96,
        nominal_coverage=0.95,
        p95_latency_ms=45.0,
        live_sample_count=100,
        min_telemetry_samples=50,
        exchangeability_p_value=0.002,
        drift_alpha=0.01,
    )
    verdict = evaluate_promotion_gates(criteria, incumbent_pr_auc=0.85)
    assert verdict.alias == PromotionAlias.CHALLENGER
    assert verdict.gate == "calibration_not_exchangeable"
    assert verdict.failed_gate == "gate_4_exchangeability_drift"


def test_evaluate_promotion_gates_all_pass_production() -> None:
    """Verify candidate is promoted to @Production when all 4 gates pass cleanly."""
    criteria = PromotionCriteria(
        candidate_pr_auc=0.90,
        fp32_to_int8_drop=0.01,
        empirical_coverage=0.96,
        nominal_coverage=0.95,
        p95_latency_ms=45.0,
        live_sample_count=150,
        min_telemetry_samples=50,
        exchangeability_p_value=0.45,
        drift_alpha=0.01,
    )
    verdict = evaluate_promotion_gates(criteria, incumbent_pr_auc=0.85)
    assert verdict.alias == PromotionAlias.PRODUCTION
    assert verdict.gate == "promoted_all_4_gates"
    assert verdict.failed_gate is None
    assert len(verdict.passed_gates) == 4


def test_promote_candidate_model_e2e_with_incumbent_demotion(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify promote_candidate_model assigns @Production and demotes incumbent @Production to @Challenger."""
    mock_client = MagicMock()
    prod_version = MagicMock(version="1", run_id="run_prod_1")
    prod_run = MagicMock()
    prod_run.data.metrics = {"val_promotion_score": 0.82}

    mock_client.get_model_version_by_alias.return_value = prod_version
    mock_client.get_run.return_value = prod_run

    monkeypatch.setattr(mlflow, "set_tag", MagicMock())
    monkeypatch.setattr(mlflow, "log_metric", MagicMock())

    target = ModelPromotionTarget(
        model_name="taxon_vision_classifier",
        version="2",
        client=mock_client,
        extractor="dinov3",
        run_id="run_cand_2",
    )
    criteria = PromotionCriteria(
        candidate_pr_auc=0.91,
        fp32_to_int8_drop=0.01,
        empirical_coverage=0.96,
        nominal_coverage=0.95,
        p95_latency_ms=50.0,
        live_sample_count=100,
        exchangeability_p_value=0.30,
    )

    verdict = promote_candidate_model(target, criteria)
    assert verdict.alias == PromotionAlias.PRODUCTION
    assert verdict.gate == "promoted_all_4_gates"

    mock_client.set_registered_model_alias.assert_any_call("taxon_vision_classifier", "Challenger", "1")
    mock_client.set_registered_model_alias.assert_any_call("taxon_vision_classifier", "Production", "2")
    mock_client.set_model_version_tag.assert_any_call("taxon_vision_classifier", "2", "gate", "promoted_all_4_gates")
    mock_client.set_model_version_tag.assert_any_call("taxon_vision_classifier", "2", "val_pr_auc", "0.9100")
    mock_client.set_model_version_tag.assert_any_call("taxon_vision_classifier", "2", "extractor", "dinov3")


def test_mirror_run_to_dagshub_skips_when_unconfigured_or_same_target(monkeypatch: pytest.MonkeyPatch) -> None:
    """Skip mirroring cleanly without remote RPC when DAGSHUB_TOKEN is unset or target matches source."""
    from taxon_vision.config import get_settings
    from taxon_vision.models.training.tracking import mirror_run_to_dagshub

    monkeypatch.delenv("DAGSHUB_TOKEN", raising=False)
    monkeypatch.delenv("MLFLOW_TRACKING_PASSWORD", raising=False)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://mlflow-server.taxon-vision.svc.cluster.local:5000")
    get_settings.cache_clear()
    try:
        assert mirror_run_to_dagshub("run-1") is None

        monkeypatch.setenv("DAGSHUB_TOKEN", "pat-123")
        monkeypatch.setenv("MLFLOW_TRACKING_URI", "https://dagshub.com/foersben/taxon-vision-mlops.mlflow")
        assert mirror_run_to_dagshub("run-1") == "run-1"
    finally:
        get_settings.cache_clear()


def test_mirror_run_to_dagshub_replicates_metrics_params_and_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Replicate source MLflow run metadata and artifacts to public DagsHub tracking server."""
    from types import SimpleNamespace

    from taxon_vision.config import get_settings
    from taxon_vision.models.training import tracking as tracking_mod

    report_file = tmp_path / "summary.txt"
    report_file.write_text("ok", encoding="utf-8")

    monkeypatch.setenv("DAGSHUB_TOKEN", "secret-pat")
    monkeypatch.setenv("TAXON_MLFLOW__TRACKING_URI", "http://mlflow-server.taxon-vision.svc.cluster.local:5000")
    get_settings.cache_clear()

    calls: list[tuple[str, str]] = []

    class _FakeClient:
        def __init__(self, tracking_uri: str) -> None:
            self.tracking_uri = tracking_uri
            calls.append(("init", tracking_uri))

        def get_run(self, run_id: str) -> SimpleNamespace:
            return SimpleNamespace(
                info=SimpleNamespace(run_id=run_id, experiment_id="exp-1", run_name="eval-run", start_time=1000),
                data=SimpleNamespace(
                    params={"backbone": "dinov3"},
                    tags={"stage": "evaluate", "mlflow.note.content": "keep"},
                    metrics={"eval.top1_accuracy": 0.94},
                ),
            )

        def get_experiment(self, experiment_id: str) -> SimpleNamespace:
            return SimpleNamespace(experiment_id=experiment_id, name="taxon_vision_evaluation")

        def get_metric_history(self, _run_id: str, key: str) -> list[SimpleNamespace]:
            return [SimpleNamespace(key=key, value=0.94, timestamp=1000, step=1)]

        def get_experiment_by_name(self, _name: str) -> None:
            return None

        def create_experiment(self, name: str) -> str:
            calls.append(("create_experiment", name))
            return "42"

        def create_run(self, *, run_name: str = "", **_kwargs: object) -> SimpleNamespace:
            calls.append(("create_run", run_name))
            return SimpleNamespace(info=SimpleNamespace(run_id="dst-run-99"))

        def log_batch(
            self,
            *,
            metrics: list[object] = (),
            params: list[object] = (),
            tags: list[object] = (),
            **_kwargs: object,
        ) -> None:
            calls.append(("log_batch", f"{len(metrics)}:{len(params)}:{len(tags)}"))

        def log_artifact(self, _run_id: str, local_path: str) -> None:
            calls.append(("log_artifact", local_path))

        def set_terminated(self, run_id: str, status: str = "FINISHED") -> None:
            calls.append(("set_terminated", f"{run_id}:{status}"))

    monkeypatch.setattr(tracking_mod, "MlflowClient", _FakeClient)
    try:
        dst_id = tracking_mod.mirror_run_to_dagshub("src-run-1", artifact_paths=[report_file])
        assert dst_id == "dst-run-99"
        assert ("init", "https://dagshub.com/foersben/taxon-vision-mlops.mlflow") in calls
        assert ("create_experiment", "taxon_vision_evaluation") in calls
        assert ("log_batch", "0:1:0") in calls
        assert ("log_batch", "1:0:0") in calls
        assert ("log_batch", "0:0:4") in calls
        assert ("log_artifact", str(report_file)) in calls
        assert ("set_terminated", "dst-run-99:FINISHED") in calls
    finally:
        get_settings.cache_clear()
