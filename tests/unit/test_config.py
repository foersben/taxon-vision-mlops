# SPDX-FileCopyrightText: 2026 Benjamin Förster
# SPDX-License-Identifier: MIT
"""Unit tests for centralized configuration and settings parsing."""

from __future__ import annotations

from pathlib import Path

import pytest

from taxon_vision.config import Settings, get_settings


def test_default_settings_loading(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify default system settings are populated from configuration YAML."""
    monkeypatch.delenv("TAXON_OPTUNA_STORAGE_URI", raising=False)
    get_settings.cache_clear()
    settings = get_settings()
    assert isinstance(settings, Settings)
    assert settings.model.image_size == 224
    assert settings.model.channels == 3
    assert "mobilenetv4_conv_small" in settings.model.backbone_registry
    assert "dinov3" in settings.model.backbone_registry
    assert "bioclip-2" in settings.model.backbone_registry
    assert settings.model.default_extractor == "dinov3"
    assert settings.mlflow.experiment_name == "taxon-vision-head-tuning"
    assert settings.conformal.alpha == 0.05
    assert settings.conformal.k_max == 3
    assert settings.conformal.nominal_coverage == 0.95
    assert settings.conformal.min_confidence_threshold == 0.60
    assert settings.conformal.min_confidence == 0.60
    assert settings.conformal.ood_tpr == 0.95
    assert settings.evaluation.latency_sla_ms == 25.0
    assert settings.ood.energy_threshold == -2.40
    assert settings.ood.entropy_threshold == 1.60
    assert settings.training.epochs == 30
    assert settings.training.batch_size == 32
    assert settings.training.learning_rate == 0.001
    assert settings.training.weight_decay == 0.0001
    assert settings.training.cb_beta == 0.9999
    assert settings.training.dropout == 0.2
    assert settings.training.noise_std == 0.01
    assert settings.training.early_stopping_patience == 5
    assert settings.training.reduce_lr_patience == 2
    assert settings.training.warmup_epochs == 3
    assert settings.training.val_split == 0.10
    assert settings.training.cal_split == 0.10
    assert settings.training.test_split == 0.10
    assert settings.training.random_seed == 42
    assert settings.training.extractor == "dinov3"
    assert settings.training.tune is False
    assert len(settings.ingestion.taxa_to_fetch) == 10
    assert settings.ingestion.per_page == 200
    assert settings.streaming.http_max_retries == 5
    assert settings.streaming.http_backoff_factor == 0.5
    assert settings.streaming.http_max_connections == 16
    assert settings.streaming.http_keepalive_connections == 8
    assert settings.streaming.http_timeout_seconds == 15.0
    assert settings.streaming.min_extraction_success_ratio == 0.95
    assert settings.optuna.optuna_storage_uri == "sqlite:///data/optuna/optuna_studies.db"
    assert settings.optuna.optuna_study_prefix == "taxon_vision"
    assert settings.optuna.n_trials == 30
    assert settings.optuna.epochs == 30
    assert settings.optuna.min_resource == 3
    assert settings.optuna.reduction_factor == 2


def test_custom_settings_override(tmp_path: Path) -> None:
    """Verify custom settings YAML properly overrides defaults."""
    custom_yaml = tmp_path / "custom.yaml"
    custom_yaml.write_text(
        """
model:
  image_size: 336
  default_extractor: "bioclip-2"
mlflow:
  experiment_name: "custom-experiment"
"""
    )
    custom_settings = get_settings(config_path=str(custom_yaml))
    assert custom_settings.model.image_size == 336
    assert custom_settings.model.default_extractor == "bioclip-2"
    assert custom_settings.mlflow.experiment_name == "custom-experiment"


def test_optuna_storage_uri_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify TAXON_OPTUNA_STORAGE_URI overrides the default optuna storage URI.

    Why:
        Validates unnested environment variable injection for headless Optuna orchestrations.

    How:
        Sets TAXON_OPTUNA_STORAGE_URI via monkeypatch, clears the settings cache, and asserts the overridden URI is loaded.

    Args:
        monkeypatch (pytest.MonkeyPatch): Pytest environment isolation fixture.
    """
    test_uri = "sqlite:///data/optuna/test_override.db"
    monkeypatch.setenv("TAXON_OPTUNA_STORAGE_URI", test_uri)
    get_settings.cache_clear()
    settings = get_settings()
    assert settings.optuna.optuna_storage_uri == test_uri
    get_settings.cache_clear()


def test_mlflow_tracking_uri_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify MLFLOW_TRACKING_URI and TAXON_MLFLOW_TRACKING_URI override default MLflow tracking URI."""
    test_uri = "sqlite:///tmp/mlflow_override.db"
    monkeypatch.delenv("TAXON_MLFLOW_TRACKING_URI", raising=False)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", test_uri)
    get_settings.cache_clear()
    settings = get_settings()
    assert settings.mlflow.tracking_uri == test_uri

    taxon_uri = "sqlite:///tmp/taxon_mlflow_override.db"
    monkeypatch.setenv("TAXON_MLFLOW_TRACKING_URI", taxon_uri)
    get_settings.cache_clear()
    settings = get_settings()
    assert settings.mlflow.tracking_uri == taxon_uri
    get_settings.cache_clear()
