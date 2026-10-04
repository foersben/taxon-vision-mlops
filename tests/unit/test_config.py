# SPDX-FileCopyrightText: 2026 Benjamin Förster
# SPDX-License-Identifier: MIT
"""Unit tests for centralized configuration and settings parsing."""

from pathlib import Path

from taxon_vision.config import Settings, get_settings


def test_default_settings_loading() -> None:
    settings = get_settings()
    assert isinstance(settings, Settings)
    assert settings.model.image_size == 224
    assert settings.model.channels == 3
    assert "mobilenetv4_conv_small" in settings.model.backbone_registry
    assert "bioclip-2" in settings.model.backbone_registry
    assert settings.model.default_extractor == "mobilenetv4_conv_small"
    assert settings.mlflow.experiment_name == "taxon-vision-head-tuning"
    assert settings.conformal.alpha == 0.05
    assert settings.ood.energy_threshold == -12.5


def test_custom_settings_override(tmp_path: Path) -> None:
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
