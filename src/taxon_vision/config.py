# SPDX-FileCopyrightText: 2026 Benjamin Förster
# SPDX-License-Identifier: MIT
"""Centralized application configuration schemas and strict validation.

This module defines the type-safe validation schemas for the TaxonVision platform.
To maintain the Single Source of Truth (SSOT) and DRY principles, default configuration
values are declared strictly in `config/default_config.yaml`. This module serves as
the schema validator and runtime accessor, preventing configuration drift between
code defaults and YAML declarations.

Settings are loaded with precedence:
1. Environment variables (prefixed with `TAXON_`, e.g. `TAXON_MODEL__IMAGE_SIZE=336`).
2. Custom YAML configuration overrides (if supplied).
3. Base declarative configuration (`config/default_config.yaml`).

Typical usage example:
    from taxon_vision.config import get_settings

    settings = get_settings()
    backbone_name = settings.model.default_extractor
    image_dim = settings.model.image_size
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ServiceConfig(BaseModel):
    """FastAPI service and HTTP server operational settings schema.

    Attributes:
        name (str): Human-readable name of the web service.
        version (str): Application semantic version string.
        port (int): Network TCP port for HTTP server binding.
        host (str): Network IP address interface to bind.
        max_image_upload_bytes (int): Maximum allowable upload payload size in bytes.
        allowed_mime_types (list[str]): Allowed HTTP MIME content types for uploaded images.
    """

    name: str
    version: str
    port: int = Field(gt=0, lt=65536)
    host: str
    max_image_upload_bytes: int = Field(gt=0)
    allowed_mime_types: list[str]


class ModelConfig(BaseModel):
    """Computer vision model architectures, dimensions, and checkpoint paths schema.

    Attributes:
        default_extractor (str): Default vision backbone identifier.
        fallback_extractor (str): Lightweight un-pretrained architecture for offline CI runs.
        available_extractors (list[str]): List of supported feature extractor architectures.
        image_size (int): Standard square image spatial dimension (height and width in pixels).
        channels (int): Number of image input color channels.
        onnx_model_path (str): Relative or absolute path to exported quantized ONNX checkpoint.
        temperature (float): Softmax logit scaling temperature for probability calibration.
        backbone_registry (dict[str, str]): Mapping of canonical model identifiers to
            Hugging Face Hub or timm model definitions.
    """

    default_extractor: str
    fallback_extractor: str
    available_extractors: list[str]
    image_size: int = Field(gt=0)
    channels: int = Field(default=3, gt=0)
    mean: list[float]
    std: list[float]
    onnx_model_path: str
    temperature: float = Field(gt=0.0)
    backbone_registry: dict[str, str]
    head_checkpoint_path: str


class MLflowConfig(BaseModel):
    """Experiment tracking and DagsHub MLflow remote server configuration schema.

    Attributes:
        tracking_uri (str): MLflow tracking server URI.
        experiment_name (str): Target experiment grouping name for training runs.
    """

    tracking_uri: str
    experiment_name: str


class ConformalConfig(BaseModel):
    r"""Statistical conformal prediction coverage configuration schema.

    Attributes:
        alpha (float): Conformal error rate bound (nominal coverage is $1 - \alpha$).
        k_max (int): Maximum prediction set cardinality before routing to human expert triage.
        quantile_threshold (float): Pre-calibrated non-conformity scalar quantile threshold.
    """

    alpha: float = Field(gt=0.0, lt=1.0)
    k_max: int = Field(gt=0)
    quantile_threshold: float = Field(gt=0.0, lt=1.0)


class OODConfig(BaseModel):
    """Out-of-Distribution (OOD) energy and entropy detection thresholds schema.

    Attributes:
        energy_threshold (float): Free energy score threshold below which observations are flagged as OOD.
        entropy_threshold (float): Shannon predictive entropy threshold for uncertainty triage.
    """

    energy_threshold: float
    entropy_threshold: float


class Settings(BaseSettings):
    """Global application settings container with strict schema validation.

    Aggregates subsystem configurations. Default values are populated from
    `config/default_config.yaml` and can be overridden by environment variables.
    """

    model_config = SettingsConfigDict(
        env_prefix="TAXON_",
        env_nested_delimiter="__",
        extra="ignore",
    )

    service: ServiceConfig
    model: ModelConfig
    mlflow: MLflowConfig
    conformal: ConformalConfig
    ood: OODConfig


def _load_yaml_dict(path: Path) -> dict[str, Any]:
    """Safely load and parse YAML configuration dictionary from disk.

    Args:
        path (Path): Filesystem path to the YAML configuration file.

    Returns:
        dict[str, Any]: Parsed configuration mapping, or empty dict if missing or invalid.
    """
    if not path.is_file():
        return {}
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
        return data if isinstance(data, dict) else {}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge override dictionary into base dictionary.

    Args:
        base (dict[str, Any]): Base dictionary.
        override (dict[str, Any]): Override dictionary to overlay.

    Returns:
        dict[str, Any]: Newly merged dictionary preserving untouched base keys.
    """
    merged = base.copy()
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


@lru_cache(maxsize=1)
def get_settings(config_path: str | Path | None = None) -> Settings:
    """Retrieve the validated global application configuration settings.

    Loads base configuration from `config/default_config.yaml`. If an explicit
    `config_path` is provided, its values are layered on top of the base configuration,
    followed by any active `TAXON_` environment variables.

    Args:
        config_path (str | Path | None, optional): Explicit filesystem path to YAML configuration
            overrides. If None, resolves from the `TAXON_CONFIG_PATH` environment variable.
            Defaults to None.

    Returns:
        Settings: The validated application configuration instance.

    Raises:
        FileNotFoundError: If the base configuration file does not exist.
        pydantic.ValidationError: If the loaded configuration fails schema constraints.
    """
    base_path = Path(__file__).resolve().parent.parent.parent / "config" / "default_config.yaml"
    if not base_path.is_file():
        raise FileNotFoundError(f"Base configuration file not found at: {base_path}")

    config_data = _load_yaml_dict(base_path)

    # Check for custom override path
    custom_path_str = config_path or os.getenv("TAXON_CONFIG_PATH")
    if custom_path_str:
        custom_path = Path(custom_path_str)
        if custom_path.resolve() != base_path.resolve() and custom_path.is_file():
            override_data = _load_yaml_dict(custom_path)
            config_data = _deep_merge(config_data, override_data)

    return Settings(**config_data)
