# SPDX-FileCopyrightText: 2026 Benjamin Förster
# SPDX-License-Identifier: MIT
"""Centralized application configuration schemas and strict validation.

This module defines the type-safe validation schemas for the TaxonVision platform. To maintain the Single Source of Truth (SSOT) and DRY principles, default configuration values are declared strictly in `config/default_config.yaml`. This module serves as the schema validator and runtime accessor, preventing configuration drift between code defaults and YAML declarations.

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
from typing import Any, Literal

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
        mean (list[float]): Mean values for channel-wise normalization.
        std (list[float]): Standard deviation values for channel-wise normalization.
        onnx_model_path (str): Relative or absolute path to exported quantized ONNX checkpoint.
        temperature (float): Softmax logit scaling temperature for probability calibration.
        backbone_registry (dict[str, str]): Mapping of canonical model identifiers to Hugging Face Hub or timm model definitions.
        head_checkpoint_path (str): Relative or absolute path to the trained classification head checkpoint.
        inference_backend (str): Serving backend: 'onnx' (require the ONNX artefact), 'torch' (PyTorch only) or 'auto' (ONNX when the artefact exists, otherwise PyTorch).
        quantization_mode (str): INT8 mode used by the exporter: 'static' (Conv/ViT graphs), 'dynamic' (linear-only graphs, data-free smoke tests) or 'auto'.
        calibration_dir (str): Directory of representative images used for static PTQ calibration (a subset of the calibration split).
        calibration_samples (int): Maximum number of calibration images fed to static PTQ.
        allow_untrained_fallback (bool): Whether to permit uninitialized/fallback weights for offline testing.
        prefer_onnx_runtime (bool): Whether to prioritize ONNX Runtime over PyTorch for inference.
        onnx_opset_version (int): ONNX target operator set version for graph serialization.
        cam_numerical_epsilon (float): Small floating point epsilon to prevent division by zero in CAM normalization.
        ood_temperature (float): Temperature parameter for Helmholtz free energy OOD detection.
        ood_energy_threshold (float): Energy score threshold for flagging observations as Out-of-Distribution.
        flagged_storage_dir (str): Filesystem directory for persisting flagged OOD and ambiguous observations.
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
    taxa_catalog_path: str = "config/taxa_catalog.yaml"
    eval_resize_ratio: float = Field(default=256 / 224, gt=1.0)
    inference_backend: Literal["auto", "onnx", "torch"] = "auto"
    quantization_mode: Literal["auto", "static", "dynamic"] = "auto"
    calibration_dir: str = "data/sample"
    calibration_samples: int = Field(default=128, gt=0)
    allow_untrained_fallback: bool = False
    prefer_onnx_runtime: bool = True
    onnx_opset_version: int = Field(default=17, gt=0)
    cam_numerical_epsilon: float = Field(default=1.0e-8, gt=0.0)
    ood_temperature: float = Field(default=1.0, gt=0.0)
    ood_energy_threshold: float = Field(default=-2.40)
    flagged_storage_dir: str = "data/flagged_observations"


class TrainingConfig(BaseModel):
    """Parameters governing representation caching and linear classification head training.

    Attributes:
        epochs (int): Default number of training epochs for linear probing.
        batch_size (int): Mini-batch size for training and feature extraction.
        learning_rate (float): Initial learning rate for optimization.
        val_split (float): Validation split fraction.
        random_seed (int): Deterministic PRNG seed for reproducible splits.
        extractor (str): Default backbone identifier for training.
    """

    epochs: int = Field(default=5, gt=0)
    batch_size: int = Field(default=32, gt=0)
    learning_rate: float = Field(default=0.001, gt=0.0)
    val_split: float = Field(default=0.2, ge=0.0, lt=1.0)
    random_seed: int = Field(default=42)
    extractor: str = "dinov3"


class IngestionConfig(BaseModel):
    """Configuration for raw observation metadata and image ingestion.

    Attributes:
        taxa_to_fetch (list[int]): Authoritative iNaturalist taxon IDs to download.
        per_page (int): Number of observation records requested per API page.
    """

    taxa_to_fetch: list[int] = Field(default_factory=list)
    per_page: int = Field(default=200, gt=0)


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
        q_hat (float): Calibrated non-conformity scalar quantile threshold.
        k_max (int): Maximum prediction set cardinality before routing to human expert triage.
        quantile_threshold (float): Pre-calibrated non-conformity scalar quantile threshold (alias for q_hat).
        nominal_coverage (float): Target marginal coverage rate (1.0 - alpha).
        method (str): Conformal prediction method (e.g. 'split-conformal').
        empty_set_action (str): Action taken when conformal set is empty.
    """

    alpha: float = Field(default=0.05, gt=0.0, lt=1.0)
    q_hat: float = Field(default=0.85, gt=0.0, lt=1.0)
    k_max: int = Field(default=3, gt=0)
    quantile_threshold: float = Field(default=0.85, gt=0.0, lt=1.0)
    nominal_coverage: float = Field(default=0.95, gt=0.0, lt=1.0)
    method: str = "split-conformal"
    empty_set_action: str = "route_to_human_triage"


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

    Aggregates subsystem configurations. Default values are populated from `config/default_config.yaml` and can be overridden by environment variables.
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
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    ingestion: IngestionConfig = Field(default_factory=IngestionConfig)


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


def _load_conformal_override(path: Path) -> dict[str, Any]:
    """Parse calibration parameters from conformal_config.yaml if present.

    Args:
        path (Path): Path to conformal_config.yaml.

    Returns:
        dict[str, Any]: Dictionary with conformal sub-key mapping, or empty dict.
    """
    if not path.is_file():
        return {}
    calib = _load_yaml_dict(path).get("calibration", {})
    mapping = {
        "alpha": "alpha",
        "q_hat": "q_hat",
        "max_set_size_for_auto_accept": "k_max",
        "nominal_coverage": "nominal_coverage",
        "method": "method",
        "empty_set_action": "empty_set_action",
    }
    override = {target_k: calib[yaml_k] for yaml_k, target_k in mapping.items() if yaml_k in calib}
    if "q_hat" in override:
        override["quantile_threshold"] = override["q_hat"]
    elif "quantile_threshold" in calib:
        override["quantile_threshold"] = calib["quantile_threshold"]
        override["q_hat"] = calib["quantile_threshold"]
    return {"conformal": override} if override else {}


def _resolve_custom_override(custom_path_str: str | None, base_path: Path) -> dict[str, Any]:
    """Load configuration overrides from custom user path or environment variable.

    Args:
        custom_path_str (str | None): Optional path string from argument or env var.
        base_path (Path): Resolved canonical default configuration path.

    Returns:
        dict[str, Any]: Parsed override mapping, or empty dict if not applicable.
    """
    if not custom_path_str:
        return {}
    custom_path = Path(custom_path_str)
    if custom_path.resolve() == base_path.resolve() or not custom_path.is_file():
        return {}
    return _load_yaml_dict(custom_path)


def _apply_direct_env_overrides(config_data: dict[str, Any]) -> dict[str, Any]:
    """Apply targeted unnested environment variable overrides for test isolation.

    Args:
        config_data (dict[str, Any]): Merged configuration dictionary.

    Returns:
        dict[str, Any]: Updated configuration dictionary with direct environment overrides applied.
    """
    model_overrides: dict[str, Any] = {}
    env_untrained = os.getenv("TAXON_ALLOW_UNTRAINED_FALLBACK")
    if env_untrained is not None:
        model_overrides["allow_untrained_fallback"] = env_untrained.strip().lower() in ("true", "1", "yes")

    env_storage = os.getenv("TAXON_FLAGGED_STORAGE_DIR")
    if env_storage is not None:
        model_overrides["flagged_storage_dir"] = env_storage

    if model_overrides:
        config_data = _deep_merge(config_data, {"model": model_overrides})
    return config_data


@lru_cache(maxsize=1)
def get_settings(config_path: str | Path | None = None) -> Settings:
    """Retrieve the validated global application configuration settings.

    Loads base configuration from `config/default_config.yaml`. Layers `params.yaml` and `conformal_config.yaml` if present in the configuration root. If an explicit `config_path` is provided, its values are layered on top, followed by any active `TAXON_` environment variables.

    Args:
        config_path (str | Path | None, optional): Explicit filesystem path to YAML configuration overrides. If None, resolves from the `TAXON_CONFIG_PATH` environment variable. Defaults to None.

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
    config_dir = base_path.parent

    # Layer params.yaml if present
    params_data = _load_yaml_dict(config_dir / "params.yaml")
    if params_data:
        config_data = _deep_merge(config_data, params_data)

    # Layer conformal_config.yaml if present
    conformal_override = _load_conformal_override(config_dir / "conformal_config.yaml")
    if conformal_override:
        config_data = _deep_merge(config_data, conformal_override)

    # Layer custom override if present
    custom_override = _resolve_custom_override(
        str(config_path) if config_path is not None else os.getenv("TAXON_CONFIG_PATH"),
        base_path,
    )
    if custom_override:
        config_data = _deep_merge(config_data, custom_override)

    config_data = _apply_direct_env_overrides(config_data)
    return Settings(**config_data)
