# SPDX-FileCopyrightText: 2026 Benjamin Förster
# SPDX-License-Identifier: MIT
"""Configuration loading, YAML layering, and environment override parsing."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from taxon_vision.config.schemas import Settings


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
        "min_confidence": "min_confidence",
        "min_confidence_threshold": "min_confidence_threshold",
        "ood_tpr": "ood_tpr",
    }
    override = {target_k: calib[yaml_k] for yaml_k, target_k in mapping.items() if yaml_k in calib}
    if "min_confidence" in override and "min_confidence_threshold" not in override:
        override["min_confidence_threshold"] = override["min_confidence"]
    elif "min_confidence_threshold" in override and "min_confidence" not in override:
        override["min_confidence"] = override["min_confidence_threshold"]
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
    """Apply targeted unnested environment variable overrides for test isolation and Optuna storage.

    Why:
        Enables seamless runtime configuration of test modes, MLflow tracking, and database URIs without nested delimiters.

    How:
        Checks for TAXON_ALLOW_UNTRAINED_FALLBACK, TAXON_FLAGGED_STORAGE_DIR, TAXON_OPTUNA_STORAGE_URI,
        TAXON_MLFLOW_TRACKING_URI, and MLFLOW_TRACKING_URI, injecting them into the merged dictionary.

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

    optuna_overrides: dict[str, Any] = {}
    env_optuna_uri = os.getenv("TAXON_OPTUNA_STORAGE_URI")
    if env_optuna_uri is not None:
        optuna_overrides["optuna_storage_uri"] = env_optuna_uri

    if optuna_overrides:
        config_data = _deep_merge(config_data, {"optuna": optuna_overrides})

    mlflow_overrides: dict[str, Any] = {}
    env_mlflow_uri = os.getenv("TAXON_MLFLOW_TRACKING_URI") or os.getenv("MLFLOW_TRACKING_URI")
    if env_mlflow_uri is not None:
        mlflow_overrides["tracking_uri"] = env_mlflow_uri

    if mlflow_overrides:
        config_data = _deep_merge(config_data, {"mlflow": mlflow_overrides})

    return config_data


def get_params_path() -> Path:
    """Resolve the canonical params.yaml path, checking repository root first then config/.

    Returns:
        Path: Canonical path to params.yaml.
    """
    repo_root = Path(__file__).resolve().parents[3]
    root_params = repo_root / "params.yaml"
    if root_params.is_file():
        return root_params
    return repo_root / "config" / "params.yaml"


@lru_cache(maxsize=1)
def get_settings(config_path: str | Path | None = None) -> Settings:
    """Retrieve the validated global application configuration settings.

    Loads base configuration from `config/default_config.yaml`. Layers `params.yaml` and `conformal_config.yaml`
    if present in the configuration root. If an explicit `config_path` is provided, its values are layered on top,
    followed by any active `TAXON_` environment variables.

    Args:
        config_path (str | Path | None, optional): Explicit filesystem path to YAML configuration overrides.
            If None, resolves from the `TAXON_CONFIG_PATH` environment variable. Defaults to None.

    Returns:
        Settings: The validated application configuration instance.

    Raises:
        FileNotFoundError: If the base configuration file does not exist.
        pydantic.ValidationError: If the loaded configuration fails schema constraints.
    """
    repo_root = Path(__file__).resolve().parents[3]
    base_path = repo_root / "config" / "default_config.yaml"
    if not base_path.is_file():
        raise FileNotFoundError(f"Base configuration file not found at: {base_path}")

    config_data = _load_yaml_dict(base_path)
    config_dir = base_path.parent

    # Layer params.yaml if present (check repo root first, then config_dir)
    params_path = get_params_path()
    params_data = _load_yaml_dict(params_path)
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


def reset_settings() -> None:
    """Clear the cached settings instance to force re-reading configuration files and environment."""
    get_settings.cache_clear()
