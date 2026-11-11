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

    Why:
        Centralises safe YAML deserialisation across pipeline parameter files, calibration tables, and service configuration files with deterministic error suppression.

    How:
        Verifies filesystem existence and employs PyYAML safe loader to produce an in-memory dictionary representation, returning an empty dictionary on missing files or non-dictionary payloads.

    Args:
        path: Filesystem path to the YAML configuration file.

    Returns:
        Parsed configuration mapping, or empty dict if missing or invalid.
    """
    if not path.is_file():
        return {}
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
        return data if isinstance(data, dict) else {}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge override dictionary into base dictionary.

    Why:
        Enables multi-tiered configuration layering where environment variables, DVC parameters, and user overrides selectively mutate deeply nested settings without clobbering sibling configuration keys.

    How:
        Performs depth-first dictionary recursion, recursively combining matching sub-dictionaries while overwriting terminal scalar or sequence values in a shallow-copied container.

    Args:
        base: Base dictionary.
        override: Override dictionary to overlay.

    Returns:
        Newly merged dictionary preserving untouched base keys.
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

    Why:
        Preserves alignment between offline Stage 6 calibration outputs and runtime inference serving, preventing split-brain discrepancies where the service defaults to uncalibrated non-conformity quantiles.

    How:
        Reads calibration parameters from the specified path, translates legacy and alias keys into canonical schema identifiers, and wraps the resulting dictionary under the top-level conformal key.

    Args:
        path: Path to conformal_config.yaml.

    Returns:
        Dictionary with conformal sub-key mapping, or empty dict.
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

    Why:
        Allows ad-hoc experiments, staging overlays, and alternative backbone evaluations to inject distinct parameter files without modifying default configuration files tracked by version control.

    How:
        Validates whether the custom path string exists, ensures it does not circularly point to the base configuration file, and parses it via the YAML loader.

    Args:
        custom_path_str: Optional path string from argument or env var.
        base_path: Resolved canonical default configuration path.

    Returns:
        Parsed override mapping, or empty dict if not applicable.
    """
    if not custom_path_str:
        return {}
    custom_path = Path(custom_path_str)
    if custom_path.resolve() == base_path.resolve() or not custom_path.is_file():
        return {}
    return _load_yaml_dict(custom_path)


def _extract_model_env_overrides() -> dict[str, Any]:
    """Extract model subsystem configuration overrides from process environment variables.

    Why:
        Containerised execution environments and integration test harnesses require dynamic control over untrained model fallback modes, persistent storage directories, and ONNX Runtime intra-op thread pools without mutating configuration files on disk.

    How:
        Inspects TAXON_ALLOW_UNTRAINED_FALLBACK, TAXON_FLAGGED_STORAGE_DIR, and TAXON_MODEL__ONNX_INTRA_OP_THREADS, normalising boolean flags and integer thread counts into a typed sub-dictionary.

    Returns:
        Mapping of model configuration keys to parsed environment override values.
    """
    overrides: dict[str, Any] = {}
    env_untrained = os.getenv("TAXON_ALLOW_UNTRAINED_FALLBACK")
    if env_untrained is not None:
        overrides["allow_untrained_fallback"] = env_untrained.strip().lower() in ("true", "1", "yes")

    env_storage = os.getenv("TAXON_FLAGGED_STORAGE_DIR")
    if env_storage is not None:
        overrides["flagged_storage_dir"] = env_storage

    env_intra = os.getenv("TAXON_MODEL__ONNX_INTRA_OP_THREADS") or os.getenv("ORT_INTRA_OP_NUM_THREADS")
    if env_intra is not None:
        try:
            overrides["onnx_intra_op_threads"] = int(env_intra)
        except ValueError:
            pass
    return overrides


def _extract_service_env_overrides() -> dict[str, Any]:
    """Extract web microservice and caching configuration overrides from environment variables.

    Why:
        Distributed serving pods in Kubernetes clusters connect to shared ephemeral cache instances (Redis) to decouple multipart file uploads from asynchronous Class Activation Map (CAM) explanation routes across replica pods.

    How:
        Evaluates TAXON_SERVICE__REDIS_URL, TAXON_SERVICE_REDIS_URL, and legacy TAXON_REDIS_URL, returning a service override dictionary containing the resolved Redis connection URL if declared.

    Returns:
        Mapping containing service configuration overrides, or an empty dictionary if no variables are set.
    """
    env_redis = (
        os.getenv("TAXON_SERVICE__REDIS_URL") or os.getenv("TAXON_SERVICE_REDIS_URL") or os.getenv("TAXON_REDIS_URL")
    )
    return {"redis_url": env_redis} if env_redis is not None else {}


def _extract_optuna_mlflow_overrides() -> dict[str, dict[str, Any]]:
    """Extract experiment tracking and hyperparameter optimisation storage targets from environment variables.

    Why:
        Decouples local developer workstation SQLite storage targets from central cluster PostgreSQL databases and MLflow tracking servers without requiring manual parameter file editing across execution contexts.

    How:
        Reads TAXON_OPTUNA_STORAGE_URI and TAXON_MLFLOW_TRACKING_URI, constructing discrete sectional override mappings for deep merging into root configuration dictionaries.

    Returns:
        Nested dictionary containing optuna and mlflow sectional overrides.
    """
    res: dict[str, dict[str, Any]] = {}
    env_optuna = os.getenv("TAXON_OPTUNA_STORAGE_URI")
    if env_optuna is not None:
        res["optuna"] = {"optuna_storage_uri": env_optuna}

    env_mlflow = os.getenv("TAXON_MLFLOW_TRACKING_URI") or os.getenv("MLFLOW_TRACKING_URI")
    if env_mlflow is not None:
        res["mlflow"] = {"tracking_uri": env_mlflow}
    return res


def _extract_telemetry_env_overrides() -> dict[str, Any]:
    """Extract production telemetry database connection strings and fallback toggles from environment variables.

    Why:
        Enables runtime switching between local in-memory telemetry buffers and persistent PostgreSQL relational storage backends while guaranteeing test isolation without disk side-effects.

    How:
        Evaluates TAXON_TELEMETRY_DATABASE_URL and TAXON_TELEMETRY_ALLOW_IN_MEMORY_FALLBACK, parsing truthy boolean strings and constructing the telemetry override sub-dictionary.

    Returns:
        Mapping containing telemetry configuration keys and parsed override values.
    """
    overrides: dict[str, Any] = {}
    env_uri = os.getenv("TAXON_TELEMETRY_DATABASE_URL") or os.getenv("TAXON_TELEMETRY__DATABASE_URL")
    if env_uri is not None:
        overrides["database_url"] = env_uri

    env_fallback = os.getenv("TAXON_TELEMETRY_ALLOW_IN_MEMORY_FALLBACK") or os.getenv(
        "TAXON_TELEMETRY__ALLOW_IN_MEMORY_FALLBACK"
    )
    if env_fallback is not None:
        overrides["allow_in_memory_fallback"] = env_fallback.strip().lower() in ("true", "1", "yes")
    return overrides


def _apply_direct_env_overrides(config_data: dict[str, Any]) -> dict[str, Any]:
    """Apply targeted unnested environment variable overrides for test isolation and Optuna storage.

    Why:
        Enables seamless runtime configuration of test modes, MLflow tracking, and database URIs without nested delimiters.

    How:
        Extracts sectional environment overrides for model, service, optuna, mlflow, and telemetry sub-dictionaries, deep-merging them into the unified config container.

    Args:
        config_data: Merged configuration dictionary.

    Returns:
        Updated configuration dictionary with direct environment overrides applied.
    """
    model_ovr = _extract_model_env_overrides()
    if model_ovr:
        config_data = _deep_merge(config_data, {"model": model_ovr})

    service_ovr = _extract_service_env_overrides()
    if service_ovr:
        config_data = _deep_merge(config_data, {"service": service_ovr})

    for section, ovr in _extract_optuna_mlflow_overrides().items():
        config_data = _deep_merge(config_data, {section: ovr})

    telemetry_ovr = _extract_telemetry_env_overrides()
    if telemetry_ovr:
        config_data = _deep_merge(config_data, {"telemetry": telemetry_ovr})

    return config_data


def _resolve_workspace_root() -> Path:
    """Resolve the active workspace root directory across source checkouts and packaged installs.

    Why:
        Preserves deterministic path resolution across editable developer workspaces, containerised execution paths, and wheel installations without hardcoding host filesystem assumptions.

    How:
        Tests environment overrides (TAXON_WORKSPACE_ROOT, TAXON_CONFIG_ROOT), then checks the package parent directory hierarchy and the active current working directory for config/default_config.yaml.

    Returns:
        Canonical directory containing the active config/default_config.yaml.
    """
    env_root = os.getenv("TAXON_WORKSPACE_ROOT") or os.getenv("TAXON_CONFIG_ROOT")
    if env_root:
        candidate = Path(env_root).resolve()
        if (candidate / "config" / "default_config.yaml").is_file():
            return candidate
        if (candidate / "default_config.yaml").is_file():
            return candidate.parent

    source_root = Path(__file__).resolve().parents[3]
    if (source_root / "config" / "default_config.yaml").is_file():
        return source_root

    cwd = Path.cwd().resolve()
    if (cwd / "config" / "default_config.yaml").is_file():
        return cwd

    return source_root


def get_params_path() -> Path:
    """Resolve the canonical params.yaml path, checking workspace root first then config/.

    Why:
        DVC pipeline definitions place params.yaml at the repository root, while container deployments may mount it under config/ or the application working directory.

    How:
        Probes the resolved workspace root for params.yaml, falling back to config/params.yaml if absent.

    Returns:
        Canonical path to params.yaml.
    """
    repo_root = _resolve_workspace_root()
    root_params = repo_root / "params.yaml"
    if root_params.is_file():
        return root_params
    return repo_root / "config" / "params.yaml"


@lru_cache(maxsize=1)
def get_settings(config_path: str | Path | None = None) -> Settings:
    """Retrieve and cache the validated global application configuration settings.

    Why:
        Centralises configuration resolution across YAML defaults, DVC pipeline parameters, conformal calibration artefacts, and runtime environment variables into a single type-safe, immutable Pydantic hierarchy, avoiding repeated disk I/O via LRU caching.

    How:
        1. Resolves workspace root and loads base defaults from config/default_config.yaml.
        2. Deep-merges DVC hyperparameter overrides from params.yaml.
        3. Overlays calibrated conformal thresholds from config/conformal_config.yaml.
        4. Applies optional custom file overrides (config_path or TAXON_CONFIG_PATH) and TAXON_ env variables.
        5. Validates and constructs the Settings Pydantic root model.

    Args:
        config_path: Optional filesystem path to YAML configuration overrides.

    Returns:
        The validated application configuration instance.

    Raises:
        FileNotFoundError: If the base configuration file does not exist.
        pydantic.ValidationError: If the loaded configuration fails schema constraints.
    """
    repo_root = _resolve_workspace_root()
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
    """Clear the cached settings instance to force re-reading configuration files and environment.

    Why:
        Enables clean state reset in automated unit test fixtures and runtime reloads without requiring process termination.

    How:
        Calls cache_clear() on the get_settings LRU cache wrapper.
    """
    get_settings.cache_clear()
