# SPDX-FileCopyrightText: 2026 Benjamin Förster
# SPDX-License-Identifier: MIT
"""Configuration schema definitions and strict Pydantic models."""

from __future__ import annotations

from typing import Literal

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
        taxa_catalog_path (str): Path to taxa catalog YAML definition.
        eval_resize_ratio (float): Image resize ratio prior to center cropping.
        inference_backend (str): Serving backend: 'onnx', 'torch' or 'auto'.
        quantization_mode (str): INT8 mode used by exporter: 'static', 'dynamic' or 'auto'.
        calibration_dir (str): Directory of representative images used for static PTQ calibration.
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
        weight_decay (float): Initial weight decay coefficient for regularization.
        cb_beta (float): Class-Balanced Loss effective sample weighting parameter beta.
        dropout (float): Dropout probability for classifier head regularization.
        noise_std (float): Standard deviation of Gaussian noise injected into embeddings.
        early_stopping_patience (int): Number of epochs without improvement before early stopping.
        reduce_lr_patience (int): Number of epochs without improvement before reducing learning rate.
        warmup_epochs (int): Number of initial warmup epochs.
        val_split (float): Validation split fraction.
        cal_split (float): Calibration split fraction for conformal prediction.
        test_split (float): Final test evaluation split fraction.
        random_seed (int): Deterministic PRNG seed for reproducible splits.
        extractor (str): Default backbone identifier for training.
        tune (bool): Whether to run Optuna Bayesian hyperparameter search.
    """

    epochs: int = Field(default=30, gt=0)
    batch_size: int = Field(default=32, gt=0)
    learning_rate: float = Field(default=0.001, gt=0.0)
    weight_decay: float = Field(default=0.0001, ge=0.0)
    cb_beta: float = Field(default=0.9999, gt=0.0, lt=1.0)
    dropout: float = Field(default=0.2, ge=0.0, lt=1.0)
    noise_std: float = Field(default=0.01, ge=0.0)
    early_stopping_patience: int = Field(default=5, gt=0)
    reduce_lr_patience: int = Field(default=2, gt=0)
    warmup_epochs: int = Field(default=3, ge=0)
    val_split: float = Field(default=0.10, ge=0.0, lt=1.0)
    cal_split: float = Field(default=0.10, ge=0.0, lt=1.0)
    test_split: float = Field(default=0.10, ge=0.0, lt=1.0)
    random_seed: int = Field(default=42)
    extractor: str = "dinov3"
    tune: bool = Field(default=False)


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

    tracking_uri: str = "https://dagshub.com/foersben/taxon-vision-mlops.mlflow"
    experiment_name: str = "taxon-vision-head-tuning"


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
        min_confidence (float): Minimum acceptable top-1 class confidence to bypass human review.
        min_confidence_threshold (float): Alias for min_confidence.
        ood_tpr (float): Target true positive rate on in-distribution data for OOD energy thresholding.
    """

    alpha: float = Field(default=0.05, gt=0.0, lt=1.0)
    q_hat: float = Field(default=0.85, gt=0.0, lt=1.0)
    k_max: int = Field(default=3, gt=0)
    quantile_threshold: float = Field(default=0.85, gt=0.0, lt=1.0)
    nominal_coverage: float = Field(default=0.95, gt=0.0, lt=1.0)
    method: str = "split-conformal"
    empty_set_action: str = "route_to_human_triage"
    min_confidence: float = Field(default=0.60, ge=0.0, le=1.0)
    min_confidence_threshold: float = Field(default=0.60, ge=0.0, le=1.0)
    ood_tpr: float = Field(default=0.95, gt=0.0, le=1.0)


class OODConfig(BaseModel):
    """Out-of-Distribution (OOD) energy and entropy detection thresholds schema.

    Attributes:
        energy_threshold (float): Free energy score threshold below which observations are flagged as OOD.
        entropy_threshold (float): Shannon predictive entropy threshold for uncertainty triage.
    """

    energy_threshold: float
    entropy_threshold: float


class StreamingConfig(BaseModel):
    """Configuration schema for streaming observation data ingestion and HTTP pooling.

    Why:
        Controls connection reuse, concurrency bounds, and retry backoff for network streaming pipelines.

    How:
        Parsed from declarative YAML and mapped to typed Pydantic fields, enforcing valid retry and timeout limits.

    Attributes:
        http_max_retries (int): Maximum retry attempts for transient network request failures.
        http_backoff_factor (float): Multiplicative delay factor for exponential retry backoff.
        http_max_connections (int): Maximum total HTTP socket connections in the connection pool.
        http_keepalive_connections (int): Maximum idle HTTP keep-alive connections retained in pool.
        http_timeout_seconds (float): Timeout duration in seconds before HTTP requests abort.
        min_extraction_success_ratio (float): Minimum fractional threshold of successful downloads required.
    """

    http_max_retries: int = Field(default=5, ge=0)
    http_backoff_factor: float = Field(default=0.5, ge=0.0)
    http_max_connections: int = Field(default=16, gt=0)
    http_keepalive_connections: int = Field(default=8, ge=0)
    http_timeout_seconds: float = Field(default=15.0, gt=0.0)
    min_extraction_success_ratio: float = Field(default=0.95, ge=0.0, le=1.0)


class OptunaConfig(BaseModel):
    """Configuration schema for Optuna hyperparameter optimization studies.

    Why:
        Centralizes the Optuna persistence backend storage URI and study naming conventions.

    How:
        Populated from default YAML configuration and overridable via the TAXON_OPTUNA_STORAGE_URI environment variable.

    Attributes:
        optuna_storage_uri (str): SQLAlchemy database URI or RDB URL for persisting Optuna study records.
        optuna_study_prefix (str): Prefix prepended to generated Optuna study names.
        n_trials (int): Total number of hyperparameter evaluation trials.
        epochs (int): Maximum training epochs allocated to each trial.
        min_resource (int): Minimum resource allocation for successive halving pruning.
        reduction_factor (int): Reduction factor for successive halving pruning.
    """

    optuna_storage_uri: str = "sqlite:///data/optuna/optuna_studies.db"
    optuna_study_prefix: str = "taxon_vision"
    n_trials: int = Field(default=20, gt=0)
    epochs: int = Field(default=20, gt=0)
    min_resource: int = Field(default=3, gt=0)
    reduction_factor: int = Field(default=2, gt=1)


class EvaluationConfig(BaseModel):
    """Configuration schema for model evaluation and latency SLAs.

    Attributes:
        latency_sla_ms (float): Latency service level agreement threshold in milliseconds (default: 25.0).
    """

    latency_sla_ms: float = Field(default=25.0, gt=0.0)


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
    streaming: StreamingConfig = Field(default_factory=StreamingConfig)
    optuna: OptunaConfig = Field(default_factory=OptunaConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)

    @property
    def data(self) -> StreamingConfig:
        """Alias for streaming configuration to support data subsystem references."""
        return self.streaming
