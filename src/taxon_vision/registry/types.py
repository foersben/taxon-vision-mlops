# SPDX-FileCopyrightText: 2026 Benjamin Förster
# SPDX-License-Identifier: MIT
"""Shared data containers, model schemas, and fault-tolerant helpers for MLflow tracking and promotion.

Summary:
    Centralises immutable dataclasses, NamedTuples, TensorSpec signature builders, and safe MLflow invocation wrappers consumed across run context lifecycle management, dataset lineage logging, and Model Registry promotion gates.

Why (Architectural Rationale & Trade-offs):
    Centralising shared schema primitives and typed request containers prevents cross-module duplication between tracking and promotion, eliminates tuple blindness, and enforces Rule 08 parameter arity bounds.

How (Algorithmic Mechanism & Invariants):
    Provides frozen DTOs (`ParentRunContext`, `LoggedDatasets`, `PyTorchModelLogSpec`, `ONNXModelLogSpec`, `PromotionCriteria`, `ModelPromotionTarget`, `PromotionVerdict`) and NamedTuple `ModelDimensions`, inspecting `nn.Linear` modules in O(L) over network depth.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, NamedTuple

import numpy as np
from mlflow.models import ModelSignature
from mlflow.types.schema import Schema, TensorSpec
from torch import nn

from taxon_vision.config.loader import get_settings
from taxon_vision.models.training.types import PromotionAlias

logger = logging.getLogger(__name__)


class ModelDimensions(NamedTuple):
    """Resolved input embedding dimension and output class count for a classifier module."""

    input_dim: int
    num_classes: int


@dataclass(frozen=True)
class ParentRunContext:
    """Immutable context bundle extracted from `reports/train_metrics.json` for cross-stage MLflow resumption."""

    run_id: str
    model_id: str | None
    extractor: str
    status: str


@dataclass(frozen=True)
class LoggedDatasets:
    """Container of dataset lineage objects registered with MLflow during training."""

    train: Any | None
    val: Any | None
    manifest: Any | None = None
    cal: Any | None = None
    test: Any | None = None


@dataclass(frozen=True)
class PyTorchModelLogSpec:
    """Typed specification encapsulating PyTorch model registration inputs for MLflow 3."""

    model: nn.Module
    model_name: str = "taxon_vision_classifier"
    input_dim: int | None = None
    num_classes: int | None = None
    signature: ModelSignature | None = None
    input_example: np.ndarray | None = None
    registered_model_name: str | None = None
    pip_requirements: list[str] | None = None
    params: dict[str, Any] | None = None
    model_type: str = "classifier"


@dataclass(frozen=True)
class ONNXModelLogSpec:
    """Typed specification encapsulating ONNX Runtime artefact registration inputs for MLflow."""

    onnx_model: Any
    artifact_path: str = "model"
    num_classes: int | None = None
    image_size: int | None = None
    signature: ModelSignature | None = None
    input_example: np.ndarray | None = None
    registered_model_name: str | None = None
    extra_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PromotionCriteria:
    """Strongly typed evaluation gates and thresholds governing candidate model promotion."""

    candidate_pr_auc: float
    fp32_to_int8_drop: float
    empirical_coverage: float
    nominal_coverage: float
    p95_latency_ms: float
    exchangeability_p_value: float | None = None
    live_sample_count: int = 0
    quantization_drop_tol: float = 0.03
    coverage_tolerance: float = 0.05
    latency_slo_ms: float = 350.0
    min_telemetry_samples: int = 50
    drift_alpha: float = 0.01


@dataclass(frozen=True)
class ModelPromotionTarget:
    """Target Model Registry coordinates and provenance metadata for candidate promotion."""

    model_name: str
    version: str
    client: Any
    extractor: str | None = None
    run_id: str | None = None


@dataclass(frozen=True)
class PromotionVerdict:
    """Strongly typed decision output returned by the Four-Gate promotion evaluator."""

    alias: PromotionAlias
    gate: str
    passed_gates: tuple[str, ...]
    failed_gate: str | None
    metadata_tags: dict[str, str]


def safe_mlflow_call[T](action_name: str, fn: Callable[[], T], default: T | None = None) -> T | None:
    """Execute a non-critical MLflow API call with warning diagnostics on transient failures.

    Args:
        action_name: Short descriptive identifier for warning logs.
        fn: Zero-argument callable encapsulating the MLflow API request.
        default: Optional value returned when an exception is intercepted.

    Returns:
        Result returned by `fn()`, or `default` on failure.
    """
    try:
        return fn()
    except Exception as err:
        logger.warning("MLflow non-critical operation '%s' failed: %s", action_name, err)
        return default


def resolve_model_dims(
    model: nn.Module,
    input_dim: int | None,
    num_classes: int | None,
) -> ModelDimensions:
    """Infer embedding feature dimensionality and class count from linear probe layers.

    Args:
        model: PyTorch classification module to inspect.
        input_dim: Optional explicit embedding dimensionality override.
        num_classes: Optional explicit output class count override.

    Returns:
        `ModelDimensions` containing resolved `(input_dim, num_classes)`.

    Raises:
        ValueError: If `nn.Linear` layers are absent and explicit dimensions were not provided.
    """
    linear_layers = [m for m in model.modules() if isinstance(m, nn.Linear)]
    resolved_dim = input_dim if input_dim is not None else (linear_layers[0].in_features if linear_layers else None)
    resolved_classes = (
        num_classes if num_classes is not None else (linear_layers[-1].out_features if linear_layers else None)
    )
    if resolved_dim is None or resolved_classes is None:
        raise ValueError("Cannot infer input_dim/num_classes: model has no nn.Linear layers and dims were omitted.")
    return ModelDimensions(input_dim=resolved_dim, num_classes=resolved_classes)


def build_embedding_signature(input_dim: int, num_classes: int) -> ModelSignature:
    """Construct explicit MLflow ModelSignature mapping backbone embeddings to taxonomic logits.

    Args:
        input_dim: Feature embedding dimensionality.
        num_classes: Number of target taxonomic classes.

    Returns:
        Configured `ModelSignature` with float32 `TensorSpec` schemas.
    """
    input_schema = Schema([TensorSpec(np.dtype(np.float32), (-1, input_dim), name="backbone_embeddings")])
    output_schema = Schema([TensorSpec(np.dtype(np.float32), (-1, num_classes), name="taxonomic_logits")])
    return ModelSignature(inputs=input_schema, outputs=output_schema)


def build_onnx_signature(num_classes: int, image_size: int | None = None) -> ModelSignature:
    """Construct explicit MLflow ModelSignature mapping NCHW float32 images to dual ONNX outputs.

    Args:
        num_classes: Number of target taxonomic classes.
        image_size: Spatial image input resolution in pixels (defaults to `settings.model.image_size`).

    Returns:
        Configured `ModelSignature` matching `input`, `logits`, and `embeddings` ONNX tensors.
    """
    resolved_size = image_size if image_size is not None else get_settings().model.image_size
    input_schema = Schema([TensorSpec(np.dtype(np.float32), (-1, 3, resolved_size, resolved_size), name="input")])
    output_schema = Schema(
        [
            TensorSpec(np.dtype(np.float32), (-1, num_classes), name="logits"),
            TensorSpec(np.dtype(np.float32), (-1, -1), name="embeddings"),
        ]
    )
    return ModelSignature(inputs=input_schema, outputs=output_schema)
