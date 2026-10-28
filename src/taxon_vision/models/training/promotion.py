# SPDX-FileCopyrightText: 2026 Benjamin Förster
# SPDX-License-Identifier: MIT
"""MLflow Model Registry promotion gates and champion/challenger lifecycle management."""

from __future__ import annotations

import copy
import logging
from collections.abc import Callable
from typing import Any, NamedTuple

import mlflow
import mlflow.pytorch
import mlflow.tracking
import numpy as np
import torch
from mlflow.models import ModelSignature
from mlflow.types.schema import Schema, TensorSpec
from torch import nn

from taxon_vision.models.training.dataset import get_dvc_dataset_hash
from taxon_vision.models.training.types import PromotionAlias

logger = logging.getLogger(__name__)


class ModelDimensions(NamedTuple):
    """Resolved input embedding dimension and output class count for a model."""

    input_dim: int
    num_classes: int


class SignatureAndExample(NamedTuple):
    """MLflow ModelSignature paired with a representative NumPy input example."""

    signature: ModelSignature
    input_example: np.ndarray


def _safe_mlflow_call[T](action_name: str, fn: Callable[[], T], default: T | None = None) -> T | None:
    """Execute a non-critical MLflow action, suppressing transient network or API errors.

    Args:
        action_name: Short human-readable identifier for the operation being performed.
        fn: Zero-argument callable executing the MLflow operation.
        default: Fallback return value if an exception occurs.

    Returns:
        The result of fn(), or default if an exception was raised.
    """
    try:
        return fn()
    except Exception as err:
        logger.warning("MLflow non-critical operation '%s' failed: %s", action_name, err)
        return default


def _resolve_model_dims(
    model: nn.Module,
    input_dim: int | None,
    num_classes: int | None,
) -> ModelDimensions:
    """Resolve embedding dimension and class count from linear layer geometry.

    Args:
        model: PyTorch model module to inspect.
        input_dim: Optional explicit input dimension.
        num_classes: Optional explicit number of output classes.

    Returns:
        ModelDimensions of (input_dim, num_classes).
    """
    resolved_dim = input_dim or 1280
    resolved_classes = num_classes or 10
    linear_layers = [m for m in model.modules() if isinstance(m, nn.Linear)]
    if not linear_layers:
        return ModelDimensions(input_dim=resolved_dim, num_classes=resolved_classes)
    if input_dim is None:
        resolved_dim = linear_layers[0].in_features
    if num_classes is None:
        resolved_classes = linear_layers[-1].out_features
    return ModelDimensions(input_dim=resolved_dim, num_classes=resolved_classes)


def _build_embedding_signature(input_dim: int, num_classes: int) -> ModelSignature:
    """Construct explicit MLflow ModelSignature for embedding vector -> taxonomic logits.

    Args:
        input_dim: Feature embedding length.
        num_classes: Number of target taxonomic classes.

    Returns:
        ModelSignature with backbone_embeddings input and taxonomic_logits output TensorSpecs.
    """
    input_schema = Schema([TensorSpec(np.dtype(np.float32), (-1, input_dim), name="backbone_embeddings")])
    output_schema = Schema([TensorSpec(np.dtype(np.float32), (-1, num_classes), name="taxonomic_logits")])
    return ModelSignature(inputs=input_schema, outputs=output_schema)


def _prepare_model_signature_and_example(
    model: nn.Module,
    val_embeddings: torch.Tensor | None = None,
) -> SignatureAndExample:
    """Infer model signature and construct a dummy or sampled input example.

    Args:
        model: PyTorch model module.
        val_embeddings: Optional real validation embeddings used for signature inference.

    Returns:
        SignatureAndExample of (signature, input_example).
    """
    input_dim, num_classes = _resolve_model_dims(model, None, None)

    if val_embeddings is not None and val_embeddings.shape[0] > 0:
        input_dim = val_embeddings.shape[1]
        sample_size = min(5, val_embeddings.shape[0])
        input_example = val_embeddings[:sample_size].detach().cpu().numpy()
    else:
        input_example = np.zeros((1, input_dim), dtype=np.float32)

    signature = _build_embedding_signature(input_dim, num_classes)
    return SignatureAndExample(signature=signature, input_example=input_example)


def _log_model_artifact(
    model: nn.Module,
    model_name: str,
    signature: ModelSignature,
    input_example: np.ndarray,
) -> Any:
    """Log PyTorch model to MLflow artifact store without registering to registry.

    Args:
        model: PyTorch model module.
        model_name: Target model name.
        signature: Inferred model signature.
        input_example: Real or zero-padded input example array.

    Returns:
        ModelInfo instance on success, or None on failure.
    """
    cpu_model = copy.deepcopy(model).cpu().eval()
    pip_reqs = ["torch", "numpy", "cloudpickle"]
    try:
        return mlflow.pytorch.log_model(
            pytorch_model=cpu_model,
            artifact_path=model_name,
            signature=signature,
            input_example=input_example,
            pip_requirements=pip_reqs,
        )
    except Exception as err:
        logger.warning("Failed to log PyTorch model to MLflow: %s", err)
        return None


def _get_alias_version_safe(client: Any, model_name: str, alias: str) -> Any | None:
    """Query model version by alias, returning None if entity is missing or unreachable.

    Args:
        client: MlflowClient instance.
        model_name: Registered model name.
        alias: Targeted model alias (e.g. 'Production' or 'Challenger').

    Returns:
        ModelVersion instance if found, or None.
    """
    return _safe_mlflow_call(
        f"get_model_version_by_alias({alias})",
        lambda: client.get_model_version_by_alias(model_name, alias),
    )


def _fetch_run_safely(client: Any, run_id: str) -> Any | None:
    """Safely fetch run information from MLflow client.

    Args:
        client: MlflowClient instance.
        run_id: Unique identifier of the MLflow run.

    Returns:
        Run instance if successfully fetched, or None on failure.
    """
    return _safe_mlflow_call(f"get_run({run_id})", lambda: client.get_run(run_id))


def _get_incumbent_score(client: Any, model_name: str) -> float | None:
    """Retrieve the highest promotion score among Production and Challenger aliases.

    Args:
        client: MlflowClient instance.
        model_name: Registered model name.

    Returns:
        Highest incumbent score observed, or None if no aliased versions exist.
    """
    scores: list[float] = []
    for alias in (PromotionAlias.PRODUCTION, PromotionAlias.CHALLENGER):
        mv = _get_alias_version_safe(client, model_name, alias)
        if mv and mv.run_id:
            run = _fetch_run_safely(client, str(mv.run_id))
            if run and run.data and run.data.metrics:
                score = run.data.metrics.get("val_promotion_score")
                if score is not None:
                    scores.append(float(score))
    return max(scores) if scores else None


def _apply_model_version_metadata(
    client: Any,
    model_name: str,
    version: str,
    extractor: str | None,
    pr_auc: float,
) -> None:
    """Attach human-readable description and audit tags to registered model version.

    Args:
        client: MlflowClient instance.
        model_name: Registered model name.
        version: Model version identifier string.
        extractor: Vision foundation backbone name.
        pr_auc: Validation Macro PR-AUC score.
    """

    def _apply() -> None:
        desc = f"TaxonVision classifier head ({extractor or 'backbone'}). Validation Macro PR-AUC: {pr_auc:.4f}."
        client.update_model_version(name=model_name, version=version, description=desc)
        if extractor:
            client.set_model_version_tag(model_name, version, "extractor", extractor)
        client.set_model_version_tag(model_name, version, "val_pr_auc", f"{pr_auc:.4f}")
        dataset_hash = get_dvc_dataset_hash()
        if dataset_hash:
            client.set_model_version_tag(model_name, version, "dvc_dataset_hash", dataset_hash)

    _safe_mlflow_call("set_model_version_metadata", _apply)


def _determine_promotion_alias(
    client: Any,
    model_name: str,
    version: str,
    pr_auc: float,
) -> PromotionAlias:
    """Compare performance against current production alias and tag candidate accordingly.

    Args:
        client: MlflowClient instance.
        model_name: Registered model name.
        version: Model version identifier string.
        pr_auc: Validation Macro PR-AUC score.

    Returns:
        Applied alias enum (PromotionAlias.CHALLENGER, PromotionAlias.ARCHIVED, or PromotionAlias.PRODUCTION).
    """
    prod_version = _get_alias_version_safe(client, model_name, PromotionAlias.PRODUCTION)
    if prod_version is None or not prod_version.run_id:
        logger.info("No production alias found. Tagging initial model as Production.")
        client.set_registered_model_alias(model_name, PromotionAlias.PRODUCTION, version)
        return PromotionAlias.PRODUCTION

    prod_run = _fetch_run_safely(client, str(prod_version.run_id))
    prod_score = 0.0
    if prod_run and prod_run.data and prod_run.data.metrics:
        prod_score = prod_run.data.metrics.get("val_promotion_score", 0.0)

    if pr_auc >= prod_score:
        logger.info(
            "New model meets or outperforms production (%.4f >= %.4f). Tagging as Challenger.",
            pr_auc,
            prod_score,
        )
        client.set_registered_model_alias(model_name, PromotionAlias.CHALLENGER, version)
        return PromotionAlias.CHALLENGER

    logger.info("New model is worse than production (%.4f < %.4f). Tagging as Archived.", pr_auc, prod_score)
    client.set_registered_model_alias(model_name, PromotionAlias.ARCHIVED, version)
    return PromotionAlias.ARCHIVED


def _promote_model_if_better(
    model: nn.Module,
    pr_auc: float,
    val_embeddings: torch.Tensor | None = None,
    extractor: str | None = None,
) -> str | None:
    """Evaluate performance and conditionally promote the model via MLflow Model Registry.

    Why:
        Automated continuous deployment requires programmatic model promotion gates. Pre-gating prevents registering unpromoted models, preserving a clean registry.

    How:
        Logs the validation PR-AUC promotion score and model artifact. Checks incumbent scores across Production and Challenger aliases. If candidate does not strictly beat the incumbent, skips registration and tags the run as skipped_below_incumbent. If it beats the incumbent (or is the initial model), registers it and tags appropriately.

    Args:
        model: The trained PyTorch classification head.
        pr_auc: The final validation Macro PR-AUC score.
        val_embeddings: Optional real validation embeddings used for signature inference.
        extractor: Optional backbone feature extractor identifier.

    Returns:
        The applied promotion alias string ('Challenger', 'Production', 'skipped_below_incumbent'), or None.
    """
    mlflow.log_metric("val_promotion_score", pr_auc)
    model_name = "taxon_vision_classifier"

    signature, input_example = _prepare_model_signature_and_example(model, val_embeddings)
    model_info = _log_model_artifact(model, model_name, signature, input_example)
    if model_info is None:
        return None

    client = mlflow.tracking.MlflowClient()
    incumbent_score = _get_incumbent_score(client, model_name)

    if incumbent_score is not None and pr_auc <= incumbent_score:
        logger.info(
            "Candidate PR-AUC (%.4f) did not strictly beat incumbent (%.4f). Skipping registry registration.",
            pr_auc,
            incumbent_score,
        )
        _safe_mlflow_call(
            "tag_skipped_status",
            lambda: mlflow.set_tag("registry.status", PromotionAlias.SKIPPED_BELOW_INCUMBENT),
        )
        return PromotionAlias.SKIPPED_BELOW_INCUMBENT

    try:
        reg_model = mlflow.register_model(model_uri=model_info.model_uri, name=model_name)
        version = str(reg_model.version)
    except Exception as reg_err:
        logger.warning("Failed to register model in MLflow Model Registry: %s", reg_err)
        return None

    _apply_model_version_metadata(client, model_name, version, extractor, pr_auc)
    promotion_status = _determine_promotion_alias(client, model_name, version, pr_auc)
    _safe_mlflow_call(
        "tag_promoted_status",
        lambda: mlflow.set_tag("registry.status", f"promoted_{promotion_status.lower()}"),
    )
    return str(promotion_status)
