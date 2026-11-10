# SPDX-FileCopyrightText: 2026 Benjamin Förster
# SPDX-License-Identifier: MIT
"""MLflow Model Registry promotion gates and champion/challenger lifecycle management.

Summary:
    Implements the Four-Gate Model Promotion Architecture governing candidate progression across Model Registry aliases (@Production, @Challenger, @Archived) based on predictive performance, INT8 quantization fidelity, statistical conformal coverage guarantees, latency service-level objectives, and multivariate calibration exchangeability.

Why (Architectural Rationale & Trade-offs):
    Promoting models immediately upon training completion using isolated validation metrics introduces catastrophic operational risks:
    1. A model with high FP32 validation PR-AUC may collapse in accuracy following INT8 dynamic post-training quantization.
    2. Split conformal prediction sets calibrated on holdout sets may fail to achieve the required nominal coverage (1 - alpha) on unbiased test observations.
    3. The calibration distribution may not be statistically exchangeable with recent live production inference traffic, invalidating mathematical coverage guarantees in real-world deployment.
    To prevent these failure modes, promotion decisions are deferred to holdout evaluation in Stage 7, evaluating candidates across four rigorous, sequential gates before granting deployment aliases.

How (Algorithmic Mechanism & Invariants):
    1. Gate 1 (Predictive Viability): Validates that candidate Macro PR-AUC matches or exceeds the incumbent model baseline, and that INT8 quantization drop remains within the configured tolerance threshold.
    2. Gate 2 (Conformal Safety): Verifies that empirical holdout coverage meets nominal coverage (1 - alpha) minus a finite-sample binomial sampling margin.
    3. Gate 3 (Latency Budget): Verifies that quantized 95th-percentile inference latency satisfies physical CPU execution ceilings.
    4. Gate 4 (Exchangeability & Staging): Evaluates multivariate distribution exchangeability between calibration embeddings and recent production telemetry via an unbiased two-sample RBF MMD squared permutation test. If traffic is insufficient or drift is detected, the candidate is staged as Challenger rather than promoted to Production. If all four gates pass, the candidate is promoted to @Production and the previous incumbent is demoted to @Challenger.

Standard Fields:
    References: Vovk et al. (2005) 'Algorithmic Learning in a Random World'; Gretton et al. (2012) 'A Kernel Two-Sample Test'.
"""

from __future__ import annotations

import copy
import logging
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
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


@dataclass(frozen=True)
class PromotionCriteria:
    """Strongly typed input criteria for evaluating model promotion gates.

    Attributes:
        candidate_pr_auc: Candidate model Macro PR-AUC score.
        fp32_to_int8_drop: Observed accuracy drop between FP32 baseline and INT8 quantized model.
        empirical_coverage: Holdout test split empirical conformal coverage.
        nominal_coverage: Target marginal conformal coverage guarantee (1 - alpha).
        p95_latency_ms: Observed 95th-percentile inference latency in milliseconds.
        exchangeability_p_value: Finite-sample permutation test p-value from two-sample RBF MMD squared evaluation.
        live_sample_count: Number of recent live production inference embeddings evaluated.
        quantization_drop_tol: Maximum tolerable accuracy drop between FP32 and INT8 (default: 0.03).
        coverage_tolerance: Margin allowed below nominal coverage for finite-sample variation (default: 0.05).
        latency_slo_ms: Upper bound latency budget in milliseconds (default: 120.0).
        min_telemetry_samples: Minimum live samples required to evaluate distribution exchangeability (default: 50).
        drift_alpha: Significance ceiling for rejecting calibration exchangeability (default: 0.01).
    """

    candidate_pr_auc: float
    fp32_to_int8_drop: float
    empirical_coverage: float
    nominal_coverage: float
    p95_latency_ms: float
    exchangeability_p_value: float | None = None
    live_sample_count: int = 0
    quantization_drop_tol: float = 0.03
    coverage_tolerance: float = 0.05
    latency_slo_ms: float = 120.0
    min_telemetry_samples: int = 50
    drift_alpha: float = 0.01


@dataclass(frozen=True)
class ModelPromotionTarget:
    """Target registry coordinates and provenance metadata for candidate promotion.

    Attributes:
        model_name: Registered model identifier in the MLflow Model Registry.
        version: Registered version string of the candidate model.
        client: Active MlflowClient instance.
        extractor: Vision foundation backbone name, if known.
        run_id: Associated MLflow experiment run ID, if known.
    """

    model_name: str
    version: str
    client: Any
    extractor: str | None = None
    run_id: str | None = None


@dataclass(frozen=True)
class PromotionVerdict:
    """Strongly typed decision verdict resulting from promotion gate evaluation.

    Attributes:
        alias: Target promotion alias (Production, Challenger, or Archived).
        gate: Distinct gate identifier describing the decision rationale.
        passed_gates: Tuple of gate identifiers that passed verification.
        failed_gate: Specific gate identifier that failed, if any.
        metadata_tags: Dictionary of audit metadata tags for registry persistence.
    """

    alias: PromotionAlias
    gate: str
    passed_gates: tuple[str, ...]
    failed_gate: str | None
    metadata_tags: dict[str, str]


def _get_git_commit_hash() -> str | None:
    """Retrieve current Git commit SHA-1 hash if executing in a Git repository.

    Returns:
        Current Git commit SHA-1 hash if executing in a Git repository, None otherwise.
    """
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=2.0,
        )
        return res.stdout.strip() or None
    except Exception:
        return None


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


def evaluate_promotion_gates(
    criteria: PromotionCriteria,
    incumbent_pr_auc: float | None = None,
) -> PromotionVerdict:
    """Evaluate candidate model performance sequentially across the Four-Gate Model Promotion Architecture.

    Why:
        Enforces physical, mathematical, and operational constraints prior to model registration, ensuring that only performant, calibrated, low-latency, and distribution-exchangeable models achieve production status.

    How:
        1. Evaluates Gate 1: Candidate Macro PR-AUC must match or exceed the incumbent score, and the FP32-to-INT8 accuracy drop must not exceed quantization_drop_tol.
        2. Evaluates Gate 2: Empirical test coverage must meet nominal coverage (1 - alpha) minus coverage_tolerance.
        3. Evaluates Gate 3: Quantized p95 execution latency must not exceed latency_slo_ms.
        4. Evaluates Gate 4: If no incumbent exists, bootstrap initial Production. Otherwise, verify sufficient live telemetry volume and that the MMD permutation p-value meets or exceeds drift_alpha.

    Args:
        criteria: Strongly typed PromotionCriteria containing candidate evaluation metrics.
        incumbent_pr_auc: Optional benchmark Macro PR-AUC of the active production model.

    Returns:
        PromotionVerdict containing target alias, gate string, passed gates, and metadata tags.
    """
    passed: list[str] = []

    # Gate 1: Predictive Viability & INT8 Quantization Fidelity
    if incumbent_pr_auc is not None and criteria.candidate_pr_auc < incumbent_pr_auc:
        return PromotionVerdict(
            alias=PromotionAlias.ARCHIVED,
            gate="below_incumbent",
            passed_gates=tuple(passed),
            failed_gate="gate_1_predictive_viability",
            metadata_tags={"gate": "below_incumbent", "failure_reason": "candidate_pr_auc_below_incumbent"},
        )
    if criteria.fp32_to_int8_drop > criteria.quantization_drop_tol:
        return PromotionVerdict(
            alias=PromotionAlias.ARCHIVED,
            gate="quantization_drop_exceeded",
            passed_gates=tuple(passed),
            failed_gate="gate_1_quantization_fidelity",
            metadata_tags={"gate": "quantization_drop_exceeded", "failure_reason": "fp32_to_int8_drop_exceeded"},
        )
    passed.append("gate_1_predictive_viability")

    # Gate 2: Conformal Coverage Guarantee
    min_coverage = criteria.nominal_coverage - criteria.coverage_tolerance
    if criteria.empirical_coverage < min_coverage:
        return PromotionVerdict(
            alias=PromotionAlias.ARCHIVED,
            gate="conformal_coverage_violation",
            passed_gates=tuple(passed),
            failed_gate="gate_2_conformal_coverage",
            metadata_tags={
                "gate": "conformal_coverage_violation",
                "failure_reason": "empirical_coverage_below_tolerance",
            },
        )
    passed.append("gate_2_conformal_coverage")

    # Gate 3: Physical Latency Budget
    if criteria.p95_latency_ms > criteria.latency_slo_ms:
        return PromotionVerdict(
            alias=PromotionAlias.ARCHIVED,
            gate="latency_slo_violation",
            passed_gates=tuple(passed),
            failed_gate="gate_3_latency_slo",
            metadata_tags={"gate": "latency_slo_violation", "failure_reason": "p95_latency_exceeded_slo"},
        )
    passed.append("gate_3_latency_slo")

    # Gate 4: Exchangeability & Bootstrap
    if incumbent_pr_auc is None:
        passed.append("gate_4_bootstrap_production")
        return PromotionVerdict(
            alias=PromotionAlias.PRODUCTION,
            gate="initial_production",
            passed_gates=tuple(passed),
            failed_gate=None,
            metadata_tags={"gate": "initial_production", "status": "bootstrapped_production"},
        )

    if criteria.live_sample_count < criteria.min_telemetry_samples:
        return PromotionVerdict(
            alias=PromotionAlias.CHALLENGER,
            gate="insufficient_traffic",
            passed_gates=tuple(passed),
            failed_gate="gate_4_traffic_volume",
            metadata_tags={"gate": "insufficient_traffic", "status": "challenger_insufficient_live_samples"},
        )

    if criteria.exchangeability_p_value is not None and criteria.exchangeability_p_value < criteria.drift_alpha:
        return PromotionVerdict(
            alias=PromotionAlias.CHALLENGER,
            gate="calibration_not_exchangeable",
            passed_gates=tuple(passed),
            failed_gate="gate_4_exchangeability_drift",
            metadata_tags={"gate": "calibration_not_exchangeable", "status": "challenger_drift_detected"},
        )

    passed.append("gate_4_exchangeability")
    return PromotionVerdict(
        alias=PromotionAlias.PRODUCTION,
        gate="promoted_all_4_gates",
        passed_gates=tuple(passed),
        failed_gate=None,
        metadata_tags={"gate": "promoted_all_4_gates", "status": "promoted_production"},
    )


def _apply_promotion_aliases(
    target: ModelPromotionTarget,
    alias: PromotionAlias,
) -> None:
    """Update model registry aliases, ensuring graceful demotion of incumbents when necessary.

    Args:
        target: Target registry information.
        alias: Desired promotion alias to assign.
    """
    if alias == PromotionAlias.PRODUCTION:
        prev_prod = _get_alias_version_safe(target.client, target.model_name, PromotionAlias.PRODUCTION)
        if prev_prod and str(prev_prod.version) != str(target.version):
            _safe_mlflow_call(
                "demote_previous_production",
                lambda: target.client.set_registered_model_alias(
                    target.model_name,
                    PromotionAlias.CHALLENGER,
                    str(prev_prod.version),
                ),
            )
        _safe_mlflow_call(
            "set_production_alias",
            lambda: target.client.set_registered_model_alias(
                target.model_name,
                PromotionAlias.PRODUCTION,
                str(target.version),
            ),
        )
    elif alias == PromotionAlias.CHALLENGER:
        _safe_mlflow_call(
            "set_challenger_alias",
            lambda: target.client.set_registered_model_alias(
                target.model_name,
                PromotionAlias.CHALLENGER,
                str(target.version),
            ),
        )
    else:
        _safe_mlflow_call(
            "set_archived_alias",
            lambda: target.client.set_registered_model_alias(
                target.model_name,
                PromotionAlias.ARCHIVED,
                str(target.version),
            ),
        )


def promote_candidate_model(
    target: ModelPromotionTarget,
    criteria: PromotionCriteria,
) -> PromotionVerdict:
    """Evaluate candidate model gates, apply registry metadata, and update MLflow model aliases.

    Why:
        Automates model governance by providing atomic, traceable registry promotion. Attaching cryptographic provenance tags directly to registered model versions guarantees auditability in production environments.

    How:
        1. Retrieves incumbent PR-AUC across existing Production and Challenger aliases.
        2. Evaluates the candidate against evaluate_promotion_gates.
        3. Persists provenance tags (git SHA, DVC dataset hash, backbone extractor, and performance metrics) to the model version.
        4. Reallocates aliases based on the verdict (demoting previous Production models to Challenger when replaced).
        5. Logs promotion verdict metrics and tags to the active MLflow run if present.

    Args:
        target: ModelPromotionTarget specifying model name, version, client, and extractor.
        criteria: PromotionCriteria containing measured evaluation and latency metrics.

    Returns:
        PromotionVerdict detailing the assigned alias, gate rationale, and passed gate list.
    """
    incumbent_score = _get_incumbent_score(target.client, target.model_name)
    verdict = evaluate_promotion_gates(criteria, incumbent_score)

    def _apply_tags() -> None:
        desc = (
            f"TaxonVision model {target.model_name} (version {target.version}). "
            f"Promotion verdict: {verdict.alias.value} (gate: {verdict.gate})."
        )
        target.client.update_model_version(name=target.model_name, version=target.version, description=desc)

        tags = {
            "val_pr_auc": f"{criteria.candidate_pr_auc:.4f}",
            "empirical_coverage": f"{criteria.empirical_coverage:.4f}",
            "p95_latency_ms": f"{criteria.p95_latency_ms:.2f}",
            "gate": verdict.gate,
            "promotion_verdict": verdict.alias.value,
        }
        if target.extractor:
            tags["extractor"] = target.extractor

        git_sha = _get_git_commit_hash()
        if git_sha:
            tags["git_sha"] = git_sha

        dvc_hash = get_dvc_dataset_hash()
        if dvc_hash:
            tags["dvc_dataset_hash"] = dvc_hash

        if criteria.exchangeability_p_value is not None:
            tags["exchangeability_mmd2_p_value"] = f"{criteria.exchangeability_p_value:.4f}"

        for k, v in tags.items():
            target.client.set_model_version_tag(target.model_name, target.version, k, v)

    _safe_mlflow_call("apply_model_version_provenance_tags", _apply_tags)
    _apply_promotion_aliases(target, verdict.alias)

    _safe_mlflow_call(
        "log_active_run_promotion_tags",
        lambda: mlflow.set_tag("registry.status", f"promoted_{verdict.alias.value.lower()}"),
    )
    _safe_mlflow_call(
        "log_active_run_gate_tag",
        lambda: mlflow.set_tag("promotion.gate", verdict.gate),
    )
    _safe_mlflow_call(
        "log_active_run_verdict_tag",
        lambda: mlflow.set_tag("promotion.verdict", verdict.alias.value),
    )
    _safe_mlflow_call(
        "log_active_run_promotion_score",
        lambda: mlflow.log_metric("val_promotion_score", criteria.candidate_pr_auc),
    )
    p_val = criteria.exchangeability_p_value
    if p_val is not None:
        _safe_mlflow_call(
            "log_active_run_exchangeability_metric",
            lambda: mlflow.log_metric("exchangeability_mmd2_p_value", p_val),
        )

    return verdict


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
