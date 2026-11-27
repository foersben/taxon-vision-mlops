# SPDX-FileCopyrightText: 2026 Benjamin Förster
# SPDX-License-Identifier: MIT
"""MLflow Model Registry promotion gates and Champion/Challenger lifecycle governance.

Summary:
    Implements the Four-Gate Model Promotion Architecture governing candidate progression across Model Registry aliases (`@Production`, `@Challenger`, `@Archived`) using predictive performance, INT8 quantisation fidelity, conformal coverage guarantees, latency service-level objectives, and multivariate calibration exchangeability.

Why (Architectural Rationale & Trade-offs):
    Promoting candidates immediately after Stage 4 training using isolated validation metrics introduces quantisation collapse, unverified holdout conformal miscalibration, latency regressions, and non-exchangeable shift hazards. Evaluating four sequential gates during Stage 7 separates safe promotions from incumbent degradation or stale calibration distributions.

How (Algorithmic Mechanism & Invariants):
    1. Gate 1 validates candidate Macro PR-AUC against incumbent aliases and enforces INT8 quantisation drop within `quantization_drop_tol`.
    2. Gate 2 verifies empirical holdout coverage meets `nominal_coverage - coverage_tolerance`.
    3. Gate 3 verifies quantised CPU p95 execution latency satisfies `latency_slo_ms`.
    4. Gate 4 bootstraps initial `@Production` on an empty registry, stages `@Challenger` if live telemetry volume is insufficient or MMD squared p-value indicates drift, or promotes to `@Production` whilst demoting prior `@Production` to `@Challenger`.

Standard Fields:
    References: Vovk et al. (2005) 'Algorithmic Learning in a Random World'; Gretton et al. (2012) 'A Kernel Two-Sample Test'.
"""

from __future__ import annotations

import logging
import subprocess
from typing import Any

import mlflow
import mlflow.tracking

from taxon_vision.models.training.dataset import get_dvc_dataset_hash
from taxon_vision.models.training.types import PromotionAlias
from taxon_vision.registry.types import (
    ModelPromotionTarget,
    PromotionCriteria,
    PromotionVerdict,
    safe_mlflow_call,
)

logger = logging.getLogger(__name__)


def _get_git_commit_hash() -> str | None:
    """Retrieve the active Git HEAD commit SHA-1 digest, or None outside a repository."""
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


def _get_alias_version_safe(client: Any, model_name: str, alias: str) -> Any | None:
    """Look up registered model version by alias safely, returning None when unset or unreachable."""
    return safe_mlflow_call(
        f"get_model_version_by_alias({alias})",
        lambda: client.get_model_version_by_alias(model_name, alias),
    )


def _fetch_run_safely(client: Any, run_id: str) -> Any | None:
    """Retrieve an MLflow Run object safely, returning None on network or permission failure."""
    return safe_mlflow_call(f"get_run({run_id})", lambda: client.get_run(run_id))


def _get_incumbent_score(client: Any, model_name: str) -> float | None:
    """Query the highest `val_promotion_score` recorded across `@Production` and `@Challenger` aliases."""
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


def evaluate_promotion_gates(
    criteria: PromotionCriteria,
    incumbent_pr_auc: float | None = None,
) -> PromotionVerdict:
    """Evaluate candidate model performance sequentially across the Four-Gate Model Promotion Architecture.

    Why:
        Enforces physical, mathematical, and operational invariants prior to Model Registry promotion so only accurate, INT8-faithful, conformally calibrated, low-latency, and distribution-exchangeable models receive `@Production`.

    How:
        Sequentially checks Gate 1 (incumbent Macro PR-AUC and INT8 drop), Gate 2 (holdout empirical coverage margin), Gate 3 (ONNX p95 latency SLO), and Gate 4 (live telemetry volume and RBF MMD squared exchangeability).

    Args:
        criteria: Strongly typed `PromotionCriteria` holding candidate evaluation measurements.
        incumbent_pr_auc: Optional benchmark Macro PR-AUC from active `@Production`/`@Challenger` versions.

    Returns:
        `PromotionVerdict` containing target alias, decision gate tag, passed gates, and audit tags.
    """
    passed: list[str] = []

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

    if criteria.p95_latency_ms > criteria.latency_slo_ms:
        return PromotionVerdict(
            alias=PromotionAlias.ARCHIVED,
            gate="latency_slo_violation",
            passed_gates=tuple(passed),
            failed_gate="gate_3_latency_slo",
            metadata_tags={"gate": "latency_slo_violation", "failure_reason": "p95_latency_exceeded_slo"},
        )
    passed.append("gate_3_latency_slo")

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


def _apply_promotion_aliases(target: ModelPromotionTarget, alias: PromotionAlias) -> None:
    """Assign MLflow Model Registry aliases and demote prior `@Production` versions to `@Challenger`."""
    if alias == PromotionAlias.PRODUCTION:
        prev_prod = _get_alias_version_safe(target.client, target.model_name, PromotionAlias.PRODUCTION)
        if prev_prod and str(prev_prod.version) != str(target.version):
            safe_mlflow_call(
                "demote_previous_production",
                lambda: target.client.set_registered_model_alias(
                    target.model_name,
                    PromotionAlias.CHALLENGER,
                    str(prev_prod.version),
                ),
            )
    safe_mlflow_call(
        f"set_{alias.value.lower()}_alias",
        lambda: target.client.set_registered_model_alias(
            target.model_name,
            alias,
            str(target.version),
        ),
    )


def promote_candidate_model(
    target: ModelPromotionTarget,
    criteria: PromotionCriteria,
) -> PromotionVerdict:
    """Evaluate Four-Gate promotion criteria, persist provenance tags, and rotate MLflow aliases.

    Why:
        Automates traceable champion/challenger promotion while stamping cryptographic Git SHA, DVC hash, and conformal empirical guarantees directly onto registered model versions.

    How:
        Queries incumbent PR-AUC scores, invokes `evaluate_promotion_gates`, attaches provenance tags via `MlflowClient`, rotates aliases via `_apply_promotion_aliases`, and logs verdict telemetry to the active MLflow run.

    Args:
        target: `ModelPromotionTarget` holding registry coordinates and client handle.
        criteria: `PromotionCriteria` holding Stage 7 evaluation measurements and thresholds.

    Returns:
        `PromotionVerdict` describing the assigned alias and gate rationale.
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
        if git_sha := _get_git_commit_hash():
            tags["git_sha"] = git_sha
        if dvc_hash := get_dvc_dataset_hash():
            tags["dvc_dataset_hash"] = dvc_hash
        if criteria.exchangeability_p_value is not None:
            tags["exchangeability_mmd2_p_value"] = f"{criteria.exchangeability_p_value:.4f}"

        for k, v in tags.items():
            target.client.set_model_version_tag(target.model_name, target.version, k, v)

    safe_mlflow_call("apply_model_version_provenance_tags", _apply_tags)
    _apply_promotion_aliases(target, verdict.alias)

    safe_mlflow_call(
        "log_active_run_promotion_tags",
        lambda: mlflow.set_tags(
            {
                "registry.status": f"promoted_{verdict.alias.value.lower()}",
                "promotion.gate": verdict.gate,
                "promotion.verdict": verdict.alias.value,
            }
        ),
    )
    run_metrics = {"val_promotion_score": criteria.candidate_pr_auc}
    if criteria.exchangeability_p_value is not None:
        run_metrics["exchangeability_mmd2_p_value"] = criteria.exchangeability_p_value
    safe_mlflow_call("log_active_run_promotion_metrics", lambda: mlflow.log_metrics(run_metrics))

    return verdict
