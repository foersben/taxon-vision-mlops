# SPDX-FileCopyrightText: 2026 Benjamin Förster
# SPDX-License-Identifier: MIT
"""MLflow holdout audit logging, PyFunc bundling, and Stage 7/8 model promotion governance.

Summary:
    Resumes the Stage 4 parent MLflow run to record holdout dataset lineage, per-taxon audit tables, diagnostic visualisations, the deployable `TaxonVisionPyFuncModel` bundle, and Four-Gate Model Registry promotion decisions.

Why (Architectural Rationale & Trade-offs):
    Separating MLflow 3.x tracking, exchangeability MMD² checking, and `@Production`/`@Challenger` alias promotion from CLI orchestration (`evaluate.py`) and offline ONNX evaluation (`eval_metrics.py`) isolates registry SDK dependencies and keeps every sibling module below 500 LOC (Rule 08).

How (Algorithmic Mechanism & Invariants):
    Enters `resume_parent_mlflow_run()`, binds `mlflow.data` holdout metrics to `ctx.model_id`, packages `TaxonVisionPyFuncModel` with co-versioned configs, evaluates calibration exchangeability against live PostgreSQL telemetry via `compute_mmd2_permutation`, and invokes `promote_candidate_model`.
"""

from __future__ import annotations

import contextlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlflow
import mlflow.data
import mlflow.models
import mlflow.pyfunc
import numpy as np
import pandas as pd
import torch

from taxon_vision.config import get_settings
from taxon_vision.inference.mlflow_model import TaxonVisionPyFuncModel
from taxon_vision.models.training.types import PromotionAlias
from taxon_vision.models.training.visualizations import (
    HoldoutVisualizationPayload,
    log_holdout_visualizations,
)
from taxon_vision.monitoring.drift import compute_mmd2_permutation
from taxon_vision.pipelines.eval_metrics import (
    HoldoutEvaluationResult,
    ONNXLatencyProfile,
    ParetoResult,
)
from taxon_vision.registry.promotion import promote_candidate_model
from taxon_vision.registry.run_context import resume_parent_mlflow_run
from taxon_vision.registry.types import ModelPromotionTarget, PromotionCriteria
from taxon_vision.service.telemetry_store import get_telemetry_store

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MLflowHoldoutLoggingPayload:
    """Encapsulates all holdout artefacts and metadata for MLflow parent run logging.

    Attributes:
        eval_result: Holdout evaluation metrics and arrays.
        latency_profile: ONNX Runtime latency measurements.
        pareto_profiles: Candidate backbone architectural and performance profiles.
        per_taxon_df: Per-taxon evaluation results for reporting.
        taxa_catalog: Taxonomy metadata for display in reports.
        onnx_path: Path to the exported ONNX model.
        conformal_config_path: Path to the calibration configuration.
        latency_slo_ms: Target latency for deployment.
    """

    eval_result: HoldoutEvaluationResult
    latency_profile: ONNXLatencyProfile
    pareto_profiles: list[ParetoResult]
    per_taxon_df: pd.DataFrame
    taxa_catalog: list[dict[str, Any]]
    onnx_path: Path
    conformal_config_path: Path
    latency_slo_ms: float


def load_reference_calibration_embeddings(cal_path: Path) -> np.ndarray | None:
    """Load calibration embeddings partition from disk if present."""
    if not cal_path.exists():
        return None
    try:
        cal_data = torch.load(cal_path, map_location="cpu", weights_only=True)
        if isinstance(cal_data, dict):
            raw = cal_data.get("embeddings", cal_data.get("cal_embeddings"))
            return raw.numpy() if isinstance(raw, torch.Tensor) else np.asarray(raw)
    except Exception:
        return None
    return None


def resolve_model_version(client: Any, model_name: str, run_id: str, artifact_path: str) -> str | None:
    """Resolve registered model version for the given run, or register candidate artifact."""
    try:
        versions = client.search_model_versions(f"name='{model_name}' and run_id='{run_id}'")
        if versions:
            return str(versions[0].version)
    except Exception:
        pass
    try:
        reg = mlflow.register_model(model_uri=f"runs:/{run_id}/{artifact_path}", name=model_name)
        return str(reg.version)
    except Exception as err:
        logger.debug("Model version registration fallback failed for %s: %s", model_name, err)
        return None


def execute_stage7_promotion(payload: MLflowHoldoutLoggingPayload, run_id: str, extractor: str | None) -> bool:
    """Evaluate candidate against Four-Gate Model Promotion Architecture and update MLflow registry."""
    settings = get_settings()
    prom_cfg = settings.promotion
    mean_pr_auc = (
        float(payload.per_taxon_df["ap"].mean())
        if not payload.per_taxon_df.empty and "ap" in payload.per_taxon_df
        else payload.eval_result.test_accuracy
    )

    ref_embeddings = load_reference_calibration_embeddings(Path("data/embeddings/cal_embeddings.pt"))
    live_sample_count = 0
    exchangeability_p_value: float | None = None
    try:
        store = get_telemetry_store()
        recent = store.recent_embeddings(limit=settings.drift.window_size)
        if recent is not None:
            live_sample_count = len(recent)
            if (
                live_sample_count >= prom_cfg.min_telemetry_samples
                and ref_embeddings is not None
                and len(ref_embeddings) >= 2
            ):
                mmd_res = compute_mmd2_permutation(
                    ref_embeddings,
                    recent,
                    max_samples=settings.drift.mmd_max_samples,
                    n_permutations=settings.drift.mmd_permutations,
                    seed=42,
                )
                exchangeability_p_value = mmd_res.p_value
    except Exception as err:
        logger.debug("Telemetry exchangeability check skipped: %s", err)

    criteria = PromotionCriteria(
        candidate_pr_auc=mean_pr_auc,
        fp32_to_int8_drop=payload.eval_result.fp32_to_int8_drop,
        empirical_coverage=payload.eval_result.empirical_coverage,
        nominal_coverage=payload.eval_result.nominal_coverage,
        p95_latency_ms=payload.latency_profile.p95_ms,
        exchangeability_p_value=exchangeability_p_value,
        live_sample_count=live_sample_count,
        quantization_drop_tol=prom_cfg.quantization_drop_tol,
        coverage_tolerance=prom_cfg.coverage_tolerance,
        latency_slo_ms=prom_cfg.latency_slo_ms,
        min_telemetry_samples=prom_cfg.min_telemetry_samples,
        drift_alpha=prom_cfg.drift_alpha,
    )

    client = mlflow.tracking.MlflowClient()
    promoted = False

    head_ver = resolve_model_version(client, settings.mlflow.model_registry_name, run_id, "taxon_vision_classifier")
    if head_ver:
        target_head = ModelPromotionTarget(
            model_name=settings.mlflow.model_registry_name,
            version=head_ver,
            client=client,
            extractor=extractor,
            run_id=run_id,
        )
        head_verdict = promote_candidate_model(target_head, criteria)
        promoted = promoted or (head_verdict.alias == PromotionAlias.PRODUCTION)

    onnx_ver = resolve_model_version(client, settings.mlflow.onnx_registry_name, run_id, "onnx_model")
    if onnx_ver:
        target_onnx = ModelPromotionTarget(
            model_name=settings.mlflow.onnx_registry_name,
            version=onnx_ver,
            client=client,
            extractor=extractor,
            run_id=run_id,
        )
        onnx_verdict = promote_candidate_model(target_onnx, criteria)
        promoted = promoted or (onnx_verdict.alias == PromotionAlias.PRODUCTION)

    return promoted


def _log_pyfunc_and_promotion(payload: MLflowHoldoutLoggingPayload, ctx: Any) -> None:
    """Log the PyFunc bundle and execute Stage 7 promotion gating inside an active MLflow run."""
    with contextlib.suppress(Exception):
        artifacts: dict[str, str] = {
            "onnx_model": str(payload.onnx_path),
            "conformal_config": str(payload.conformal_config_path),
        }
        catalog_path = Path("config/taxa_catalog.yaml")
        if catalog_path.exists():
            artifacts["taxa_catalog"] = str(catalog_path)

        mlflow.pyfunc.log_model(
            artifact_path="pyfunc_model",
            python_model=TaxonVisionPyFuncModel(),
            artifacts=artifacts,
        )

    promoted = False
    try:
        promoted = execute_stage7_promotion(payload, ctx.run_id, ctx.extractor)
    except Exception:
        logger.exception("Stage 7 promotion gate execution failed")
    status_label = "Production" if promoted else "Rejected"
    with contextlib.suppress(Exception):
        mlflow.log_metric("eval.promoted", 1.0 if promoted else 0.0)
        mlflow.set_tag("registry.status", status_label)
        mlflow.set_tag("promotion.verdict", status_label)


def log_holdout_audit_to_mlflow(payload: MLflowHoldoutLoggingPayload) -> None:
    """Log holdout datasets, metrics, pyfunc model, per-taxon table, and visualisations to MLflow."""
    with resume_parent_mlflow_run() as ctx:
        if ctx is None:
            return

        test_ds = None
        with contextlib.suppress(Exception):
            test_ds = mlflow.data.from_numpy(  # type: ignore[attr-defined]
                features=payload.eval_result.logits,
                targets=payload.eval_result.labels,
                name="test_embeddings",
            )
            mlflow.log_input(test_ds, context="testing")

        with contextlib.suppress(Exception):

            def _predict_fn(x: Any) -> np.ndarray:
                arr = x.to_numpy() if isinstance(x, pd.DataFrame) else np.asarray(x)
                return np.argmax(arr, axis=1)

            if test_ds is not None:
                mlflow.models.evaluate(  # type: ignore[no-untyped-call]
                    model=_predict_fn,
                    data=test_ds,
                    model_type="classifier",
                    evaluators=["default"],
                    evaluator_config={"default": {"log_explainer": False, "metric_prefix": "test_"}},
                )

        eval_res = payload.eval_result
        lat = payload.latency_profile
        sla_met = lat.p95_ms <= payload.latency_slo_ms

        metrics = {
            "test_worst_class_ap": round(eval_res.worst_class_ap, 4),
            "test_median_class_ap": round(eval_res.median_class_ap, 4),
            "eval.top1_accuracy": round(eval_res.test_accuracy, 4),
            "eval.fp32_to_int8_accuracy_drop": round(eval_res.fp32_to_int8_drop, 4),
            "eval.empirical_test_coverage": round(eval_res.empirical_coverage, 4),
            "eval.conformal_empirical_coverage": round(eval_res.empirical_coverage, 4),
            "eval.mean_prediction_set_size": round(eval_res.mean_set_size, 2),
            "eval.singleton_rate": round(eval_res.singleton_rate, 4),
            "eval.empty_set_rate": round(eval_res.empty_set_rate, 4),
            "eval.ambiguous_referral_rate": round(eval_res.ambiguous_referral_rate, 4),
            "eval.ece": round(eval_res.ece, 4),
            "eval.p50_latency_ms": lat.p50_ms,
            "eval.p95_latency_ms": lat.p95_ms,
            "eval.throughput_fps": lat.throughput_fps,
            "eval.latency_sla_satisfied": 1.0 if sla_met else 0.0,
        }

        if ctx.model_id and test_ds is not None:
            try:
                mlflow.log_metrics(metrics, model_id=ctx.model_id, dataset=test_ds)
            except Exception:
                with contextlib.suppress(Exception):
                    mlflow.log_metrics(metrics, model_id=ctx.model_id)
        elif ctx.model_id:
            with contextlib.suppress(Exception):
                mlflow.log_metrics(metrics, model_id=ctx.model_id)
        else:
            with contextlib.suppress(Exception):
                mlflow.log_metrics(metrics)

        with contextlib.suppress(Exception):
            mlflow.log_table(data=payload.per_taxon_df, artifact_file="reports/per_taxon_holdout_audit.json")

        _log_pyfunc_and_promotion(payload, ctx)

        with contextlib.suppress(Exception):
            vis_payload = HoldoutVisualizationPayload(
                test_labels=eval_res.labels,
                test_probs=eval_res.probs,
                test_logits=eval_res.logits,
                set_sizes=eval_res.set_sizes,
                q_hat=eval_res.q_hat,
                nominal_coverage=eval_res.nominal_coverage,
                empirical_coverage=eval_res.empirical_coverage,
                calibrated_energy_threshold=eval_res.tau_ood,
                k_max=eval_res.k_max,
                taxa_catalog=payload.taxa_catalog,
                pareto_profiles=payload.pareto_profiles,
                latency_slo_ms=payload.latency_slo_ms,
            )
            log_holdout_visualizations(vis_payload)

        for artefact_path in (
            Path("reports/train_metrics.json"),
            Path("reports/pareto_evaluation.json"),
            payload.conformal_config_path,
            Path("docs/assets/pareto_results.json"),
        ):
            if artefact_path.is_file():
                with contextlib.suppress(Exception):
                    mlflow.log_artifact(str(artefact_path))
