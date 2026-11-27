# SPDX-FileCopyrightText: 2026 Benjamin Förster
# SPDX-License-Identifier: MIT
"""Stage 7: Model Pareto Frontier, Holdout Statistical Evaluation & MLflow Audit CLI.

Why:
    Promoting a candidate model based solely on validation PR-AUC violates core MLOps invariants. Stage 7 coordinates unbiased holdout verification directly against the quantised ONNX graph via `eval_metrics` and unifies MLflow audit logging and promotion gating via `eval_reporting`.

How:
    Parses CLI arguments mapped from `params.yaml` / `dvc.yaml`, prints Pareto comparison tables, delegates holdout conformal evaluation and CPU ONNX latency profiling to `eval_metrics`, writes JSON evaluation artefacts, and dispatches MLflow parent-run logging and Stage 8 promotion via `eval_reporting`.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd

from taxon_vision.inference.mlflow_model import load_catalog_entries
from taxon_vision.pipelines.eval_metrics import (
    ONNXLatencyProfile,
    build_per_taxon_audit_table,
    evaluate_onnx_holdout,
    get_canonical_pareto_profiles,
    profile_onnx_inference,
)
from taxon_vision.pipelines.eval_reporting import (
    MLflowHoldoutLoggingPayload,
    log_holdout_audit_to_mlflow,
)


def _build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser for Stage 7: Evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate Pareto backbone profiles and holdout test performance.")
    parser.add_argument(
        "--onnx-model",
        type=Path,
        default=Path("models/checkpoints/model_int8.onnx"),
        help="Path to trained quantized ONNX graph.",
    )
    parser.add_argument(
        "--embeddings",
        type=Path,
        default=Path("data/embeddings/test_embeddings.pt"),
        help="Path to test embeddings partition (.pt).",
    )
    parser.add_argument(
        "--train-embeddings",
        type=Path,
        default=Path("data/embeddings/train_embeddings.pt"),
        help="Path to train embeddings partition (.pt) for sample count alignment.",
    )
    parser.add_argument(
        "--conformal-config",
        type=Path,
        default=Path("config/conformal_config.yaml"),
        help="Path to calibrated conformal configuration YAML.",
    )
    parser.add_argument(
        "--latency-slo-ms",
        "--latency-sla-ms",
        dest="latency_sla_ms",
        type=float,
        default=25.0,
        help="Fast-path latency SLO threshold in milliseconds (default: 25.0).",
    )
    parser.add_argument("--out", type=Path, default=Path("docs/assets/pareto_results.json"))
    parser.add_argument(
        "--metrics-out",
        type=Path,
        default=None,
        help="Optional destination path to export summary evaluation metrics JSON.",
    )
    return parser


def evaluate_cli(argv: list[str] | None = None) -> int:
    """Evaluate candidate backbone Pareto frontier and holdout test set performance.

    Args:
        argv: List of command-line arguments (default: None, uses sys.argv).

    Returns:
        0 if successful, 1 otherwise.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s", force=True)
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    results = get_canonical_pareto_profiles()
    print("=" * 100)
    print("TAXONVISION BACKBONE PARETO EVALUATION")
    print("=" * 100)
    print(f"{'Extractor':<25} {'Architecture':<28} {'Params':<8} {'Dim':<6} {'p95 Lat (ms)':<14} {'Target Profile'}")
    print("-" * 100)
    for r in results:
        print(
            f"{r.extractor:<25} {r.architecture:<28} {r.parameters_m:<8.1f} {r.feature_dim:<6} {r.p95_latency_ms:<14.1f} {r.target_profile}"
        )
    print("=" * 100)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        json.dump([asdict(r) for r in results], f, indent=2)
    print(f"Pareto profiles exported to {args.out}")

    eval_result = evaluate_onnx_holdout(
        onnx_path=args.onnx_model,
        embeddings_path=args.embeddings,
        conformal_config_path=args.conformal_config,
    )

    test_metrics: dict[str, Any] = {}
    if eval_result:
        test_metrics = {
            "test_sample_count": eval_result.test_sample_count,
            "test_accuracy": round(eval_result.test_accuracy, 4),
            "nominal_coverage": round(eval_result.nominal_coverage, 4),
            "test_empirical_coverage": round(eval_result.empirical_coverage, 4),
            "coverage_guarantee_satisfied": eval_result.coverage_guarantee_satisfied,
            "mean_set_size": round(eval_result.mean_set_size, 2),
            "singleton_rate": round(eval_result.singleton_rate, 4),
            "empty_set_rate": round(eval_result.empty_set_rate, 4),
            "referral_rate": round(eval_result.ambiguous_referral_rate, 4),
            "test_worst_class_ap": round(eval_result.worst_class_ap, 4),
            "test_median_class_ap": round(eval_result.median_class_ap, 4),
            "fp32_to_int8_accuracy_drop": round(eval_result.fp32_to_int8_drop, 4),
            "ece": round(eval_result.ece, 4),
            "q_hat": round(eval_result.q_hat, 4),
            "alpha": round(eval_result.alpha, 4),
        }
        print("\nHOLDOUT TEST SET CONFORMAL EVALUATION:")
        print(f"  Samples:            {test_metrics['test_sample_count']}")
        print(f"  Top-1 Accuracy:     {test_metrics['test_accuracy']:.2%}")
        print(f"  Nominal Coverage:   {test_metrics['nominal_coverage']:.2%}")
        print(f"  Empirical Coverage: {test_metrics['test_empirical_coverage']:.2%}")
        print(f"  Mean Set Size:      {test_metrics['mean_set_size']:.2f}")
        print(f"  Referral Rate:      {test_metrics['referral_rate']:.2%}")
        print(f"  Worst-Class AP:     {test_metrics['test_worst_class_ap']:.4f}")
        print(f"  Median-Class AP:    {test_metrics['test_median_class_ap']:.4f}")
        print(f"  10-bin ECE:         {test_metrics['ece']:.4f}")
        print(f"  Guarantee Met:      {test_metrics['coverage_guarantee_satisfied']}")

    p50, p95, fps = profile_onnx_inference(args.onnx_model)
    sla_met = p95 <= args.latency_sla_ms
    print(f"\nONNX RUNTIME LATENCY PROFILING (Target SLO: {args.latency_sla_ms} ms):")
    print(f"  p50 Latency:        {p50:.2f} ms")
    print(f"  p95 Latency:        {p95:.2f} ms (SLO Satisfied: {sla_met})")
    print(f"  Throughput:         {fps:.1f} FPS")

    if args.metrics_out is not None:
        args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
        export_payload = {
            "p50_latency_ms": p50,
            "p95_latency_ms": p95,
            "throughput_fps": fps,
            "latency_sla_ms": args.latency_sla_ms,
            "latency_sla_satisfied": sla_met,
            **test_metrics,
        }
        with args.metrics_out.open("w", encoding="utf-8") as f:
            json.dump(export_payload, f, indent=2)
        print(f"Evaluation metrics exported to {args.metrics_out}")

    if eval_result:
        taxa_catalog = load_catalog_entries(Path("config/taxa_catalog.yaml"))
        per_taxon_df: pd.DataFrame = build_per_taxon_audit_table(
            eval_result=eval_result,
            taxa_catalog=taxa_catalog,
            train_embeddings_path=args.train_embeddings,
        )
        logging_payload = MLflowHoldoutLoggingPayload(
            eval_result=eval_result,
            latency_profile=ONNXLatencyProfile(p50_ms=p50, p95_ms=p95, throughput_fps=fps),
            pareto_profiles=results,
            per_taxon_df=per_taxon_df,
            taxa_catalog=taxa_catalog,
            onnx_path=args.onnx_model,
            conformal_config_path=args.conformal_config,
            latency_slo_ms=args.latency_sla_ms,
        )
        log_holdout_audit_to_mlflow(logging_payload)

    return 0


main = evaluate_cli

if __name__ == "__main__":
    sys.exit(evaluate_cli())
