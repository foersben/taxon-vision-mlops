---
type: Workflow
title: Conformal Uncertainty Audit
status: stable
stale_after: "2027-01-01T00:00:00Z"
version: 1.0
description: Verifies statistical coverage guarantees (1 - alpha), OOD calibration thresholds, and human referral rates.
tags: [workflow, conformal, uncertainty, calibration]
generated: {by: process:okf-updater, at: "2026-10-17T22:30:00Z"}
verified: {by: process:okf-updater, at: "2026-10-17T22:30:00Z"}
---

# Conformal Uncertainty Audit (/conformal-uncertainty-audit)

* **Description:** Verifies non-conformity score quantile calibration, energy-based out-of-distribution (OOD) decision boundaries, and statistical coverage guarantees on holdout evaluation sets under root `params.yaml`.
* **Invocation:** Triggered via `/conformal-uncertainty-audit` or `/audit-conformal` in the AI IDE.

## Step-by-Step Procedure

* Calibrate Split Conformal non-conformity quantiles and Helmholtz OOD thresholds using `cal_embeddings.pt` and `model_int8.onnx`:

```bash
pixi run -e dev python -m taxon_vision.pipelines.calibrate \
  --onnx-model models/checkpoints/model_int8.onnx \
  --embeddings data/embeddings/cal_embeddings.pt \
  --out config/conformal_config.yaml \
  --metrics-out reports/conformal_metrics.json \
  --alpha 0.05 \
  --ood-tpr 0.95
```

* Inspect generated calibration parameters in `config/conformal_config.yaml` and `reports/conformal_metrics.json`:
    * Verify `alpha: 0.05` corresponds to 95% nominal confidence level.
    * Verify `q_hat` is computed via finite-sample empirical correction `ceil((n + 1) * (1 - alpha)) / n`.
    * Verify `ood_energy_threshold` (and `tau_ood`) is calibrated at 95% true positive rate on in-distribution logits.

* Run Pareto holdout evaluation against `test_embeddings.pt` to assert empirical coverage meets nominal guarantees:

```bash
pixi run -e dev python -m taxon_vision.pipelines.evaluate \
  --onnx-model models/checkpoints/model_int8.onnx \
  --embeddings data/embeddings/test_embeddings.pt \
  --conformal-config config/conformal_config.yaml \
  --out docs/assets/pareto_results.json \
  --metrics-out reports/pareto_evaluation.json \
  --latency-sla-ms 25.0
```

* Run the mathematical coverage invariance tests:

```bash
pixi run -e dev pytest tests/invariants/test_conformal_coverage_invariant.py tests/unit/test_conformal_bounds.py -q -o "addopts="
```
