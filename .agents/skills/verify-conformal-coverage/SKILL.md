---
type: Agent Skill
title: Verify Conformal Coverage
status: stable
stale_after: "2027-01-01T00:00:00Z"
version: 1.0
description: Skill to assert empirical coverage meets nominal 1 - alpha rate.
tags: [skill, python, conformal, uncertainty, calibration]
generated: {by: process:okf-updater, at: "2026-10-17T22:30:00Z"}
verified: {by: process:okf-updater, at: "2026-10-17T22:30:00Z"}
name: Verify Conformal Coverage
sources:
  - id: calibrate_pipeline
    resource: src/taxon_vision/pipelines/calibrate.py
  - id: calibrate_conformal_script
    resource: scripts/calibrate_conformal.py
---

# Verify Conformal Coverage

Skill to calibrate non-conformity quantiles and Helmholtz OOD thresholds ensuring empirical coverage meets the nominal $1 - \alpha$ statistical guarantee.

## Execution

* Run canonical Stage 6 calibration pipeline:

```bash
pixi run -e dev python -m taxon_vision.pipelines.calibrate \
  --onnx-model models/checkpoints/model_int8.onnx \
  --embeddings data/embeddings/cal_embeddings.pt \
  --out config/conformal_config.yaml \
  --metrics-out reports/conformal_metrics.json \
  --alpha 0.05 \
  --ood-tpr 0.95
```

* Alternatively, run the backwards-compatible CLI script:

```bash
pixi run -e dev python scripts/calibrate_conformal.py \
  --onnx-model models/checkpoints/model_int8.onnx \
  --embeddings data/embeddings/cal_embeddings.pt \
  --out config/conformal_config.yaml \
  --metrics-out reports/conformal_metrics.json \
  --alpha 0.05 \
  --ood-tpr 0.95
```

* Verify coverage invariance with test suite:

```bash
pixi run -e dev pytest tests/invariants/test_conformal_coverage_invariant.py -q -o "addopts="
```
