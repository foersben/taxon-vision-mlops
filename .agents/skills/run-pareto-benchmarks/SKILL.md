---
type: Agent Skill
title: Run Pareto Benchmarks
status: stable
stale_after: "2027-01-01T00:00:00Z"
version: 1.0
description: Skill to evaluate candidate backbone (DINOv3, BioCLIP-2, DINOv2, MobileNetV4) accuracy vs latency vs cost.
tags: [skill, python, pareto, benchmarks, evaluation]
generated: {by: process:okf-updater, at: "2026-10-17T22:30:00Z"}
verified: {by: process:okf-updater, at: "2026-10-17T22:30:00Z"}
name: Run Pareto Benchmarks
sources:
  - id: evaluate_pipeline
    resource: src/taxon_vision/pipelines/evaluate.py
  - id: benchmark_pareto_script
    resource: scripts/benchmark_pareto.py
---

# Run Pareto Benchmarks

Skill to evaluate candidate vision backbones across Top-1 accuracy, p95 ONNX Runtime latency, memory footprint, and query cost.

## Execution

* Run canonical Stage 7 evaluation pipeline:

```bash
pixi run -e dev python -m taxon_vision.pipelines.evaluate \
  --onnx-model models/checkpoints/model_int8.onnx \
  --embeddings data/embeddings/test_embeddings.pt \
  --conformal-config config/conformal_config.yaml \
  --out docs/assets/pareto_results.json \
  --metrics-out reports/pareto_evaluation.json \
  --latency-sla-ms 25.0
```

* Alternatively, run the backwards-compatible CLI script:

```bash
pixi run -e dev python scripts/benchmark_pareto.py \
  --onnx-model models/checkpoints/model_int8.onnx \
  --embeddings data/embeddings/test_embeddings.pt \
  --conformal-config config/conformal_config.yaml \
  --out docs/assets/pareto_results.json \
  --metrics-out reports/pareto_evaluation.json \
  --latency-sla-ms 25.0
```
