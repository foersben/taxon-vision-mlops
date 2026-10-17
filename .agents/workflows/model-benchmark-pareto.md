---
type: Workflow
title: Model Benchmark Pareto
status: stable
stale_after: "2027-01-01T00:00:00Z"
version: 1.0
description: Protocol for evaluating accuracy vs latency vs cost across candidate backbones.
tags: [workflow, benchmark, pareto, latency, onnx]
generated: {by: process:okf-updater, at: "2026-10-17T22:30:00Z"}
verified: {by: process:okf-updater, at: "2026-10-17T22:30:00Z"}
---

# Model Benchmark Pareto (/model-benchmark-pareto)

* **Description:** Comprehensive benchmarking protocol evaluating Top-1 accuracy, p95 ONNX Runtime latency, memory footprint, and query cost profiles across candidate foundation vision backbones (DINOv3, BioCLIP-2, DINOv2, MobileNetV4, EfficientNet).
* **Invocation:** Triggered via `/model-benchmark-pareto` or `/benchmark-pareto` in the AI IDE.

## Step-by-Step Procedure

* Ensure the quantized production model and holdout test embeddings are available:

```bash
pixi run -e dev dvc pull models/checkpoints/model_int8.onnx data/embeddings/test_embeddings.pt
```

* Execute the canonical Stage 7 evaluation pipeline enforcing the 25.0 ms fast-path SLA from root `params.yaml`:

```bash
pixi run -e dev python -m taxon_vision.pipelines.evaluate \
  --onnx-model models/checkpoints/model_int8.onnx \
  --embeddings data/embeddings/test_embeddings.pt \
  --conformal-config config/conformal_config.yaml \
  --out docs/assets/pareto_results.json \
  --metrics-out reports/pareto_evaluation.json \
  --latency-sla-ms 25.0
```

* Inspect the serialized metrics and Pareto frontier:
    * `reports/pareto_evaluation.json`: Captures holdout top-1 accuracy, nominal vs. empirical conformal coverage, mean prediction set size, and p50/p95 latency metrics.
    * `docs/assets/pareto_results.json`: Exports full comparative architectural profiles for interactive documentation rendering.

* Execute unit benchmark regression tests:

```bash
pixi run -e dev pytest tests/unit/test_quantizer.py tests/unit/test_onnx_exporter.py -q -o "addopts="
```
