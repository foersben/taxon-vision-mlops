---
type: Concept
title: Model & Inference Engineering
status: stable
stale_after: "2027-01-01T00:00:00Z"
version: "1.0"
description: Architectural design of vision backbones, Class-Balanced Loss optimization, static INT8 quantization, and latency benchmarking.
tags: [models, inference, backbones, optimization, onnx, pareto]
generated: {by: "process:docs-librarian", at: "2026-10-18T18:00:00Z"}
verified: {by: "process:docs-librarian", at: "2026-10-18T18:00:00Z"}
---

# Model & Inference Engineering

This section focuses on the mathematical heart of the platform: the neural architectures and their highly optimized execution runtimes.

We cover the entire lifecycle of the classification engine, beginning with the selection and construction of the vision backbones (e.g., DINOv3, BioCLIP-2, MobileNetV4). The documentation walks through the GPU-accelerated training pipeline, the implementation of Class-Balanced Loss, and the rigorous Pareto benchmarking used to evaluate model candidates. Finally, it explores critical optimization techniques - specifically static ONNX Post-Training Quantization (PTQ) and forward-hooked CAM - required to serve these models within strict latency budgets.

## Recommended Reading Order

* **[Vision Model Architecture](model_architecture.md)**: Structure of the frozen foundation feature extractors (`dinov3_vits14`, `bioclip-2`, `dinov2_vits14`, `mobilenetv4_conv_small`) and convex linear probe heads.
* **[Training Pipeline & Optuna HPO](training_pipeline.md)**: End-to-end execution of the GPU training loop, Class-Balanced Loss, and PostgreSQL-backed Optuna TPE + ASHA hyperparameter tuning.
* **[MLflow 3.x Tracking & Registry](mlflow_tracking_and_registry.md)**: Cross-stage MLflow run continuity (`resume_parent_mlflow_run`), 5-dataset `mlflow.data` lineage, `LoggedModel` packaging (`TaxonVisionPyFuncModel`), and the 6 holdout diagnostic plots.
* **[Four-Gate Model Promotion & Passport](model_promotion_gates.md)**: Post-evaluation promotion governance (INT8 drop, Macro PR-AUC, p95 latency, and MMD² calibration exchangeability) and the live HTMX Active Model Passport.
* **[ONNX Runtime Optimization & CAM](onnx_optimization_and_latency.md)**: Dual-output ONNX export (`logits` + pre-normalised `embeddings`), logit-protecting INT8 PTQ, and forward-hooked analytical CAM.
* **[Backbone Pareto Benchmarks](backbone_benchmarking_pareto.md)**: Empirical analysis of accuracy, latency, and operational cost across candidate vision trunks.
