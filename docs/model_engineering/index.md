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

## Thematic Sections

* **[Vision Model Architecture](model_architecture.md)**: Structure of the foundational feature extractors and linear heads.
* **[Training Pipeline](training_pipeline.md)**: End-to-end execution of the model training and tuning loops.
* **[Backbone Pareto Benchmarks](backbone_benchmarking_pareto.md)**: Empirical analysis of accuracy, latency, and operational cost.
* **[ONNX Runtime Optimization](onnx_optimization_and_latency.md)**: Static INT8 PTQ quantization for maximizing inference throughput.
