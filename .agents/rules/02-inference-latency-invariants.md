---
type: Agent Rule
title: "Rule 02: Inference Latency Invariants"
status: stable
stale_after: "2027-01-01T00:00:00Z"
version: 1.0
description: "Zero memory allocations on hot inference paths, strict p95 latency budgets, and quantization standards."
tags: [onnx, latency, performance, quantization]
generated: {by: process:scaffold-init, at: "2026-10-02T10:00:00Z"}
verified: {by: process:okf-updater, at: "2026-10-30T00:00:00Z"}
trigger: always_on
severity: critical
---

# Mandates: Inference Latency Invariants

Inference execution paths must enforce deterministic p95 latency bounds and zero-allocation memory discipline.

## 1. Zero-Allocation Hot Path

* The primary inference routine (`predict`, `predict_logits`, `predict_outputs` in `ONNXInferenceEngine`) must avoid dynamic memory allocations, Python dictionary packaging, or repeated list comprehensions on the execution hot path.
* Input tensors and output prediction buffers must be pre-allocated as contiguous C-order NumPy arrays (`np.ascontiguousarray(..., dtype=np.float32)`).
* In-place buffer reuse must be preferred over repeated array cloning inside high-frequency evaluation loops.

## 2. Strict Latency Budgets & SLA Compliance

All production models must satisfy physical p95 latency ceilings evaluated under steady-state execution:

* **CPU Execution Budget (INT8 Quantized):** End-to-end inference latency (preprocessing + ONNX forward pass) must not exceed $15\,\text{ms}$ p95 on the Intel Core i7-14700K Performance cores.
* **GPU Execution Budget (FP16 CUDA EP):** End-to-end inference latency must not exceed $10\,\text{ms}$ p95 on the NVIDIA GeForce RTX 5070 Ti.
* **Microservice SLA Ceiling:** The complete FastAPI HTTP `/predict` endpoint round-trip (including JSON serialization and telemetry observation) must remain $< 25\,\text{ms}$ p95.

## 3. Dual-Output ONNX Export Standard

* Model export via `export_onnx.py` must produce dual-output graphs outputting both classification logits (`[batch, num_classes]`) and normalized feature embeddings (`[batch, embedding_dim]`).
* Separate models for feature extraction and classification are prohibited to eliminate redundant forward passes through the vision backbone.
* INT8 Dynamic Quantization must be validated against the holdout split, guaranteeing top-1 accuracy degradation $\le 0.5\%$ relative to the FP32 baseline.
