---
type: Agent Memory
title: Pareto Frontier & Latency Benchmark Memory
status: stable
stale_after: "2027-01-01T00:00:00Z"
version: 1.0
description: Empirical evaluation results across candidate backbones, quantization formats, and hardware execution budgets.
tags: [pareto, latency, onnx, benchmarks, memory]
generated: {by: process:okf-updater, at: "2026-10-30T00:00:00Z"}
verified: {by: process:okf-updater, at: "2026-10-30T00:00:00Z"}
---

# Pareto Frontier & Latency Benchmark Memory

## 1. Hardware Execution Platform

* **CPU:** Intel Core i7-14700K (8 Performance-cores pinned `0-15` via `taskset`, 12 Efficient-cores `16-27`).
* **GPU:** NVIDIA GeForce RTX 5070 Ti (16 GB GDDR7 VRAM, Blackwell architecture cc 12.0).
* **RAM:** 128 GB DDR5.
* **Storage:** 2 TB PCIe Gen4 NVMe (BTRFS `zstd:3`, `chattr +C` non-CoW on DB/DVC storage).

## 2. Backbone Benchmark Landscape

* **DINOv3 (ViT-B/16):** Highest biological embedding richness and top-1 accuracy; suitable for offline batch feature extraction and training split preparation.
* **BioCLIP-2:** Optimal zero-shot taxonomic grounding and hierarchical label clustering.
* **DINOv2 (ViT-S/14):** Intermediate Pareto point balancing representation quality with serving latency.
* **MobileNetV4 (Conv / Hybrid):** Ultra-low latency target ($< 15\,\text{ms}$ p95 ONNX CPU) for edge field observation triage.

## 3. Quantization Latency Profile (RTX 5070 Ti / ONNX Runtime)

* **FP32 Baseline:** 18.4 ms p95 latency, 342 MB working memory footprint.
* **FP16 TensorRT / CUDA EP:** 7.2 ms p95 latency, 172 MB footprint.
* **INT8 Dynamic Quantization (CPU EP fallback):** 11.8 ms p95 latency, 89 MB footprint, $< 0.4\%$ top-1 degradation.
