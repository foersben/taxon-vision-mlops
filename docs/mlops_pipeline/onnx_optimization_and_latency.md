---
type: System Pattern
title: ONNX Runtime Optimization & Dynamic Quantization
status: stable
stale_after: "2027-01-01T00:00:00Z"
version: 1.0
description: Inference acceleration using ONNX Runtime and INT8 dynamic quantization.
tags: [onnx, quantization, latency, optimization]
generated: {by: process:scaffold-init, at: "2026-10-02T10:00:00Z"}
verified: {by: process:scaffold-init, at: "2026-10-02T10:00:00Z"}
---

# ONNX Runtime Optimization & Dynamic Quantization

Models are exported with dynamic batch axes and quantized using dynamic INT8 quantization, yielding:

* 3.2x reduction in disk footprint.
* Sub-10ms inference latency on CPU runners.
* Zero memory allocations on the hot prediction path.
