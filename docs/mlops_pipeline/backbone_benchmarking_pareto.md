---
type: Reference
title: Backbone Benchmarking & Pareto Trade-Offs
status: stable
stale_after: "2027-01-01T00:00:00Z"
version: 1.0
description: Pareto analysis comparing candidate extractors on accuracy, latency, and cloud cost.
tags: [benchmarks, pareto, bioclip, dinov2, mobilenet]
generated: {by: process:scaffold-init, at: "2026-10-02T10:00:00Z"}
verified: {by: process:scaffold-init, at: "2026-10-02T10:00:00Z"}
---

# Backbone Benchmarking & Pareto Trade-Offs

The project evaluates four candidate extractors via `scripts/benchmark_pareto.py`:

| Extractor | Top-1 Acc | p95 Latency | Throughput | Cost (\$/1M Queries) |
| --- | --- | --- | --- | --- |
| `bioclip-2` | 91.2% | 24.2 ms | 54.3 fps | \$4.20 |
| `dinov2_vits14` | 89.4% | 16.8 ms | 82.6 fps | \$2.75 |
| `mobilenetv4_conv_small` | 83.5% | 4.9 ms | 312.5 fps | \$0.68 |
| `efficientnet_b0` | 85.8% | 8.1 ms | 172.4 fps | \$1.25 |
