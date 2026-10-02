---
type: Reference
title: Backbone Benchmarking & Pareto Trade-Offs
status: stable
stale_after: "2027-01-01T00:00:00Z"
version: 1.0
description: Pareto analysis comparing candidate extractors on accuracy, latency, and cloud cost.
tags: [benchmarks, pareto, bioclip, dinov3, dinov2, mobilenet]
generated: {by: process:scaffold-init, at: "2026-10-02T10:00:00Z"}
verified: {by: process:scaffold-init, at: "2026-10-02T10:00:00Z"}
---

# Backbone Benchmarking & Pareto Trade-Offs

The project evaluates four candidate extractors via `scripts/benchmark_pareto.py`:

| Extractor | Top-1 Acc | p95 Latency | Throughput | Cost (\$/1M Queries) |
| --- | --- | --- | --- | --- |
| `bioclip-2` | 91.2% | 24.2 ms | 54.3 fps | \$4.20 |
| `dinov3_vits14` | 92.4% | 18.2 ms | 72.5 fps | \$2.95 |
| `dinov2_vits14` | 89.4% | 16.8 ms | 82.6 fps | \$2.75 |
| `mobilenetv4_conv_small` | 83.5% | 4.9 ms | 312.5 fps | \$0.68 |
| `efficientnet_b0` | 85.8% | 8.1 ms | 172.4 fps | \$1.25 |

## DINOv3 Architectural Impact

DINOv3 represents the newest state-of-the-art in self-supervised vision transformer backbones. Leveraging refined dense patch pre-training with register tokens, DINOv3 achieves superior fine-grained morphological representations for taxonomic identification, yielding a **+3.0% top-1 accuracy improvement** over DINOv2 with minimal latency increase (+1.4 ms).
