---
type: Agent Skill
title: Run Pareto Benchmarks
status: stable
stale_after: "2027-01-01T00:00:00Z"
version: 1.0
description: Skill to evaluate candidate backbone (DINOv3, BioCLIP-2, DINOv2, MobileNetV4) accuracy vs latency vs cost.
tags: [skill, python]
generated: {by: process:okf-updater, at: "2026-10-02T10:00:00Z"}
verified: {by: process:okf-updater, at: "2026-10-02T10:00:00Z"}
name: Run Pareto Benchmarks
sources:
- id: run_pareto_benchmarks
  resource: scripts/benchmark_pareto.py
---

# Run Pareto Benchmarks

Skill to evaluate candidate backbone (DINOv3, BioCLIP-2, DINOv2, MobileNetV4) accuracy vs latency vs cost.

## Execution

Run using:

```bash
pixi run -e dev python scripts/benchmark_pareto.py
```
