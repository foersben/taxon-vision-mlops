---
type: Agent Rule
title: Mandates
status: stable
stale_after: "2027-01-01T00:00:00Z"
version: 1.0
description: "Zero memory allocations on hot inference path and strict p95 latency budgets."
tags: [onnx, performance]
generated: {by: process:scaffold-init, at: "2026-10-02T10:00:00Z"}
verified: {by: process:scaffold-init, at: "2026-10-02T10:00:00Z"}
trigger: always_on
severity: critical
---

# Mandates: Inference Latency Invariants

* Zero memory allocations on hot inference path and strict p95 latency budgets.
