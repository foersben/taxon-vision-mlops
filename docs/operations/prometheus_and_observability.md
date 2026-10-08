---
type: Reference
title: Prometheus Telemetry & Drift Observability
status: stable
stale_after: "2027-01-01T00:00:00Z"
version: 1.0
description: Telemetry metrics, latency histograms, and embedding drift monitors.
tags: [prometheus, observability, drift]
generated: {by: process:scaffold-init, at: "2026-10-02T10:00:00Z"}
verified: {by: process:scaffold-init, at: "2026-10-02T10:00:00Z"}
---

# Prometheus Telemetry & Drift Observability

Tracks inference latency, prediction counts by taxon, conformal set size distributions, and human referral rates in real time.

## Accurate Prometheus Telemetry & Error Masking Resolution

**What we chose:**
Inference requests that fail due to image decoding errors (e.g. malformed uploads) are caught and explicitly logged to Prometheus with `status="fallback"`, returning a synthetic fallback prediction (Monarch Butterfly). Conversely, actual runtime exceptions (like CUDA OOM or network drops) are **not** caught silently; they increment the counter with `status="error"`, log the full traceback, and bubble up as HTTP 500 errors.

**Why we chose it (Logically Validated):**
Previously, a bare `except Exception:` block caught all errors indiscriminately, masking systemic outages behind synthetic butterfly predictions. This polluted the Prometheus metrics because `status="success"` was incremented regardless of the internal failure. By moving the success increment *after* a verified forward pass and explicitly differentiating decode errors from systemic crashes, we ensure the monitoring dashboards reflect the true operational health of the inference engine. If the deployment fails, PagerDuty will trigger immediately rather than failing silently with 100% "success" rates.
