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

## Prometheus Metrics Specification

The following metrics are exposed to track system health, uncertainty, and performance:

* **`inference_latency_seconds_bucket`:** Histogram metric capturing the time taken to process an inference request. It allows calculation of p50, p95, and p99 latency percentiles to monitor adherence to latency budgets.
* **`conformal_prediction_set_size_distribution`:** Histogram metric tracking the cardinality of prediction sets returned by the conformal predictor.
* **`conformal_empty_set_total`:** Counter metric tracking occurrences where the conformal predictor returns an empty set (i.e. model is completely unsure).
* **`ood_energy_score_distribution`:** Histogram metric tracking the energy scores for out-of-distribution detection.
* **`taxonomic_triage_referred_total`:** Counter metric tracking predictions deferred for manual human triage.

## Covariate and Concept Drift Detection

To ensure models maintain performance in production over time, we employ real-time drift detection on latent vision embeddings. The system continuously evaluates statistical distance metrics such as **Wasserstein distance** and **Maximum Mean Discrepancy (MMD)** between incoming production data and the calibration set. Significant drift violations automatically trigger retraining DAGs to recalibrate the model.

## Accurate Prometheus Telemetry & Error Masking Resolution

**What we chose:**
Inference requests that fail due to image decoding errors (e.g. malformed uploads) are caught and explicitly logged to Prometheus with `status="fallback"`, returning a synthetic fallback prediction (Monarch Butterfly). Conversely, actual runtime exceptions (like CUDA OOM or network drops) are **not** caught silently; they increment the counter with `status="error"`, log the full traceback, and bubble up as HTTP 500 errors.

**Why we chose it (Logically Validated):**
Previously, a bare `except Exception:` block caught all errors indiscriminately, masking systemic outages behind synthetic butterfly predictions. This polluted the Prometheus metrics because `status="success"` was incremented regardless of the internal failure. By moving the success increment *after* a verified forward pass and explicitly differentiating decode errors from systemic crashes, we ensure the monitoring dashboards reflect the true operational health of the inference engine. If the deployment fails, PagerDuty will trigger immediately rather than failing silently with 100% "success" rates.
