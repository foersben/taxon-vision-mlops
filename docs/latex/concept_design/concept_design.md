---
type: Concept
title: Concept Design & Architectural Specification
status: stable
stale_after: "2027-01-01T00:00:00Z"
version: 1.0
description: Conceptual design and LaTeX documentation specification for TaxonVision-MLOps across 4 operational levels.
tags: [concept-design, architecture, latex, mlops]
generated: {by: process:concept-architect, at: "2026-10-02T10:00:00Z"}
verified: {by: process:concept-architect, at: "2026-10-02T10:00:00Z"}
---

# Concept Design & Architectural Specification

This document details the architectural concept design, operational levels, and mathematical invariants for TaxonVision-MLOps, serving as the Open Knowledge Format (OKF v0.2) counterpart to `docs/latex/concept_design/concept_design.tex`.

## Architectural Tenets

The platform prioritizes the operational production chain over isolated model performance: a versioned, monitored, deployed, and retrainable service is superior to complex unmanaged notebook prototypes.

## Four-Level Operational Architecture

* **Level 1: Restricted Perimeter & Closed Chain:**
  * Fixed pre-trained feature extractors (via `timm`) paired with a modular classification head.
  * Ingestion of well-represented taxa from iNaturalist Open Data AWS S3 buckets.
  * Strict filtering for open licenses (CC0, CC-BY, CC-BY-NC) with full photographer attribution preservation.
  * FastAPI serving endpoints exposing both REST APIs and HTMX operator dashboards.

* **Level 2: Scalability, Cost & Pareto Optimization:**
  * Multi-backbone evaluation (DINOv3, BioCLIP-2, DINOv2, MobileNetV4) balancing Top-1 accuracy against latency and serving cost.
  * ONNX Runtime INT8/FP16 quantization targeting zero-allocation inference on hot paths.
  * DVC dataset versioning for scalable, reproducible storage.

* **Level 3: Feedback Loops & Production Observability:**
  * Closed-loop ingestion of delayed community validations and historical performance tracking.
  * Linnaean taxonomic mutation handling (splits, lumps, revisions) mapped to DarwinCore IDs.
  * Prometheus telemetry for prediction entropy, drift detection, and referral rates.

* **Level 4: Mathematical Robustness & Active Learning:**
  * Split Conformal Prediction ensuring distribution-free coverage guarantees:
    $$P(Y_{n+1} \in \hat{C}(X_{n+1})) \ge 1 - \alpha$$
  * Energy-based Out-of-Distribution (OOD) scoring to detect non-organism or empty images.
  * Active learning query strategies (BADGE, CoreSet) prioritizing human expert triage.
  * Sub-25ms Grad-CAM explainability heatmaps.

## Artifact Manifest

* `concept_design.tex`: Formal LaTeX manuscript suitable for academic or technical reporting.
* `concept_design.md`: OKF v0.2 architectural reference document.
