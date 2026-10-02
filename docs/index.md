---
type: Concept
title: TaxonVision-MLOps Architecture & Platform Overview
status: stable
stale_after: "2027-01-01T00:00:00Z"
version: 1.0
description: Executive overview of the TaxonVision-MLOps automated species identification service.
tags: [mlops, architecture, inaturalist, conformal-prediction]
generated: {by: process:scaffold-init, at: "2026-10-02T10:00:00Z"}
verified: {by: process:scaffold-init, at: "2026-10-02T10:00:00Z"}
---

# TaxonVision-MLOps Platform Overview

TaxonVision-MLOps is an automated, end-to-end machine learning operations pipeline and production service designed for species identification from citizen-science photographs.

```mermaid
flowchart LR
    A["iNaturalist S3 / API"] --> B["Open License Filter"]
    B --> C["Timm Backbone Factory"]
    C --> D["ONNX INT8 Engine"]
    D --> E["Conformal Uncertainty Gate"]
    E -->|Certain| F["FastAPI + HTMX Dashboard"]
    E -->|Ambiguous| G["Human-in-the-Loop Review"]
```

## System Tenets

* **Production Chain Over Notebook Glory:** A simple, versioned, monitored, deployed, and automatically retrainable model is infinitely better than an isolated notebook model.
* **Mathematical Safety Invariants:** Predictions are accompanied by distribution-free **Split Conformal Prediction** sets guaranteeing a user-specified error rate ($1 - \alpha$).
* **Open Licensing & Attribution:** Adheres to DarwinCore and Creative Commons standards, strictly filtering open licenses and preserving photographer attribution.
