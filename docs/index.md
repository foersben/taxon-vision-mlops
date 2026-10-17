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
flowchart TD
    A["raw_manifest.parquet<br/>(pipelines.ingest)"] --> B["dataset_manifest.parquet<br/>(pipelines.audit_license)"]
    B --> C["4-Way Split Embeddings<br/>(pipelines.extract_features)"]
    C --> D["head.pt<br/>(pipelines.train)"]
    D --> E["model_int8.onnx<br/>(pipelines.export_onnx)"]
    C -.-> F["conformal_config.yaml<br/>(pipelines.calibrate)"]
    E --> F
    C -.-> G["pareto_evaluation.json<br/>(pipelines.evaluate)"]
    E --> G
    F --> G
```

## System Tenets

* **Epistemic Diamond DAG Topology:** A 7-stage DVC pipeline under `src/taxon_vision/pipelines/` governed by root `params.yaml`, enforcing strict license compliance before S3 streaming, zero-leakage 4-way embedding splits (`train_embeddings.pt`, `val_embeddings.pt`, `cal_embeddings.pt`, `test_embeddings.pt`), and post-quantization calibration on `model_int8.onnx`.
* **Modern Foundation Backbones:** Seamlessly supports state-of-the-art vision extractors including **DINOv3**, BioCLIP-2, DINOv2, and edge-optimized MobileNetV4 architectures.
* **Production Chain Over Notebook Glory:** A versioned, audited, and automated DVC pipeline where code, parameters, and tensor hashes guarantee end-to-end reproducibility.
* **Mathematical Safety Invariants:** Predictions are accompanied by distribution-free **Split Conformal Prediction** sets guaranteeing a user-specified coverage bound ($1 - \alpha = 0.95$) alongside Helmholtz OOD energy gating.
* **Open Licensing & Attribution:** Adheres to DarwinCore and Creative Commons standards, strictly filtering open licenses (CC0, CC-BY, CC-BY-NC) and preserving photographer attribution.

## System Reference

For detailed subsystem architecture diagrams, storage technology selections, and complete parameter matrices, consult the [Architecture, Storage & Parameter Reference](foundations/system_reference.md).
