---
type: Concept
title: TaxonVision-MLOps Architecture & Platform Overview
status: stable
stale_after: "2027-01-01T00:00:00Z"
version: 1.0
description: Executive architectural overview of the TaxonVision-MLOps automated biodiversity species identification service and engineering portal.
tags: [mlops, architecture, inaturalist, conformal-prediction, vision-rag, kubernetes]
generated: {by: process:docs-librarian, at: "2026-10-18T12:00:00Z"}
verified: {by: process:docs-librarian, at: "2026-10-18T12:00:00Z"}
---

# TaxonVision-MLOps Platform Overview

TaxonVision-MLOps is an automated, end-to-end machine learning operations platform and production service designed for biodiversity species identification from citizen-science photographs. The system bridges cloud-based GitHub Actions quality gates with private bare-metal NVIDIA GPU training clusters, enforcing distribution-free **Split Conformal Prediction** guarantees and sub-15ms inference latency.

<div class="metric-grid">
    <div class="metric-card">
        <div class="metric-header">Inference SLA</div>
        <div class="metric-value">&lt; 15 ms</div>
        <div class="metric-desc">Sub-15ms p95 latency on bare-metal hardware via zero-allocation ONNX Runtime INT8 engines.</div>
    </div>
    <div class="metric-card">
        <div class="metric-header">Conformal Coverage</div>
        <div class="metric-value">96.5%</div>
        <div class="metric-desc">Mathematically guaranteed empirical coverage satisfying nominal bound ($1 - \alpha = 0.95$).</div>
    </div>
    <div class="metric-card">
        <div class="metric-header">Feature Extraction</div>
        <div class="metric-value">768-D</div>
        <div class="metric-desc">Dense, L2-normalized morphological representations extracted via frozen ViT foundation backbones.</div>
    </div>
    <div class="metric-card">
        <div class="metric-header">License Compliance</div>
        <div class="metric-value">100%</div>
        <div class="metric-desc">Strict open licensing (CC0, CC-BY, CC-BY-NC) with full photographer attribution provenance.</div>
    </div>
</div>

---

## Interactive Architecture Hub & Visualizers

Explore the complete system architecture, DVC data lineages, and mathematical safety boundaries through our dedicated interactive explorers:

* **[Interactive Architecture & Knowledge Graph Hub](interactive_hub.md):** Full-screen physics-based Cytoscape.js network visualizer embedding the complete production runtime and OKF v0.2 knowledge graph.
* **[Interactive Conformal & OOD Simulator](uncertainty_and_safety/interactive_simulator.md):** Real-time simulator to adjust significance levels $\alpha$, explore finite-sample quantile inflation, and simulate human referral depth.
* **[Software & Config Architecture Explorer](architecture/codebase_explorer.md):** Structural guide to the 10 high-cohesion subpackages in `src/taxon_vision/`.

---

## Clickable System Architecture Map

The diagram below maps the end-to-end lifecycle from raw citizen-science image ingestion to zero-downtime Kubernetes serving. Click any stage node to jump directly to its technical specification:

```mermaid
flowchart TD
    subgraph S1["Data Ingestion & Licensing"]
        A["raw_manifest.parquet<br/>(pipelines.ingest)"] --> B["dataset_manifest.parquet<br/>(pipelines.audit_license)"]
    end

    subgraph S2["Feature Extraction & Partitioning"]
        B --> C["4-Way Split Embeddings<br/>(pipelines.extract_features)"]
    end

    subgraph S3["Model Training & Optimization"]
        C --> D["head.pt<br/>(pipelines.train)"]
        D --> E["model_int8.onnx<br/>(pipelines.export_onnx)"]
    end

    subgraph S4["Epistemic Calibration & Gating"]
        C -.-> F["conformal_config.yaml<br/>(pipelines.calibrate)"]
        E --> F
        C -.-> G["pareto_evaluation.json<br/>(pipelines.evaluate)"]
        E --> G
        F --> G
    end

    subgraph S5["Production Serving & UI"]
        E --> H["FastAPI Engine<br/>(service.inference)"]
        F --> H
        H --> I["HTMX Operator Dashboard<br/>(service.routes)"]
    end

    click A "mlops_pipeline/ingestion_and_versioning/" "Read Ingestion Pipeline Docs"
    click B "foundations/taxonomy_and_darwincore/" "Read DarwinCore & License Docs"
    click C "mlops_pipeline/dvc_dag_explorer/" "Read Feature Extraction Docs"
    click D "model_engineering/training_pipeline/" "Read Head Training Docs"
    click E "model_engineering/onnx_optimization_and_latency/" "Read ONNX Export Docs"
    click F "uncertainty_and_safety/conformal_prediction_theory/" "Read Conformal Prediction Docs"
    click G "model_engineering/backbone_benchmarking_pareto/" "Read Pareto Evaluation Docs"
    click H "operations/api_reference/" "Read FastAPI Reference"
    click I "operations/deployment_runbook/" "Read Deployment Runbook"
```

---

## Core System Invariants

* **Epistemic Diamond DAG Topology:** A 7-stage DVC pipeline under `src/taxon_vision/pipelines/` governed by root `params.yaml`, enforcing strict license compliance before S3 streaming, zero-leakage 4-way embedding splits (`train_embeddings.pt`, `val_embeddings.pt`, `cal_embeddings.pt`, `test_embeddings.pt`), and post-quantization calibration on `model_int8.onnx`.
* **Modern Foundation Backbones:** Seamlessly supports state-of-the-art vision extractors including **DINOv3**, BioCLIP-2, DINOv2, and edge-optimized MobileNetV4 architectures with frozen trunks ($\nabla_\theta = 0$).
* **Dual-Pipeline Paradigm:** Decouples weight-centric linear classification probes from data-centric Multimodal Knowledge Graph (MMKG) retrieval to prevent massive retraining cascades.
* **Mathematical Safety Invariants:** Predictions are accompanied by distribution-free **Split Conformal Prediction** sets guaranteeing a user-specified coverage bound ($1 - \alpha = 0.95$) alongside Helmholtz OOD energy gating.
* **Open Licensing & Attribution:** Adheres to DarwinCore and Creative Commons standards, strictly filtering open licenses (CC0, CC-BY, CC-BY-NC) and preserving photographer attribution.
* **Zero-Allocation Fast Path:** Pre-allocated `IOBinding` buffers in ONNX Runtime ensure zero heap allocations during real-time image scoring.
