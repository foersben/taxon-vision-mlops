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

TaxonVision-MLOps is an automated, end-to-end machine learning operations platform and production service designed for biodiversity species identification from citizen-science photographs. The system bridges cloud-based GitHub Actions quality gates with private bare-metal NVIDIA GPU training clusters, enforcing distribution-free **Split Conformal Prediction** guarantees and verified multi-tier inference latency budgets ($50.38\text{ ms}$ p95 empirical full-graph CPU INT8 execution against a $\le 120.0\text{ ms}$ pipeline promotion SLA ceiling, with $\le 15.0\text{ ms}$ p95 targeted on bare-metal GPU FP16/CUDA and $< 0.2\text{ ms}$ for standalone linear-probe head projection).

<div class="metric-grid">
    <div class="metric-card">
        <div class="metric-header">Inference Latency</div>
        <div class="metric-value">50.4 ms</div>
        <div class="metric-desc">Empirical p95 CPU INT8 latency (47.38 ms p50, 20.9 FPS) for full fused ViT-S/16 + head graph against &le; 120.0 ms pipeline SLA gate (&le; 15.0 ms GPU target; &lt; 0.2 ms standalone head).</div>
    </div>
    <div class="metric-card">
        <div class="metric-header">Conformal Coverage</div>
        <div class="metric-value">96.8%</div>
        <div class="metric-desc">Empirical calibration coverage (91.5% unseen holdout test coverage, 92.1% top-1 INT8 accuracy) satisfying finite-sample bound ($1 - \alpha = 0.95$).</div>
    </div>
    <div class="metric-card">
        <div class="metric-header">Feature Extraction</div>
        <div class="metric-value">384-D</div>
        <div class="metric-desc">Dense, L2-normalised DINOv3 ViT-S/16 representations (384-D across 1,645 cached observations), with multi-backbone support spanning 384-D to 1280-D.</div>
    </div>
    <div class="metric-card">
        <div class="metric-header">License Compliance</div>
        <div class="metric-value">100%</div>
        <div class="metric-desc">Strict open licensing (1,646 / 2,000 verified CC0, CC-BY, CC-BY-NC records; 354 violations purged) with full photographer attribution provenance.</div>
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

    subgraph S2["Feature Extraction & Drift Gate"]
        B --> C["4-Way Split Embeddings<br/>(pipelines.extract_features)"]
        C --> DD["data_drift.json<br/>(pipelines.data_drift)"]
    end

    subgraph S3["Model Training & Optimization"]
        DD --> D["head.pt<br/>(pipelines.train)"]
        D --> E["model_int8.onnx<br/>(pipelines.export_onnx)"]
    end

    subgraph S4["Epistemic Calibration & Four-Gate Promotion"]
        C -.-> F["conformal_config.yaml<br/>(pipelines.calibrate)"]
        E --> F
        C -.-> G["pareto_evaluation.json & 4-Gate Promotion<br/>(pipelines.evaluate)"]
        E --> G
        F --> G
    end

    subgraph S5["Production Serving, Telemetry & Closed Loop"]
        E --> H["FastAPI Engine & Model Passport<br/>(service.inference / passport)"]
        F --> H
        H --> I["HTMX Operator Dashboard<br/>(service.routes)"]
        I <--> R["PostgreSQL Telemetry & Redis Cache<br/>(telemetry_store / storage)"]
        R --> J["Two-Tier Drift & Airflow 3 DAGs<br/>(drift_runner / model_retraining)"]
        J -.->|Merged Feedback & Jenkins Trigger| A
    end

    click A "mlops_pipeline/ingestion_and_versioning/" "Read Ingestion Pipeline Docs"
    click B "foundations/taxonomy_and_darwincore/" "Read DarwinCore & License Docs"
    click C "mlops_pipeline/dvc_dag_explorer/" "Read Feature Extraction Docs"
    click DD "operations/drift_detection_and_telemetry/" "Read Two-Tier Drift Detection Docs"
    click D "model_engineering/training_pipeline/" "Read Head Training Docs"
    click E "model_engineering/onnx_optimization_and_latency/" "Read ONNX Export Docs"
    click F "uncertainty_and_safety/conformal_prediction_theory/" "Read Conformal Prediction Docs"
    click G "model_engineering/model_promotion_gates/" "Read Four-Gate Promotion Docs"
    click H "operations/api_reference/" "Read FastAPI Reference"
    click I "operations/deployment_runbook/" "Read Deployment Runbook"
    click R "mlops_pipeline/storage_and_databases/" "Read Storage & Cache Docs"
    click J "operations/airflow_orchestration/" "Read Airflow 3 Orchestration Docs"
```

---

## Core System Invariants

* **8-Stage Epistemic Diamond DAG Topology:** An 8-stage DVC pipeline under `src/taxon_vision/pipelines/` governed by root `params.yaml` (`ingestion` -> `audit_license` -> `extract_features` -> `data_drift` -> `train` -> `export_onnx` -> `calibrate` -> `evaluate`), enforcing strict licence compliance before S3 streaming, zero-leakage 4-way embedding splits (`train_embeddings.pt`, `val_embeddings.pt`, `cal_embeddings.pt`, `test_embeddings.pt`), pre-training representation drift verification (`reports/data_drift.json`), and post-quantisation calibration on `model_int8.onnx`.
* **Closed-Loop MLOps Lifecycle:** Unifies online FastAPI telemetry (`PostgresTelemetryStore`), two-tier representation drift detection (1D Wasserstein-1 on $L_2$ norms + multivariate RKHS $\text{MMD}_u^2$), Grafana Alertmanager webhook relays, Apache Airflow 3 DAG orchestration (`prediction_drift_monitor` and `model_retraining`), MLflow 3.x cross-stage lineage, and Four-Gate Model Registry promotion (`@Production` / `@Challenger` / `@Archived`). See the **[End-to-End Closed-Loop MLOps Overview](mlops_pipeline/closed_loop_overview.md)**.
* **Modern Foundation Backbones:** Seamlessly supports state-of-the-art vision extractors including **DINOv3**, BioCLIP-2, DINOv2, and edge-optimized MobileNetV4 architectures with frozen trunks ($\nabla_\theta = 0$).
* **Dual-Pipeline Paradigm:** Decouples weight-centric linear classification probes from data-centric Multimodal Knowledge Graph (MMKG) and Vision-RAG retrieval to prevent massive retraining cascades. **[Phase 2.5 Specification]** The MMKG and Vision-RAG retrieval path is a design specification and is not yet implemented in `src/taxon_vision/` (see [Implementation Status](operations/implementation_status.md)).
* **Mathematical Safety Invariants:** Predictions are accompanied by distribution-free **Split Conformal Prediction** sets guaranteeing a user-specified coverage bound ($1 - \alpha = 0.95$) alongside Helmholtz OOD energy gating.
* **Open Licensing & Attribution:** Adheres to DarwinCore and Creative Commons standards, strictly filtering open licenses (CC0, CC-BY, CC-BY-NC) and preserving photographer attribution.
* **Persistent ONNX Session Fast Path:** `ONNXInferenceEngine` executes a single persistent `ORT_ENABLE_ALL` session reused across requests. Pre-allocated `IOBinding` buffers for strictly zero heap allocations on the scoring path are a planned Phase 2.5 latency optimization.
