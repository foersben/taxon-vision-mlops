---
type: Report
title: Implementation Status & Roadmap
status: stable
stale_after: "2027-01-01T00:00:00Z"
version: "1.2"
description: Three-tier implementation status, code-verified capabilities, and architectural roadmap for TaxonVision-MLOps.
tags: [roadmap, status, project-management, mlops, verification]
generated: {by: "process:docs-librarian", at: "2026-10-18T18:00:00Z"}
verified: {by: "process:docs-librarian", at: "2026-10-18T18:00:00Z"}
---

# Implementation Status & Roadmap

This document records the exact implementation state of the TaxonVision-MLOps platform on `develop`, categorizing platform capabilities across three rigorous operational tiers: production-verified pipelines, implemented standalone modules, and planned Phase 3 architectural targets.

---

## 1. Verified Implementation Matrix

### Tier A: Production-Complete & Verified on `develop` (Phase 1 & Core Phase 2)

All components in Tier A are fully integrated into continuous testing, governed by DVC or Kubernetes manifests, and tested against genuine biological data.

| Subsystem | Primary Source Files | Verification Target & Artifacts | Operational Status |
| :--- | :--- | :--- | :--- |
| **7-Stage Epistemic Diamond DVC DAG** | `src/taxon_vision/pipelines/{ingest,audit_license,extract_features,train,export_onnx,calibrate,evaluate}.py` | `dvc.yaml`, `dvc.lock`, `params.yaml`, `tests/unit/test_pipelines.py` | **Production:** Single-trunk deterministic DAG running automated hash checks and metric tracking. |
| **Three-Tier S3 Topology & Zero-Leakage Splits** | `src/taxon_vision/data/{s3_streamer,license_filter}.py` | `deploy/k8s/mlops-storage-and-optuna.yaml`, `data/embeddings/*.pt` | **Production:** Anonymous Arrow streaming from AWS Open Data, SeaweedFS in-cluster cache (`s3://taxon-dvc-cache`), and DagsHub S3 model release storage. 4-way stratified partitions (`train`, `val`, `cal`, `test`). |
| **Model Factory & Imbalance Loss** | `src/taxon_vision/models/{factory,head,loss}.py` | `tests/unit/test_models_components.py`, `models/checkpoints/head.pt` | **Production:** Frozen `BackboneFactory` supporting `dinov3_vits14`, `bioclip-2`, `dinov2_vits14`, `mobilenetv4_conv_small`, and `efficientnet_b0`. Optimized via `ClassBalancedLoss` ($\beta \in [0.9, 0.9999]$). |
| **Persistent Bayesian Telemetry & HPO** | `src/taxon_vision/models/training/{tuning,runner,loops,checkpoint}.py` | `tests/unit/test_tuning.py`, persistent SQLite WAL on k3s PVC (`NodePort 30081`) | **Production:** Optuna TPE with ASHA early stopping pruning underperforming trials on cached embeddings without GPU thrashing. |
| **MLflow Model Registry & Lineage** | `src/taxon_vision/models/training/tracking.py` | `reports/train_metrics.json`, DagsHub MLflow remote | **Production:** Automated model registration with `Production`/`Challenger`/`Archived` aliases, explicit `TensorSpec` `ModelSignature`, and DVC dataset MD5 hash extraction. |
| **Dual-Output ONNX & Static INT8 PTQ** | `src/taxon_vision/inference/{onnx_exporter,quantizer,calibration}.py` | `models/checkpoints/model_int8.onnx`, `tests/unit/test_quantizer.py` | **Production:** Exports fused `logits` $[N, K]$ and `embeddings` $[N, D]$ graph. Static INT8 QDQ PTQ calibrated via `TaxonCalibrationDataReader` strictly banning synthetic noise. |
| **Four Calibration Layers** | `src/taxon_vision/inference/{calibration,head_projector}.py`, `pipelines/calibrate.py` | `config/conformal_config.yaml`, `reports/conformal_metrics.json` | **Production:** Layer 1 (INT8 activation range), Layer 2 (Temperature $T$), Layer 3 (Split Conformal quantile $\hat{q}_\alpha$), Layer 4 (Thermodynamic energy cutoff $\tau_{\text{OOD}}$). |
| **Thermodynamic OOD & Shannon Clutter Gating** | `src/taxon_vision/uncertainty/{ood_detector,conformal}.py` | `tests/unit/test_ood_energy.py`, `tests/unit/test_conformal_bounds.py` | **Production:** Gated by Helmholtz Free Energy ($E(x; T) \le \tau_{\text{OOD}}$) and Normalized Shannon Entropy ($\bar{H} \ge 0.82$) to reject non-biological inputs and camera-trap clutter. |
| **FastAPI Service & HTMX Operator Dashboard** | `src/taxon_vision/service/{api,inference,schemas}.py`, `service/routes/views.py` | `tests/integration/test_api_endpoints.py`, `tests/integration/test_views_and_routes.py` | **Production:** Zero-allocation ONNX `IOBinding` engine, multipart image upload, server-rendered HTMX verification portal (`/`, `/ui`, `/triage`), and OpenAPI schema. |
| **Forward-Hooked Linear CAM** | `src/taxon_vision/inference/cam.py` | `tests/unit/test_cam.py`, `/api/v1/explain` | **Production:** Analytical Class Activation Mapping computed directly on linear probe weights in $< 5\text{ ms}$ without backward-pass graph overhead. |
| **Prometheus Telemetry & Online Drift** | `src/taxon_vision/monitoring/{telemetry,drift}.py` | `/metrics`, `tests/unit/test_monitoring.py` | **Production:** Prometheus 5s scrape counters and histograms (`taxon_predictions_total`, latency histograms, conformal set sizes) and online 1D Wasserstein-1 scalar norm drift detection. |
| **Bare-Metal k3s & Jenkins CI/CD** | `deploy/k8s/*.yaml`, `Jenkinsfile`, `docker-compose.local.yaml` | `https://taxon.benjamin-foerster.eu/`, host `hive-mind` | **Production:** 4-replica NVIDIA GPU time-slicing on RTX 5070 Ti, non-root multi-stage Docker container (`ghcr.io/foersben/taxon-vision-mlops:latest`), and automated rolling rollouts. |

---

### Tier B: Implemented & Unit-Tested Standalone, Pending Live Route Wiring

Components in Tier B have verified algorithmic implementations and high unit test coverage in `tests/`, but operate offline or are scheduled for subsequent route integration:

| Module / Component | Implementation Location | Test Coverage & Verification | Pending Integration Task |
| :--- | :--- | :--- | :--- |
| **BADGE Active Learning Query** | `src/taxon_vision/active_learning/query.py` (`compute_badge_embeddings`, `badge_query_sampling`) | `tests/unit/test_active_learning.py` (tested with mock embedding and gradient topologies) | Wire live triage sampling endpoint `/api/v1/active-learning/query` into the operator HTMX UI. Currently, `/api/v1/predict` marks `requires_human_review = True` and increments Prometheus metrics without triggering batch BADGE ranking. |
| **Parquet Triage Queue Manager** | `src/taxon_vision/active_learning/triage_queue.py` (`TriageQueueManager`) | `tests/unit/test_active_learning.py` (tested with atomic schema mutations and DuckDB queries) | Connect `service/inference.py` triage branching to asynchronously write ambiguous specimens ($\lvert \mathcal{C}(x) \rvert \ge 4$) directly to `data/triage/triage_queue.parquet`. |
| **Offline Full-Manifold $\text{MMD}^2$ Drift** | `src/taxon_vision/monitoring/drift.py` (`compute_mmd_feature_drift`) | `tests/unit/test_monitoring.py` (verified with RBF Gaussian kernel embeddings) | Schedule automated recurring cron evaluations against baseline validation tensors in Jenkins CI/CD; online serving currently uses fast 1D Wasserstein-1 scalar norm drift. |

---

### Tier C: Phase 3 Architectural Targets (Vision-RAG & MMKG Roadmap)

Tier C encapsulates the future multimodal retrieval roadmap detailed in Chapter 5 of the Technical Strategy Report:

| Architecture Capability | Specification & Target Components | Planned Technologies | Roadmap Objective |
| :--- | :--- | :--- | :--- |
| **Parent-Child Visual Chunking** | Open-vocabulary zero-shot organism detector isolates specimen crop $I_{\text{child}}$ from parent environmental scene $I_{\text{parent}}$. | YOLO-World / OWL-ViT v2 + BioCLIP-2 ($d=512$) | Eliminates background host plant and soil bias on Batesian mimicry pairs. |
| **Biogeographically Filtered Vector Retrieval** | Single-stage vector search combining geographical radius and temporal emergence filters. | LanceDB / embedded HNSW with Haversine spherical distance ($\le 100\text{ km}$) and Julian day phenology masks | Prunes biologically impossible taxa before nearest-neighbor evaluation. |
| **Reciprocal Rank Fusion (RRF)** | Rank-based aggregation combining visual similarity and taxonomic taxonomy graph distance. | RRF ($k=60$) rank aggregation engine | Combines specimen morphological embeddings with museum exemplar metadata. |
| **Grammar-Constrained VLM Differential Diagnosis** | Structured diagnosis card comparing sister taxa using morphological keys. | Qwen2-VL-7B-Instruct with strict JSON Schema BNF grammar decoding | Generates verifiable diagnostic cards highlighting diagnostic anatomy (wing venation, spur length). |
| **$O(1)$ DuckDB Catalog Mutations** | Instant biological taxonomy mutations (splits and lumps) without re-embedding datasets. | DuckDB vectorized relational mapping tables | Guarantees zero re-computation when taxonomic checklists split or lump species. |
| **Label Studio Bidirectional Sync** | Synchronizing human expert corrections from DagsHub Label Studio into training partitions. | DagsHub REST API + DVC pipeline webhooks | Closes the active learning loop from human taxonomist verification to automated retraining. |

---

## 2. Automated Model Promotion Strategy

To protect the core identification service against regression-especially on rare, long-tail taxa where standard accuracy is misleading-we implement a **PR-AUC Safe Promotion** strategy via the MLflow Model Registry.

### The Core Metric

Models are evaluated strictly on a robust, imbalance-aware metric rather than raw accuracy:

$$ \text{Promotion Score} = \text{Macro PR-AUC} $$

* **Macro PR-AUC (Precision-Recall AUC):** Heavily penalizes false positives on rare classes. By relying exclusively on this metric for automated promotion decisions, we ensure that performance gains are true across all taxa, not just the most common ones.

### Registry Promotion Logic

When a new classification head finishes training, it is automatically logged to the `taxon_vision_classifier` registry. The pipeline then compares its Macro PR-AUC score against the model currently tagged as `@Production`:

* **Outperforms Production:** The new model is automatically tagged as **`@Challenger`** (or `@Staging`). It is **not** immediately pushed to `@Production`. This ensures we keep at most two competing models active, and stages the new model for deeper human review or automated integration canary sweeps.
* **Underperforms Production:** The new model is tagged as **`@Archived`** (or `@Rejected`), ensuring underperforming checkpoints never pollute the deployment pipeline.

---

## 3. Verified Technical Debt & Implementation Resolutions

To maintain scientific and engineering integrity, the following implementation discrepancies between target architecture and active repository state have been resolved:

* **Synthetic Placeholder Training Split:** Resolved. Real foundation vision feature extraction is implemented in `taxon_vision.pipelines.extract_features` (`build_taxon_index_map`, `stream_and_extract_representations`, `create_stratified_embedding_split`), streaming observations from AWS S3 and partitioning into stratified embedding tensors (`train_embeddings.pt`, `val_embeddings.pt`, `cal_embeddings.pt`, `test_embeddings.pt`) under `data/embeddings/`. The training pipeline (`taxon_vision.pipelines.train`) and training runner ingest these physical splits directly, retaining the synthetic Gaussian split strictly as a graceful offline fallback.
* **Container Registry Publishing:** Resolved. GitHub Actions automatically packages the production container and publishes multi-tagged images (`latest`, `sha-<short_sha>`) to `ghcr.io/foersben/taxon-vision-mlops`, hydrated with promoted checkpoints from DagsHub DVC. The Kubernetes deployment continuously pulls releases via `imagePullPolicy: Always`.
* **Cognitive Complexity Quality Gate:** Resolved. Integrated `complexipy` into CI and refactored deep functions (`_prepare_embedding_split`, `build_taxon_index_map`) to guarantee cognitive complexity $\le 15$ across all code paths.
* **Jenkins Agent Pod GPU Allocation:** Resolved. In `Jenkinsfile`, the `Model Training & Checkpoint (GPU)` stage defines a dedicated agent pod requesting `nvidia.com/gpu: 1` in limits and requests, binding dynamically to the 4-replica time-slicing configuration on `hive-mind`.
* **Blackwell Toolchain Upgrade:** Resolved. Environment manifests (`pyproject.toml`, `pixi.lock`) are upgraded to PyTorch 2.7.0 targeting CUDA 12.8 (`cu128`), natively supporting SM 120 kernels on the host's NVIDIA Driver 615.71.09 / CUDA 13.4 UMD.
* **Error Masking in CI Pipeline:** Resolved. Removed all illegitimate `|| true` suppressions from `Jenkinsfile` across training and deployment stages (`dvc repro`, `dvc push`, `rollout restart`, and `rollout status`), replacing them with explicit service availability checks and fail-fast assertions.
* **Coverage Gate Inconsistency:** Resolved. Harmonized coverage thresholds between `Jenkinsfile` and `.github/workflows/ci.yml` at 80% (active global test suite achieves 89% line coverage across 124 passing tests).
* **Optuna Evaluation Objective & Search Space:** Resolved. In `taxon_vision.models.training.tuning`, the objective function returns $\max(\text{val\_pr\_auc})$ matching peak restored checkpoint weights, and Optuna samples categorical batch sizes (`[32, 64, 128]`) and embedding noise jitter regularisation (`noise_std`).
* **Local In-Cluster S3 Storage Engine (SeaweedFS):** Migrated from MinIO/Silo to SeaweedFS (`docker.io/chrislusf/seaweedfs:latest`), licensed under Apache 2.0. Eliminates AGPLv3 compliance risks and sidecar bucket provisioning loops. Its Haystack/Needle architecture prevents disk-seek bottlenecks, providing sub-2ms random read access for the 100,000 exemplar observation crops and LanceDB vector partitions planned in Pipeline 2.
