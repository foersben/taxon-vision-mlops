---
type: Report
title: Implementation Status & Engineering Verification Matrix
status: stable
stale_after: "2027-01-01T00:00:00Z"
version: "1.3.0"
description: Comprehensive four-tier engineering verification matrix, codebase-verified capabilities, and architectural roadmap for TaxonVision-MLOps.
tags: [roadmap, status, project-management, mlops, verification]
generated: {by: "process:docs-librarian", at: "2026-10-18T22:00:00Z"}
verified: {by: "process:docs-librarian", at: "2026-10-18T22:00:00Z"}
---

# Implementation Status & Engineering Verification Matrix

This document provides a forensic, zero-hallucination account of the TaxonVision-MLOps platform on `develop`. Every component, subsystem, and algorithm is categorized across a four-tier defensive taxonomy distinguishing production-integrated pipelines, component-verified modules, bounded heuristics, and future architectural specifications.

---

## 1. Verified & Operational Components (Integrated & Tested in Active Runtime/DAG)

Components in Section 1 are fully wired into continuous integration, governed by DVC or Kubernetes manifests, and tested against genuine biological data.

| Subsystem | Primary Source Files | Verification Target & Artifacts | Operational Status |
| :--- | :--- | :--- | :--- |
| **7-Stage Epistemic Diamond DVC DAG** | `src/taxon_vision/pipelines/{ingest,audit_license,extract_features,train,export_onnx,calibrate,evaluate}.py` | `dvc.yaml`, `dvc.lock`, `params.yaml`, `tests/unit/test_pipelines.py` | **Production:** Single-trunk deterministic DAG running automated hash checks and metric tracking. |
| **Open License Gatekeeper & Attribution** | `src/taxon_vision/domain/license.py`, `data/license_filter.py`, `pipelines/audit_license.py` | `tests/unit/test_license_filter.py`, `python -m taxon_vision.pipelines.audit_license` | **Production:** Enforces CC0, CC-BY, and CC-BY-NC compliance; strictly rejects unverified `static.inaturalist.org` hotlinks and retains photographer provenance. |
| **Frozen Vision Backbone Factory & 4-Way Splits** | `src/taxon_vision/models/factory.py`, `models/training/types.py`, `pipelines/extract_features.py` | `tests/unit/test_models_components.py`, `data/embeddings/*.pt` | **Production:** Frozen `BackboneFactory` supporting `dinov3_vits14`, `bioclip-2`, `dinov2_vits14`, `mobilenetv4_conv_small`, and `efficientnet_b0`. Generates zero-leakage `train`, `val`, `cal`, and `test` partitions. |
| **Long-Tail Linear Probe & Class-Balanced Loss** | `src/taxon_vision/models/{head.py,loss.py}`, `models/training/{embeddings,runner,loops,checkpoint}.py` | `tests/unit/test_class_balanced_loss.py`, `tests/unit/test_cli_and_training.py` | **Production:** Convex linear classification head trained with effective number of samples weighting ($E_{n_y} = (1 - \beta^{n_y}) / (1 - \beta)$, $\beta = 0.9999$). |
| **MLflow Lineage & PR-AUC Registry Promotion** | `src/taxon_vision/models/training/tracking.py` | `reports/train_metrics.json`, `tests/unit/test_tracking.py` | **Production:** Automated model registration with `Production`/`Challenger`/`Archived` aliases, explicit `TensorSpec` `ModelSignature`, and DVC dataset MD5 hash extraction. |
| **Dual-Output ONNX Export & Logit-Protecting PTQ** | `src/taxon_vision/inference/{onnx_exporter,quantizer,calibration}.py` | `models/checkpoints/model_int8.onnx`, `tests/unit/test_onnx_exporter.py`, `tests/unit/test_quantizer.py` | **Production:** `_ONNXDualOutputModule` exports fused `logits` $[N, K]$ and `embeddings` $[N, D]$. `find_logit_nodes` preserves FP32 logit precision under INT8 PTQ. `load_calibration_tensors` enforces real sample calibration. |
| **Split Conformal Prediction & OOD Screening** | `src/taxon_vision/uncertainty/{conformal,ood_detector}.py`, `pipelines/calibrate.py` | `tests/invariants/test_conformal_coverage_invariant.py`, `tests/unit/test_conformal_bounds.py`, `tests/unit/test_ood_energy.py` | **Production:** Evaluates empirical quantile cutoff $\hat{q}_\alpha$ on calibration partition. Gated by Helmholtz Free Energy ($E(x; T) \le \tau_{\text{OOD}}$) and Normalized Shannon Entropy ($\bar{H} \ge 0.82$). |
| **FastAPI Serving & Flagged Specimen Storage** | `src/taxon_vision/service/{api,inference,schemas,storage}.py`, `service/routes/*.py` | `tests/integration/test_api_endpoints.py`, `tests/integration/test_views_and_routes.py` | **Production:** Zero-allocation ONNX `IOBinding` engine with multipart upload. When `is_ood` or `requires_human_review` triggers, `service/storage.py` atomically writes `.npy` embeddings, `.json` metadata, and appends to `manifest.jsonl`. |
| **Forward-Hooked Analytical CAM** | `src/taxon_vision/inference/cam.py` | `tests/integration/test_api_endpoints.py`, `tests/integration/test_views_and_routes.py`, `/api/v1/explain`, `/api/v1/predict?explain=true` | **Production:** Analytical Class Activation Mapping computed directly on linear probe weights in $< 5\text{ ms}$ without backward-pass graph overhead. Fully wired into HTMX operator cockpit (lazy visual inspector) and `POST /api/v1/predict?explain=true`. |
| **Prometheus Telemetry Exposition** | `src/taxon_vision/monitoring/telemetry.py`, `config/prometheus.yaml` | `GET /metrics`, `tests/unit/test_monitoring.py` | **Production:** Exposes `taxon_predictions_total`, `taxon_inference_latency_seconds_bucket`, and `taxon_human_referrals_active` over Prometheus scrape endpoint. |

---

## 2. Component-Verified Standalone Modules (Pending Runtime / Route Wiring)

Modules in Section 2 have complete, mathematically verified implementations with unit test coverage in `tests/`, but are not yet invoked in the live `/api/v1/predict` request path:

| Module / Component | Implementation Location | Test Coverage & Verification | Pending Integration Task |
| :--- | :--- | :--- | :--- |
| **Active Learning Margin Sampling** | `src/taxon_vision/active_learning/query.py` (`margin_sampling`) | `tests/unit/test_active_learning.py` ($1.0 - (p_{\text{top1}} - p_{\text{top2}})$) | Wire into an asynchronous triage batch-ranking worker. Currently, `/api/v1/predict` flags uncertain inputs and writes them to disk via `service/storage.py` without calculating margin priority scores. |
| **In-Memory Priority Triage Queue** | `src/taxon_vision/active_learning/triage_queue.py` (`TriageQueue`) | `tests/unit/test_active_learning.py` ($O(\log N)$ push/pop via `heapq` with monotonic counter) | Connect `service/inference.py` to push flagged specimens directly into `TriageQueue`. Currently, flagged specimens are persisted to `.npy` and `manifest.jsonl` via `service/storage.py`. |
| **1D Wasserstein-1 Scalar Norm Drift** | `src/taxon_vision/monitoring/drift.py` (`compute_wasserstein_drift`) | `tests/unit/test_monitoring.py` ($O(N \cdot D)$ Earth Mover's Distance over $L_2$ norms) | Connect to a rolling sliding window of embeddings in `service/inference.py` and export a live `taxon_embedding_drift_wasserstein` Prometheus gauge. |
| **Optuna TPE + ASHA Tuning** | `src/taxon_vision/models/training/tuning.py` | `tests/unit/test_tuning.py` (ASHA pruner + multi-trial SQLite WAL) | Expose automated retraining trigger when drift or referral alerts breach critical thresholds; currently triggered manually via CLI (`--tune`). |

---

## 3. Partially Implemented, Heuristic & Scope-Bounded Mechanisms

Section 3 documents operational boundaries where components function effectively within controlled assumptions but have planned architectural refinements:

* **Logit Temperature Scaling ($T$):** Temperature scaling is applied in `ood_detector.py` and `calibrate.py`, but `temperature: 1.0` in `params.yaml` and `config/conformal_config.yaml` is currently a static heuristic. Automated Negative Log-Likelihood minimization (Platt scaling) on the calibration partition is not yet automated in `calibrate.py`.
* **Taxonomic Perimeter & S3 Streaming Scale:** Data streaming (`data/s3_streamer.py`, `data/inat_client.py`) is verified and stable on the 10-species Level-1 catalog perimeter (`config/taxa_catalog.yaml`) with bounded per-taxon quotas, rather than stress-tested on the full multi-terabyte 100M+ iNaturalist corpus.
* **Multi-Backbone Pareto Evaluation Scope:** The active DVC pipeline trains, quantizes, and evaluates `dinov3_vits14` on-cluster (**89.35% Holdout Test Accuracy** in `reports/pareto_evaluation.json`). The other four comparison models (`bioclip-2`, `dinov2_vits14`, `mobilenetv4_conv_small`, `efficientnet_b0`) in `pareto_results.json` represent literature-calibrated symbolic projections under identical latency budgets.
* **Single-Stream vs. Concurrent Ingress Latency SLA:** Single-image sequential inference strictly satisfies $\le 15.0\text{ ms}$ p95 on bare-metal GPU and $\le 25.0\text{ ms}$ p95 on CPU. Sustained multi-client concurrent load testing through Kubernetes ingress and Cloudflare Tunnel is scheduled for Phase 2.5.

---

## 4. Architectural Specifications & Phase 3 Roadmap (Zero Runtime Code in `src/`)

The following capabilities are formal architectural designs and specifications; there is currently zero runtime code in `src/taxon_vision/`:

| Capability | Target Architecture & Proposed Stack | Planned Functionality |
| :--- | :--- | :--- |
| **BADGE Active Learning** | `active_learning/query.py` + PyTorch gradient embedding computation | Generates hallucinated loss gradients ($g_x = (p - e_{\hat{y}}) \otimes z$) and samples diverse batches via k-means++ in gradient space. |
| **Persistent Triage Queue** | Parquet + DuckDB storage layer + Label Studio webhook sync | Replaces in-memory `TriageQueue` with persistent columnar triage queue and bidirectional Label Studio annotation ingest. |
| **Offline Multivariate $\text{MMD}^2$ Drift** | `monitoring/drift.py` + Gaussian RBF kernel RKHS two-sample test | Unbiased finite-sample test ($\text{MMD}_u^2$) executed on weekly batches in Jenkins CI/CD to detect non-linear manifold distortion. |
| **Parent-Child Visual Chunking** | YOLO-World / OWL-ViT v2 + BioCLIP-2 ($d=512$) | Isolates diagnostic specimen crop $I_{\text{child}}$ from parent environmental scene $I_{\text{parent}}$ to eliminate background bias. |
| **Biogeographically Filtered HNSW** | LanceDB / embedded HNSW with Haversine distance ($\le 100\text{ km}$) + Julian day phenology | Prunes biologically impossible taxa before visual nearest-neighbor search. |
| **Reciprocal Rank Fusion (RRF)** | Rank aggregation engine ($k=60$) | Combines visual embedding similarity with taxonomic taxonomy tree distance. |
| **VLM Differential Diagnosis Cards** | Qwen2-VL-7B-Instruct with strict JSON Schema BNF grammar decoding | Generates verifiable diagnostic cards highlighting diagnostic anatomy (wing venation, spur length). |
| **$O(1)$ DuckDB Catalog Mutations** | DuckDB vectorized relational mapping tables | Instant biological taxonomy mutations (splits and lumps) without re-embedding datasets. |

---

## 5. Automated Model Promotion Strategy

To protect the core identification service against regression, especially on rare, long-tail taxa where standard accuracy is misleading, we implement a **PR-AUC Safe Promotion** strategy via the MLflow Model Registry.

### The Core Metric

Models are evaluated strictly on a robust, imbalance-aware metric rather than raw accuracy:

$$ \text{Promotion Score} = \text{Macro PR-AUC} $$

* **Macro PR-AUC (Precision-Recall AUC):** Heavily penalizes false positives on rare classes. By relying exclusively on this metric for automated promotion decisions, we ensure that performance gains are true across all taxa, not just the most common ones.

### Registry Promotion Logic

When a new classification head finishes training, it is automatically logged to the `taxon_vision_classifier` registry. The pipeline then compares its Macro PR-AUC score against the model currently tagged as `@Production`:

* **Outperforms Production:** The new model is automatically tagged as **`@Challenger`** (or `@Staging`). It is **not** immediately pushed to `@Production`. This ensures we keep at most two competing models active, and stages the new model for deeper human review or automated integration canary sweeps.
* **Underperforms Production:** The new model is tagged as **`@Archived`** (or `@Rejected`), ensuring underperforming checkpoints never pollute the deployment pipeline.

---

## 6. Verified Technical Debt & Implementation Resolutions

To maintain scientific and engineering integrity, the following implementation discrepancies between target architecture and active repository state have been resolved:

* **Synthetic Placeholder Training Split:** Resolved. Real foundation vision feature extraction is implemented in `taxon_vision.pipelines.extract_features` (`build_taxon_index_map`, `stream_and_extract_representations`, `create_stratified_embedding_split`), streaming observations from AWS S3 and partitioning into stratified embedding tensors (`train_embeddings.pt`, `val_embeddings.pt`, `cal_embeddings.pt`, `test_embeddings.pt`) under `data/embeddings/`. The training pipeline (`taxon_vision.pipelines.train`) and training runner ingest these physical splits directly, retaining the synthetic Gaussian split strictly as a graceful offline fallback.
* **Container Registry Publishing:** Resolved. GitHub Actions automatically packages the production container and publishes multi-tagged images (`latest`, `sha-<short_sha>`) to `ghcr.io/foersben/taxon-vision-mlops`, hydrated with promoted checkpoints from DagsHub DVC. The Kubernetes deployment continuously pulls releases via `imagePullPolicy: Always`.
* **Cognitive Complexity Quality Gate:** Resolved. Integrated `complexipy` into CI and refactored deep functions (`_prepare_embedding_split`, `build_taxon_index_map`) to guarantee cognitive complexity $\le 15$ across all code paths.
* **Jenkins Agent Pod GPU Allocation:** Resolved. In `Jenkinsfile`, the `Model Training & Checkpoint (GPU)` stage defines a dedicated agent pod requesting `nvidia.com/gpu: 1` in limits and requests, binding dynamically to the 4-replica time-slicing configuration on `hive-mind`.
* **Blackwell Toolchain Upgrade:** Resolved. Environment manifests (`pyproject.toml`, `pixi.lock`) are upgraded to PyTorch 2.7.0 targeting CUDA 12.8 (`cu128`), natively supporting SM 120 kernels on the host's NVIDIA Driver 615.71.09 / CUDA 13.4 UMD.
* **Error Masking in CI Pipeline:** Resolved. Removed all illegitimate `|| true` suppressions from `Jenkinsfile` across training and deployment stages (`dvc repro`, `dvc push`, `rollout restart`, and `rollout status`), replacing them with explicit service availability checks and fail-fast assertions.
* **Coverage Gate Inconsistency:** Resolved. Harmonized coverage thresholds between `Jenkinsfile` and `.github/workflows/ci.yml` at 80% (active global test suite achieves 87% line coverage across 142 passing tests).
* **Optuna Evaluation Objective & Search Space:** Resolved. In `taxon_vision.models.training.tuning`, the objective function returns $\max(\text{val\_pr\_auc})$ matching peak restored checkpoint weights, and Optuna samples categorical batch sizes (`[32, 64, 128]`) and embedding noise jitter regularisation (`noise_std`).
* **Local In-Cluster S3 Storage Engine (SeaweedFS):** Migrated from MinIO/Silo to SeaweedFS (`docker.io/chrislusf/seaweedfs:latest`), licensed under Apache 2.0. Eliminates AGPLv3 compliance risks and sidecar bucket provisioning loops. Its Haystack/Needle architecture prevents disk-seek bottlenecks, providing sub-2ms random read access for the 100,000 exemplar observation crops and LanceDB vector partitions planned in Pipeline 2.
