---
type: Report
title: Implementation Status & Engineering Verification Matrix
status: stable
stale_after: "2027-01-01T00:00:00Z"
version: "1.4.0"
description: Comprehensive four-tier engineering verification matrix, codebase-verified capabilities, and architectural roadmap for TaxonVision-MLOps.
tags: [roadmap, status, project-management, mlops, verification]
generated: {by: "process:docs-librarian", at: "2026-11-10T16:00:00Z"}
verified: {by: "process:docs-librarian", at: "2026-11-10T16:00:00Z"}
---

# Implementation Status & Engineering Verification Matrix

This document provides a forensic, zero-hallucination account of the TaxonVision-MLOps platform on `develop`. Every component, subsystem, and algorithm is categorized across a four-tier defensive taxonomy distinguishing production-integrated pipelines, component-verified modules, bounded heuristics, and future architectural specifications.

---

## 1. Verified & Operational Components (Integrated & Tested in Active Runtime/DAG)

Components in Section 1 are fully wired into continuous integration, governed by DVC or Kubernetes manifests, and tested against genuine biological data.

| Subsystem | Primary Source Files | Verification Target & Artifacts | Operational Status |
| :--- | :--- | :--- | :--- |
| **8-Stage Epistemic Diamond DVC DAG** | `src/taxon_vision/pipelines/{ingest,audit_license,extract_features,data_drift,train,export_onnx,calibrate,evaluate}.py` | `dvc.yaml`, `dvc.lock`, `params.yaml`, `tests/unit/test_pipelines.py`, `tests/unit/test_data_drift_pipeline.py` | **Production:** Single-trunk deterministic 8-stage DAG with frozen ingestion guard (`frozen: true`) and pre-training representation drift verification (`reports/data_drift.json`). |
| **Open License Gatekeeper & Attribution** | `src/taxon_vision/domain/license.py`, `data/license_filter.py`, `pipelines/audit_license.py` | `tests/unit/test_license_filter.py`, `python -m taxon_vision.pipelines.audit_license` | **Production:** Enforces CC0, CC-BY, and CC-BY-NC compliance; strictly rejects unverified `static.inaturalist.org` hotlinks and retains photographer provenance. |
| **Active Learning Feedback Ingestion** | `src/taxon_vision/active_learning/feedback.py` | `tests/unit/test_feedback_merge.py` | **Production:** Parses Label Studio JSON and flagged triage annotations, maps verified scientific names to `taxon_id`, and deduplicates/merges records into `raw_manifest.parquet`. |
| **Frozen Vision Backbone Factory & 4-Way Splits** | `src/taxon_vision/models/factory.py`, `models/training/types.py`, `pipelines/extract_features.py` | `tests/unit/test_models_components.py`, `data/embeddings/*.pt` | **Production:** Frozen `BackboneFactory` supporting `dinov3_vits14`, `bioclip-2`, `dinov2_vits14`, `mobilenetv4_conv_small`, and `efficientnet_b0`. Generates zero-leakage `train`, `val`, `cal`, and `test` partitions. |
| **Long-Tail Linear Probe & Class-Balanced Loss** | `src/taxon_vision/models/{head.py,loss.py}`, `models/training/{embeddings,runner,loops,checkpoint}.py` | `tests/unit/test_class_balanced_loss.py`, `tests/unit/test_cli_and_training.py` | **Production:** Convex linear classification head trained with effective number of samples weighting ($E_{n_y} = (1 - \beta^{n_y}) / (1 - \beta)$, $\beta = 0.9999$). |
| **MLflow 3.x Lineage & Four-Gate Model Promotion** | `src/taxon_vision/models/training/{tracking,promotion,visualizations}.py`, `inference/mlflow_model.py` | `reports/train_metrics.json`, `tests/unit/test_tracking.py` | **Production:** Cross-stage MLflow run continuity (`resume_parent_mlflow_run`), 5-dataset `mlflow.data` lineage, `TaxonVisionPyFuncModel` packaging, and Four-Gate promotion enforcing INT8 drop ($\le 3\%$), Macro PR-AUC, p95 Latency ceilings, and MMD² calibration exchangeability. |
| **Two-Tier Representation Drift Engine & Telemetry Store** | `src/taxon_vision/monitoring/{drift,drift_runner}.py`, `pipelines/calibrate_drift.py`, `service/telemetry_store.py` | `tests/unit/test_monitoring.py`, `tests/unit/test_telemetry_store.py`, `tests/unit/test_drift_runner.py` | **Production:** Evaluates Wasserstein-1 norm drift, bootstrap-calibrated thresholds (`calibrate_drift.py`), multivariate MMD² permutation tests, and 8-D PCA Evidently 0.7 reports over sliding production windows, persisting records in PostgreSQL and exposing Prometheus metrics. |
| **Airflow 3 LocalExecutor DAG Orchestration** | `airflow/dags/{prediction_drift_monitor,model_retraining}.py`, `scripts/maintenance/visualize_airflow_dag.py` | `tests/unit/test_airflow_dags.py`, `just airflow-dag` | **Production:** Hourly automated drift monitor DAG and event-driven GPU retraining orchestrator integrated with Jenkins and MLflow promotion reports. |
| **Grafana Dashboards, Alerts & Retraining Relay** | `deploy/grafana/provisioning/`, `service/routes/retrain.py` | `tests/integration/test_retrain_routes.py` | **Production:** Provisioned 6-panel dashboard, Alertmanager rules with 6-hour repeat interval, and authenticated FastAPI webhook relay triggering Airflow DAGs. |
| **Active Model Passport & Operator Cockpit** | `src/taxon_vision/service/passport.py`, `service/routes/views.py`, `service/templates/index.html` | `tests/unit/test_passport.py`, `GET /` | **Production:** Real-time frosted-glass model passport pill strip on `GET /` cached in-memory with 60s TTL and non-blocking AnyIO threadpool execution (`load_model_passport`). |
| **Dual-Output ONNX Export & Logit-Protecting PTQ** | `src/taxon_vision/inference/{onnx_exporter,quantizer,calibration}.py` | `models/checkpoints/model_int8.onnx`, `tests/unit/test_onnx_exporter.py`, `tests/unit/test_quantizer.py` | **Production:** `_ONNXDualOutputModule` exports fused `logits` $[N, K]$ and `embeddings` $[N, D]$. `find_logit_nodes` preserves FP32 logit precision under INT8 PTQ. `load_calibration_tensors` enforces real sample calibration. |
| **Split Conformal Prediction & OOD Screening** | `src/taxon_vision/uncertainty/{conformal,ood_detector}.py`, `pipelines/calibrate.py` | `tests/invariants/test_conformal_coverage_invariant.py`, `tests/unit/test_conformal_bounds.py`, `tests/unit/test_ood_energy.py` | **Production:** Evaluates empirical quantile cutoff $\hat{q}_\alpha$ on calibration partition. Gated by Helmholtz Free Energy ($E(x; T) \le \tau_{\text{OOD}}$) and Normalized Shannon Entropy ($\bar{H} \ge 0.82$). |
| **FastAPI Serving & Flagged Specimen Storage** | `src/taxon_vision/service/{api,inference,schemas,storage}.py`, `service/routes/*.py` | `tests/integration/test_api_endpoints.py`, `tests/integration/test_views_and_routes.py` | **Production:** Persistent ONNX Runtime session with contiguous pre-allocated C-order buffers (`IOBinding` scheduled for Phase 2.5) and multipart upload. When `is_ood` or `requires_human_review` triggers, `service/storage.py` atomically writes `.npy` embeddings, `.json` metadata, and appends to `manifest.jsonl`. |
| **Forward-Hooked Analytical CAM** | `src/taxon_vision/inference/cam.py` | `tests/integration/test_api_endpoints.py`, `tests/integration/test_views_and_routes.py`, `/api/v1/explain`, `/api/v1/predict?explain=true` | **Production:** Analytical Class Activation Mapping computed directly on linear probe weights in $< 5\text{ ms}$ without backward-pass graph overhead. Fully wired into HTMX operator cockpit (lazy visual inspector) and `POST /api/v1/predict?explain=true`. |
| **Prometheus Telemetry Exposition** | `src/taxon_vision/monitoring/telemetry.py`, `config/prometheus.yaml` | `GET /metrics`, `tests/unit/test_monitoring.py` | **Production:** Exposes `taxon_predictions_total`, `taxon_inference_latency_seconds_bucket`, and `taxon_human_referrals_active` over Prometheus scrape endpoint. |
| **Secret Governance & Automated Vault Provisioning** | `scripts/deploy/provision_k8s_secrets.sh`, `deploy/k8s/base/api-deployment.yaml`, `.gitguardian.yaml` | `tests/unit/test_config.py` | **Production:** Strict separation of code and state via `secretKeyRef` sourcing from `/etc/taxon-vision/production.env`. Automated secret masking, idempotent provisioning, and GitGuardian secret detection policies. |
| **Multi-Tier Kustomize Staging & Production Overlays** | `deploy/k8s/{base,overlays/staging,overlays/production}`, `Jenkinsfile`, `.github/workflows/ci.yml` | `scripts/ci/verify_deployment.sh`, Jenkins Build #87 | **Production:** Branch-aware deployment routing `develop` to `taxon-vision-staging` and `main` to `taxon-vision`. CPU-only staging discipline (<15ms INT8 ONNX execution) eliminating GPU starvation deadlocks. |
| **Synthetic Traffic Generation & Live Drift Validation** | `scripts/maintenance/simulate_traffic.py`, `scripts/maintenance/smoke_test_mlflow.sh` | Live telemetry verification on staging | **Production:** High-throughput async synthetic client supporting Gaussian blur and exposure corruption modes to validate Prometheus alerts and retraining relays. |
| **Dual Grafana Dashboards & Loki Log Aggregation** | `deploy/grafana/provisioning/dashboards/json/`, `config/{loki,promtail}.yaml`, `deploy/prometheus/rules.yaml` | Ports 3000, 3100, 9080, 9090 | **Production:** Production Overview (10 panels) and Infrastructure Overview (14 panels, RSS slope, DDR5, RTX 5070 Ti VRAM). Promtail log shipping to Loki TSDB v13 with LogQL streams. Alertmanager rules with automated retrain webhook dispatch. |

---

## 2. Component-Verified Standalone Modules (Pending Runtime / Route Wiring)

Modules in Section 2 have complete, mathematically verified implementations with unit test coverage in `tests/`, but are not yet invoked in the live `/api/v1/predict` request path:

| Module / Component | Implementation Location | Test Coverage & Verification | Pending Integration Task |
| :--- | :--- | :--- | :--- |
| **Active Learning Margin Sampling** | `src/taxon_vision/active_learning/query.py` (`margin_sampling`) | `tests/unit/test_active_learning.py` ($1.0 - (p_{\text{top1}} - p_{\text{top2}})$) | Wire into an asynchronous triage batch-ranking worker. Currently, `/api/v1/predict` flags uncertain inputs and writes them to disk via `service/storage.py` without calculating margin priority scores. |
| **In-Memory Priority Triage Queue** | `src/taxon_vision/active_learning/triage_queue.py` (`TriageQueue`) | `tests/unit/test_active_learning.py` ($O(\log N)$ push/pop via `heapq` with monotonic counter) | Connect `service/inference.py` to push flagged specimens directly into `TriageQueue`. Currently, flagged specimens are persisted to `.npy` and `manifest.jsonl` via `service/storage.py`. |
| **Optuna TPE + ASHA Tuning** | `src/taxon_vision/models/training/tuning.py` | `tests/unit/test_tuning.py` (ASHA pruner + multi-trial SQLite WAL) | Expose automated retraining trigger when drift or referral alerts breach critical thresholds; currently triggered manually via CLI (`--tune`). |

---

## 3. Partially Implemented, Heuristic & Scope-Bounded Mechanisms

Section 3 documents operational boundaries where components function effectively within controlled assumptions but have planned architectural refinements:

* **Logit Temperature Scaling ($T$):** Temperature scaling is applied in `ood_detector.py` and `calibrate.py`, but `temperature: 1.0` in `params.yaml` and `config/conformal_config.yaml` is currently a static heuristic. Automated Negative Log-Likelihood minimization (Platt scaling) on the calibration partition is not yet automated in `calibrate.py`.
* **Taxonomic Perimeter & S3 Streaming Scale:** Data streaming (`data/s3_streamer.py`, `data/inat_client.py`) is verified and stable on the 10-species Level-1 catalog perimeter (`config/taxa_catalog.yaml`) with bounded per-taxon quotas, rather than stress-tested on the full multi-terabyte 100M+ iNaturalist corpus.
* **Multi-Backbone Pareto Evaluation Scope:** The active DVC pipeline trains, quantises, and evaluates `dinov3_vits14` (`vit_small_patch16_dinov3.lvd1689m`) on-cluster (**92.07% INT8 Holdout Test Accuracy** and **50.38 ms CPU INT8 p95 latency** in `reports/pareto_evaluation.json`). The other four comparison models (`bioclip-2`, `dinov2_vits14`, `mobilenetv4_conv_small`, `efficientnet_b0`) in `pareto_results.json` represent literature-calibrated symbolic projections under identical latency budgets.
* **Single-Stream vs. Concurrent Ingress Latency SLO:** Single-image sequential inference achieves `47.38 ms` p50 / `50.38 ms` p95 on CPU INT8 for the full fused ViT-S/16 + head graph (within the $\le 120.0\text{ ms}$ Stage 7 CPU/CI SLA gate; $< 0.2\text{ ms}$ for standalone head projection) and targets $\le 15.0\text{ ms}$ p95 on bare-metal GPU. Sustained multi-client concurrent load testing through Kubernetes ingress and Cloudflare Tunnel is scheduled for Phase 2.5.

---

## 4. Architectural Specifications & Phase 3 Roadmap (Zero Runtime Code in `src/`)

The following capabilities are formal architectural designs and specifications; there is currently zero runtime code in `src/taxon_vision/`:

| Capability | Target Architecture & Proposed Stack | Planned Functionality |
| :--- | :--- | :--- |
| **BADGE Active Learning** | `active_learning/query.py` + PyTorch gradient embedding computation | Generates hallucinated loss gradients ($g_x = (p - e_{\hat{y}}) \otimes z$) and samples diverse batches via k-means++ in gradient space. |
| **Persistent Triage Queue** | Parquet + DuckDB storage layer + Label Studio webhook sync | Replaces in-memory `TriageQueue` with persistent columnar triage queue and bidirectional Label Studio annotation ingest. |
| **Parent-Child Visual Chunking** | YOLO-World / OWL-ViT v2 + BioCLIP-2 ($d=512$) | Isolates diagnostic specimen crop $I_{\text{child}}$ from parent environmental scene $I_{\text{parent}}$ to eliminate background bias. |
| **Biogeographically Filtered HNSW** | LanceDB / embedded HNSW with Haversine distance ($\le 100\text{ km}$) + Julian day phenology | Prunes biologically impossible taxa before visual nearest-neighbor search. |
| **Reciprocal Rank Fusion (RRF)** | Rank aggregation engine ($k=60$) | Combines visual embedding similarity with taxonomic taxonomy tree distance. |
| **VLM Differential Diagnosis Cards** | Qwen2-VL-7B-Instruct with strict JSON Schema BNF grammar decoding | Generates verifiable diagnostic cards highlighting diagnostic anatomy (wing venation, spur length). |
| **$O(1)$ DuckDB Catalog Mutations** | DuckDB vectorized relational mapping tables | Instant biological taxonomy mutations (splits and lumps) without re-embedding datasets. |

---

## 5. Automated Four-Gate Model Promotion Strategy

To protect the live identification service against silent regressions on long-tail taxa, quantisation collapse, latency violations, or conformal non-exchangeability, model registration and alias assignment are deferred until Stage 7 (`evaluate.py`) and governed by the **Four-Gate Promotion Engine** in [`src/taxon_vision/models/training/promotion.py`](file:///home/benni/Documents/antigravity_workspace/taxon-vision-mlops/src/taxon_vision/models/training/promotion.py) (detailed in [Four-Gate Model Promotion & Active Model Passport](../model_engineering/model_promotion_gates.md)):

1. **Gate 1 - Predictive Viability & INT8 Quantisation Drop:** Compares holdout Macro PR-AUC against the incumbent `@Production` model version in the MLflow Model Registry ($\text{PR-AUC}_{\text{cand}} \ge \text{PR-AUC}_{\text{incumbent}}$) and verifies $\max(0, \text{Acc}_{\text{FP32}} - \text{Acc}_{\text{INT8}}) \le 0.03$ (`max_int8_accuracy_drop: 0.03`, empirically `1.22%`).
2. **Gate 2 - Conformal Coverage Guarantee:** Enforces $\hat{c}_{\text{test}} \ge (1 - \alpha) - 0.05 = 0.90$ across $N_{\text{test}} = 164$ holdout observations (empirically `91.46%` on holdout test and `96.77%` on calibration).
3. **Gate 3 - Hardware-Aware p95 Latency Budget:** Enforces $\text{Latency}_{\text{p95}} \le 120.0\text{ ms}$ on the Stage 7 CPU/CI SLA gate (empirically `50.38 ms` p95 / `47.38 ms` p50 on CPU INT8; $\le 15.0\text{ ms}$ p95 on bare-metal GPU serving profiles).
4. **Gate 4 - Calibration Exchangeability ($\text{MMD}_u^2$):** Asserts `mmd2_p_value >= 0.01` over the latest live telemetry window ($N_{\text{live}} \ge 50$) or allows initial cold-start bootstrap deployment (`mmd2_p_value is None`).

Candidates passing all four gates are assigned the **`@Production`** alias (or **`@Challenger`** when `stage_as_challenger: true` for Traefik 80/20 canary testing), whereas candidates failing any gate are tagged **`@Archived`** with explicit `rejection_reasons`.

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
