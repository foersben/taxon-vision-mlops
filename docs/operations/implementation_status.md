---
type: Report
title: Implementation Status & Roadmap
status: draft
stale_after: "2026-11-01T00:00:00Z"
version: 1.0
description: Tracking the progress and remaining effort for the MLOps Project Phases 1-3.
tags: [roadmap, status, project-management, mlops]
generated: {by: process:okf-updater, at: "2026-10-06T19:50:00Z"}
verified: {by: process:okf-updater, at: "2026-10-11T00:30:00Z"}
---

# Implementation Status: MLOps Roadmap

This document tracks the current implementation status of the project against the formal academic requirements, along with estimated engineering effort for the remaining items.

## Current Progress

### Phase 1 (Foundations): 100% Done

* ✅ Project objectives & reproducible environment (`pixi`, `just`).
* ✅ Collect data & database initialization (DuckDB & Parquet streaming).
* ✅ Build baseline model & inference API (`/api/v1/predict`, `/api/v1/train`, and `/api/v1/explain` endpoints).

### Phase 2 (Microservices, Tracking & Versioning): 100% Done

* ✅ S3 streaming feature extraction pipeline (`extract_features.py`) and cached representation training.
* ✅ DVC pipeline foundation (`dvc.yaml` & `dvc.lock`) with DagsHub S3 remote (`s3://dvc`).
* ✅ Ephemeral in-memory secret retrieval via KeePassXC and Linux Secret Service (`just dvc-push`).
* ✅ MLflow integration on DagsHub (logging run metrics, compound PR-AUC scores, and registry promotion).
* ✅ Containerization & Publishing: Multi-stage non-root `Dockerfile` with automated GHCR publication (`ghcr.io/foersben/taxon-vision-mlops:latest`).
* ✅ Kubernetes Deployment: `k3s` manifests with 4-replica NVIDIA GPU time-slicing and live ingress at `https://taxon.benjamin-foerster.eu/`.
* ✅ Automated Continuous Delivery: Jenkins pipeline on `hive-mind` with ephemeral k3s agent pods, GPU time-slicing slot, NVMe Rattler caching, and rolling rollout.

### Phase 3 (Monitoring & Maintenance): ~60% Done

* ✅ Prometheus metrics instrumentation (`/metrics`) exposing latency histograms, conformal set distributions, and referral counts.
* ✅ Split Conformal Prediction engine guaranteeing marginal error bounds ($1 - \alpha$).
* ✅ Out-of-Distribution (OOD) energy score gating and human triage routing.
* ✅ Server-rendered Jinja2 & HTMX operator dashboard for live image verification.
* ❌ **Future Enhancements:** Automated Evidently drift alarms in scheduled DAGs and standalone Grafana alert webhooks.

---

## Remaining Implementation Effort & Strategy

If the project requires strictly adhering to the remaining requirements, here is the estimated effort and technical approach:

### 1. DVC to version datasets and store hashes in MLflow (1-2 Hours)

* ✅ **Completed:** The `_init_mlflow_run` method in the training pipeline now directly parses `dvc.lock` (bypassing Git entirely as required) to extract the true DVC `md5` dataset hash and logs it as `dvc_dataset_hash` in MLflow.
* ✅ **Completed (Phase 3 DAG Modernization):** Modernized `dvc.yaml` into an unbroken 7-stage single-trunk DAG (`ingestion`, `audit_license`, `extract_features`, `train`, `export`, `calibrate`, `eval`) with multi-remote storage (`local-seaweedfs` for high-throughput embedding caches and `origin` DagsHub S3 for model releases), declarative S3 ConfigMap in Kubernetes (`deploy/k8s/mlops-storage-and-optuna.yaml`), uncached metrics tracking, and root symlinked `params.yaml`.

### 2. Docker-based microservices using `docker-compose` (2-3 Hours)

* **Work:** Straightforward. We need to write a cleanly segregated `docker-compose.yml` that orchestrates:
    1. The FastAPI Serving container.
    2. An MLflow tracking server container (with a local SQLite backend).
    3. (Optionally) the Airflow scheduler/webserver containers.

### 3. Scheduled training via Airflow (3-4 Hours)

* **Work:** Moderate. Airflow is resource-intensive, but we can use the official lightweight Docker-compose setup. We would write a simple Python DAG (`dags/model_training_dag.py`) that runs on a `schedule_interval='@weekly'`. The DAG would contain a `BashOperator` that triggers our `training.py` script inside the designated container.

### 4. Drift Detection with Evidently in Airflow (2-4 Hours)

* **Work:** Moderate. We would add an evaluation task to the Airflow DAG. Before kicking off an expensive training run, the DAG pulls the recent inference data from DuckDB, compares it to the reference training dataset using `evidently.Report`, and logs the resulting HTML dashboard as an artifact to MLflow. If statistical drift is detected, it triggers the downstream training task; if not, the DAG gracefully skips the training step.

---

## Automated Model Promotion Strategy

To protect the core identification service against regression-especially on rare, long-tail taxa where standard accuracy is misleading-we implement a **PR-AUC Safe Promotion** strategy via the MLflow Model Registry.

### 1. The Core Metric

Models are evaluated strictly on a robust, imbalance-aware metric rather than raw accuracy:

$$ \text{Promotion Score} = \text{Macro PR-AUC} $$

* **Macro PR-AUC (Precision-Recall AUC):** Heavily penalizes false positives on rare classes. By relying exclusively on this metric for automated promotion decisions, we ensure that performance gains are true across all taxa, not just the most common ones.

### 2. Registry Promotion Logic

When a new classification head finishes training, it is automatically logged to the `taxon_vision_classifier` registry. The pipeline then compares its Macro PR-AUC score against the model currently tagged as `@Production`:

* **Outperforms Production:** The new model is automatically tagged as **`@Challenger`** (or `@Staging`). It is **not** immediately pushed to `@Production`. This ensures we keep at most two competing models active, and stages the new model for deeper human review (or integration tests).
* **Underperforms Production:** The new model is tagged as **`@Archived`** (or `@Rejected`), ensuring poor models never pollute the deployment pipeline.

---

## Verified Implementation Gaps & Technical Debt

To maintain absolute scientific and engineering integrity, the following implementation discrepancies between target architecture and active repository state are documented:

* **Synthetic Placeholder Training Split:** Resolved. Real foundation vision feature extraction is implemented in `taxon_vision.pipelines.extract_features` (`build_taxon_index_map`, `stream_and_extract_representations`, `create_stratified_embedding_split`), streaming observations from AWS S3 and caching to `data/embeddings/cached_features.pt`. The training runner (`taxon_vision.models.training.runner`) supports `--cache-path` and auto-loads `cached_features.pt`, retaining the synthetic Gaussian split strictly as a graceful offline fallback.
* **Container Registry Publishing:** Resolved. GitHub Actions automatically packages the production container and publishes multi-tagged images (`latest`, `sha-<short_sha>`) to `ghcr.io/foersben/taxon-vision-mlops`, hydrated with promoted checkpoints from DagsHub DVC. The Kubernetes deployment continuously pulls releases via `imagePullPolicy: Always`.
* **Cognitive Complexity Quality Gate:** Resolved. Integrated `complexipy` into CI and refactored deep functions (`_prepare_embedding_split`, `build_taxon_index_map`) to guarantee cognitive complexity $\le 15$ across all code paths.
* **Jenkins Agent Pod GPU Allocation:** Resolved. In `Jenkinsfile`, the `Model Training & Checkpoint (GPU)` stage defines a dedicated agent pod requesting `nvidia.com/gpu: 1` in limits and requests, binding dynamically to the 4-replica time-slicing configuration on `hive-mind`.
* **Blackwell Toolchain Upgrade:** Resolved. Environment manifests (`pyproject.toml`, `pixi.lock`) are upgraded to PyTorch 2.7.0 targeting CUDA 12.8 (`cu128`), natively supporting SM 120 kernels on the host's NVIDIA Driver 615.71.09 / CUDA 13.4 UMD.
* **Error Masking in CI Pipeline:** Resolved. Removed all illegitimate `|| true` suppressions from `Jenkinsfile` across training and deployment stages (`dvc repro`, `dvc push`, `rollout restart`, and `rollout status`), replacing them with explicit service availability checks and fail-fast assertions.
* **Coverage Gate Inconsistency:** Resolved. Harmonized coverage thresholds between `Jenkinsfile` and `.github/workflows/ci.yml` at 80% (active global test suite achieves 89% line coverage across 124 passing tests).
* **Optuna Evaluation Objective & Search Space:** Resolved. In `taxon_vision.models.training.tuning`, the objective function returns $\max(\text{val\_pr\_auc})$ matching peak restored checkpoint weights, and Optuna samples categorical batch sizes (`[32, 64, 128]`) and embedding noise jitter regularisation (`noise_std`).
* **Unscheduled Drift and Active Learning Modules:** MMD feature drift (`monitoring/drift.py`) and BADGE acquisition (`active_learning/query.py`) exist as standalone modules, but lack recurring cron jobs, webhook triggers, or automated retraining invocations.
* **Local In-Cluster S3 Storage Engine (SeaweedFS):** Migrated from MinIO/Silo to SeaweedFS (`docker.io/chrislusf/seaweedfs:latest`), licensed under Apache 2.0. Eliminates AGPLv3 compliance risks and sidecar bucket provisioning loops. Its Haystack/Needle architecture prevents disk-seek bottlenecks, providing sub-2ms random read access for the 100,000 exemplar observation crops and LanceDB vector partitions planned in Pipeline 2.
