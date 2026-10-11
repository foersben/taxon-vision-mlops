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

* ✅ DVC pipeline foundation (`dvc.yaml` & `dvc.lock`) with DagsHub S3 remote (`s3://dvc`).
* ✅ Ephemeral in-memory secret retrieval via KeePassXC and Linux Secret Service (`just dvc-push`).
* ✅ MLflow integration on DagsHub (logging run metrics, compound PR-AUC scores, and registry promotion).
* ✅ Containerization: Multi-stage non-root `Dockerfile` and local developer stack (`docker-compose.local.yaml`).
* ✅ Kubernetes Deployment: `k3s` manifests with 4-replica NVIDIA GPU time-slicing and NodePort `30080`.
* ✅ Automated Continuous Delivery: Jenkins pipeline on `hive-mind` with ephemeral k3s agent pods, NVMe Rattler caching, and rolling rollout.

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

* **Synthetic Placeholder Training Split:** In `taxon_vision.models.training.runner`, `_prepare_embedding_split` constructs synthetic random embeddings ($N=150$) to validate pipeline mechanics. Real foundation feature extraction is implemented in `taxon_vision.models.backbones.factory.extract_and_cache_features` but is not yet wired to the training CLI runner.
* **Jenkins Agent Pod GPU Allocation:** Resolved. In `Jenkinsfile`, the `ml-runner` pod template specifies `nvidia.com/gpu: 1` limits and requests, binding dynamically to the 4-replica time-slicing configuration on `hive-mind`.
* **Blackwell Toolchain Upgrade:** Resolved. Environment manifests (`pyproject.toml`, `pixi.lock`) are upgraded to PyTorch 2.7.0 targeting CUDA 12.8 (`cu128`), natively supporting SM 120 kernels on the host's NVIDIA Driver 615.71.09 / CUDA 13.4 UMD.
* **Error Masking in CI Pipeline:** Several commands in `Jenkinsfile` are suffixed with `|| true`, which masks runtime failures in test or training steps.
* **Coverage Gate Inconsistency:** Line coverage threshold is set to 78% in `Jenkinsfile`, whereas `.github/workflows/ci.yml` enforces an 80% threshold.
* **Optuna Evaluation Objective & Search Space:** Resolved. In `taxon_vision.models.training.tuning`, the objective function returns $\max(\text{val\_pr\_auc})$ matching peak restored checkpoint weights, and Optuna samples categorical batch sizes (`[32, 64, 128]`) and embedding noise jitter regularisation (`noise_std`).
* **Unscheduled Drift and Active Learning Modules:** MMD feature drift (`monitoring/drift.py`) and BADGE acquisition (`active_learning/query.py`) exist as standalone modules, but lack recurring cron jobs, webhook triggers, or automated retraining invocations.
