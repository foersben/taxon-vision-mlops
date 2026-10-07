---
type: Report
title: Implementation Status & Roadmap
status: draft
stale_after: "2026-11-01T00:00:00Z"
version: 1.0
description: Tracking the progress and remaining effort for the MLOps Project Phases 1-3.
tags: [roadmap, status, project-management, mlops]
generated: {by: process:okf-updater, at: "2026-10-06T19:50:00Z"}
verified: {by: process:okf-updater, at: "2026-10-06T19:50:00Z"}
---

# Implementation Status: MLOps Roadmap

This document tracks the current implementation status of the project against the formal academic requirements, along with estimated engineering effort for the remaining items.

## Current Progress

### Phase 1 (Foundations): 100% Done

* ✅ Project objectives & reproducible environment (`pixi`, `just`).
* ✅ Collect data & database initialization (DuckDB integration).
* ✅ Build baseline model & inference API (`/predict`, `/training`, and `/explain` endpoints).

### Phase 2 (Microservices, Tracking & Versioning): ~60% Done

* ✅ DVC pipeline foundation is set up (`dvc.yaml` exists).
* ✅ MLflow integration (logging run metrics, compound PR-AUC scores, and registry promotion).
* ❌ **Missing:** Docker-compose microservice orchestration.
* ❌ **Missing:** Scheduled training (via cron or Airflow).

### Phase 3 (Monitoring & Maintenance): ~10% Done

* ✅ Theoretical observability design (Prometheus metrics mapped out in architecture).
* ❌ **Missing:** Evidently data drift detection.
* ❌ **Missing:** Grafana dashboard and webhook alerts.
* ❌ **Missing:** Streamlit demonstration application.

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
