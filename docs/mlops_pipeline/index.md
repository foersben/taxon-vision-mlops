---
type: Architecture
title: MLOps Pipelines & Polyglot Storage
status: stable
stale_after: "2027-01-01T00:00:00Z"
version: "1.0"
description: Overview of the 8-stage Epistemic Diamond DVC pipeline, end-to-end closed-loop MLOps feedback cycle, asymmetric dual-pipeline topology, and polyglot storage architecture.
tags: [mlops, dvc, pipelines, storage, architecture]
generated: {by: "process:docs-librarian", at: "2026-10-18T18:00:00Z"}
verified: {by: "process:docs-librarian", at: "2026-11-08T17:30:00Z"}
---

# MLOps Pipelines & Polyglot Storage

This section explores the overarching data flow, closed-loop feedback lifecycle, and polyglot storage topology that power the TaxonVision-MLOps ecosystem.

In a system dealing with evolving ecological distributions and citizen-science observation streams, a static train-once pipeline rapidly loses statistical validity. Here, we begin with a conceptual and architectural walkthrough of the **8-Station Closed-Loop MLOps Lifecycle**, then examine the **Asymmetric Dual-Pipeline Architecture** that separates neural weight updates from knowledge graph mutations, and finally detail how datasets, embeddings, and telemetry records are versioned across DVC, SeaweedFS, and our consolidated PostgreSQL 16 cluster.

## Recommended Reading Order

* **[Closed-Loop MLOps Overview](closed_loop_overview.md)**: Beginner-to-expert walkthrough of the 8-station feedback loop connecting online inference, PostgreSQL telemetry, two-tier drift detection, Airflow 3 orchestration, Jenkins GPU retraining, and MLflow 3.x promotion.
* **[Dual Pipeline Topology](dual_pipeline_architecture.md)**: Decoupling the weight-centric classification engine from the data-centric Multimodal Knowledge Graph.
* **[Ingestion, Feedback & Versioning](ingestion_and_versioning.md)**: Reproducible data ingestion (`frozen: true`), active learning feedback merge (`feedback.py`), and the 8-stage DVC `dvc.yaml` specification including Stage 3.5 (`data_drift`).
* **[Polyglot Storage & Databases](storage_and_databases.md)**: Multi-tier persistence across SeaweedFS (`taxon-dvc-cache`, `triage-queue`, `mlflow-artifacts`), the consolidated PostgreSQL 16 cluster (`telemetry`, `optuna`, `airflow`, `mlflow`), DuckDB, Polars, and Redis.
* **[Interactive DVC DAG Explorer](dvc_dag_explorer.md)**: Interactive visualisation and artefact registry of the 8-stage Epistemic Diamond build graph.
