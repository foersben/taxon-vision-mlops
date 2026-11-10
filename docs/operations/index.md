---
type: Guide
title: Operations, Telemetry & Governance
status: stable
stale_after: "2027-01-01T00:00:00Z"
version: "1.0"
description: Production operations, bare-metal k3s infrastructure, GPU time-slicing, Jenkins ephemeral CI/CD, Prometheus monitoring, and OKF governance.
tags: [operations, infrastructure, kubernetes, gpu, ci-cd, prometheus, governance]
generated: {by: "process:docs-librarian", at: "2026-10-18T18:00:00Z"}
verified: {by: "process:docs-librarian", at: "2026-10-18T18:00:00Z"}
---

# Operations, Telemetry & Governance

This section is a comprehensive guide to deploying, operating, and monitoring the TaxonVision-MLOps ecosystem in production.

This section covers physical hardware realities, including bare-metal GPU provisioning and scheduling constraints. It provides the deployment runbook for containerizing services and orchestrating them via CI/CD pipelines (integrating GitHub Actions and Jenkins). Furthermore, it details API contracts for the FastAPI serving layer and the Prometheus telemetry stack required to continuously monitor concept drift and maintain Service Level Objectives (SLOs).

## Recommended Reading Order

### 1. Infrastructure, Security & Continuous Delivery

* **[Current Infrastructure](current_infrastructure.md)**: Bare-metal hardware profile (`hive-mind`), K3s cluster configuration, consolidated PostgreSQL 16 + SeaweedFS storage plane, and local Docker Compose topology.
* **[GPU Resource Scheduling](gpu_resource_scheduling.md)**: Multiplexing workloads across the NVIDIA GeForce RTX 5070 Ti via 4-slice Kubernetes time-slicing and software VRAM fraction limits ($f=0.7$).
* **[Security & Credentials Governance](security_and_credentials.md)**: Defence-in-depth for MLOps, why `.env` files are hazardous in agentic workspaces, KeePassXC `secret-tool` in-memory injection, the Four Storage Pillars, and the 2-year amnesia-resistant dual-tier documentation strategy.
* **[Multi-Tier Deployment Strategy](deployment_strategy.md)**: Two-layer hybrid CI/CD (GitHub Actions + Jenkins), three-tier promotion funnel (`feature/*` -> `develop` in `taxon-vision-staging` -> `main` in `taxon-vision`), native Kustomize overlays, and two-stage emergency rollback playbooks.
* **[Jenkins CI/CD Infrastructure](jenkins_ci_infrastructure.md)**: Zero-open-ports controller, Cloudflare Tunnel ingress, and ephemeral K3s runner pods.
* **[Deployment & Scaling Runbook](deployment_runbook.md)**: Day-0 bootstrap commands, Day-2 operational verification, Traefik 80/20 canary weight shifting, and database/SeaweedFS administration.

### 2. Observability, Drift Orchestration & Governance

* **[Drift Detection & Telemetry](drift_detection_and_telemetry.md)**: Mathematical formulation and implementation of the two-tier representation drift engine (1D Wasserstein-1 on $L_2$ norms + multivariate RKHS $\text{MMD}_u^2$ permutation test) and Evidently 0.7 PCA snapshots.
* **[Apache Airflow 3 Orchestration](airflow_orchestration.md)**: Architecture of the 4-container `LocalExecutor` stack, the hourly `prediction_drift_monitor` DAG, the webhook-driven `model_retraining` DAG, and AST DAG visualisation.
* **[Prometheus & Grafana Observability](prometheus_and_observability.md)**: Complete Prometheus metric catalogue, Grafana 6-panel dashboard provisioning, and Alertmanager webhook rules.
* **[FastAPI & OpenAPI Reference](api_reference.md)**: Complete HTTP contract for `/api/v1/predict`, `/api/v1/explain`, `/api/v1/drift/evaluate`, `/api/v1/drift/latest`, `/api/v1/retrain/webhook`, and HTMX view routes.
* **[Agentic Governance & OKF](agentic_governance_and_okf.md)**: Open Knowledge Format v0.2 specification, trust tiers, and AI agent router invariants.
* **[Implementation Status & Roadmap](implementation_status.md)**: Forensic four-tier verification matrix distinguishing active production code from target roadmap steps.
* **[Automation & Scripts Reference](automation_and_scripts.md)**: Reference guide to `Justfile` recipes and governance/CI scripts.
