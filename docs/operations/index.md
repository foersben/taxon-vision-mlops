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

This section covers physical hardware realities, including bare-metal GPU provisioning and scheduling constraints. It provides the deployment runbook for containerizing services and orchestrating them via CI/CD pipelines (integrating GitHub Actions and Jenkins). Furthermore, it details API contracts for the FastAPI serving layer and the Prometheus telemetry stack required to continuously monitor concept drift and maintain Service Level Agreements (SLAs).

## Thematic Sections

* **[Current Infrastructure](current_infrastructure.md)**: State of the bare-metal servers, K3s, and network architecture.
* **[GPU Resource Scheduling](gpu_resource_scheduling.md)**: Multiplexing workloads across the RTX 5070 Ti.
* **[Jenkins CI/CD Infrastructure](jenkins_ci_infrastructure.md)**: Zero-open-ports controller, Cloudflare Tunnel, and k3s ephemeral runners.
* **[Deployment Runbook](deployment_runbook.md)**: Containerization, orchestration, and disaster recovery.
* **[API Reference](api_reference.md)**: FastAPI endpoints, schemas, and interaction contracts.
* **[Prometheus Observability](prometheus_and_observability.md)**: Tracking feature drift, latency histograms, and system metrics.
* **[Agentic Governance & OKF](agentic_governance_and_okf.md)**: Open Knowledge Format specification, trust tiers, and router invariants.
* **[Implementation Status & Roadmap](implementation_status.md)**: Milestone progress, Phase 1-3 roadmap, and automated model promotion criteria.
