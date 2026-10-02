---
type: Reference
title: TaxonVision Routing & Capabilities
status: stable
stale_after: "2027-01-01T00:00:00Z"
version: 1.0
description: Primary routing table for AI IDEs defining roles in .agents/roles/ and core MLOps constraints.
tags: [agents, guidelines, mlops]
generated: {by: process:okf-updater, at: "2026-10-02T10:00:00Z"}
verified: {by: process:okf-updater, at: "2026-10-02T10:00:00Z"}
---

Primary routing table for AI IDEs defining roles in `.agents/roles/` and core constraints.

## Core Architecture Constraints

* **Dual-Target Toolchain:** All development and testing execute via `pixi run -e dev` or `just`. CI runs on CPU via `ci-dev`.
* **Zero-Allocation Inference:** ONNX Runtime inference hot paths must avoid dynamic memory allocation and Python object boxing.
* **Strict Open Licensing:** Only observations under open licenses (CC0, CC-BY, CC-BY-NC) are admitted into training datasets. Attribution metadata must be retained.
* **Conformal Coverage Guarantee:** Split Conformal Prediction must mathematically guarantee user-specified coverage ($1 - \alpha$). Observations exceeding uncertainty bounds or empty prediction sets route to human triage.
* **Unified Web Service:** FastAPI powers both the machine REST API and the server-rendered Jinja2/HTMX operator dashboard.

## AI Role Registry

| Role | Description | Trigger |
| --- | --- | --- |
| `@orchestrator` | PM. Delegates tasks; enforces OKF structure. | Planning, refactoring, workflows. |
| `@taxonomic-architect` | Manages DarwinCore schemas and taxonomic trees. | Taxonomy, GBIF, biological classifications. |
| `@mlops-data-engineer` | S3 streaming, DVC versioning, license filtering. | Data ingestion, DVC, AWS datasets. |
| `@model-engineer` | Timm backbone factory, Class-Balanced Loss, training. | PyTorch training, model heads, loss functions. |
| `@inference-optimizer` | ONNX Runtime, quantization (INT8/FP16), Grad-CAM. | Inference speed, latency, ONNX export. |
| `@uncertainty-safety-engineer` | Conformal prediction sets, OOD energy detection. | Uncertainty calibration, coverage bounds, OOD. |
| `@active-learning-engineer` | BADGE, CoreSet query strategies, human review queue. | Active learning, triage ranking, annotations. |
| `@observability-engineer` | Prometheus metrics, embedding & concept drift. | Telemetry, drift detection, Pareto curves. |
| `@api-and-ui-developer` | FastAPI endpoints, Jinja2/HTMX interactive views. | REST API, upload routes, web dashboard. |
| `@qa-automator` | Fixtures (CUB-200), hypothesis tests, CI gates. | Tests, regression prevention, test suites. |
| `@docs-librarian` | Maintains Zensical docs, KaTeX equations, OKF graph. | Documentation, diagrams, LaTeX equations. |
| `@git-operator` | Manages branches, GPG signed commits, releases. | Git actions, signed commits, release tags. |
