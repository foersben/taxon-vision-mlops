---
type: Guide
title: Codebase Explorer & Fast Routing Index
status: stable
stale_after: "2027-01-01T00:00:00Z"
version: 1.0
description: Practical module routing index mapping codebase domains to primary source files, targeted unit tests, and AI specialist roles.
tags: [codebase, routing, tests, architecture, modules]
generated: {by: process:docs-librarian, at: "2026-10-18T12:00:00Z"}
verified: {by: process:docs-librarian, at: "2026-10-18T12:00:00Z"}
---

# Codebase Explorer & Fast Routing Index

This guide provides a module-by-module navigation map for developers and AI agents. To prevent cognitive overload and maintain fast iteration cycles, use this routing table to locate the primary implementation files and targeted test suites for any subsystem.

---

## 1. Domain Routing Matrix

| Subsystem Domain | Primary Source Files (`src/taxon_vision/`) | Targeted Test Suite (`tests/`) | AI Specialist Role |
| :--- | :--- | :--- | :--- |
| **Config & Types** | `config.py`, `params.yaml`, `config/default_config.yaml` | `tests/unit/test_config.py` | `@orchestrator` |
| **Data & S3** | `data/{s3_streamer,inat_client,license_filter}.py` | `tests/unit/test_data_components.py` | `@mlops-data-engineer` |
| **Pipelines** | `pipelines/{ingest,audit_license,extract_features,train,export_onnx,calibrate,evaluate}.py` | `tests/unit/test_pipelines.py` | `@mlops-data-engineer` |
| **Models & Loss** | `models/{factory,head,loss,trainer}.py` | `tests/unit/test_models_components.py` | `@model-engineer` |
| **Training & HPO** | `models/training/{runner,loops,embeddings,tuning,tracking,checkpoint,types}.py` | `tests/unit/test_tuning.py`, `tests/unit/test_tracking.py` | `@model-engineer` |
| **ONNX & Inference** | `inference/{onnx_exporter,onnx_engine,quantizer,cam,calibration,cli}.py` | `tests/unit/test_onnx_exporter.py`, `tests/unit/test_quantizer.py` | `@inference-optimizer` |
| **OOD & Conformal** | `uncertainty/{ood_detector,conformal}.py` | `tests/unit/test_ood_energy.py`, `tests/unit/test_conformal_bounds.py` | `@uncertainty-safety-engineer` |
| **Active Learning** | `active_learning/{query,triage_queue}.py` | `tests/unit/test_active_learning.py` | `@active-learning-engineer` |
| **Monitoring** | `monitoring/{telemetry,drift}.py` | `tests/unit/test_monitoring.py` | `@observability-engineer` |
| **FastAPI & HTMX** | `service/{api,inference,schemas,storage}.py`, `service/routes/*.py` | `tests/integration/test_api_endpoints.py`, `tests/integration/test_views_and_routes.py` | `@api-and-ui-developer` |

---

## 2. Fast Targeted Verification Protocol

When developing within a specific domain, suppress global coverage checks and run only the targeted unit test file for that subsystem:

```bash
# Example: Fast verification of Conformal Bounds
pixi run -e ci-dev pytest tests/unit/test_conformal_bounds.py -q -o "addopts="

# Example: Fast verification of ONNX Exporter
pixi run -e ci-dev pytest tests/unit/test_onnx_exporter.py -q -o "addopts="

# Example: Fast verification of Data Components
pixi run -e ci-dev pytest tests/unit/test_data_components.py -q -o "addopts="
```

---

## 3. Comprehensive Quality Gate

Before committing code, execute the mandatory pre-commit verification gates across all 66 source files:

```bash
pixi run -e dev ruff check .
pixi run -e dev mypy src scripts
pixi run -e dev python -m taxon_vision.pipelines.audit_license
pixi run -e dev python scripts/governance/validate_okf.py
pixi run -e dev python scripts/governance/verify_markdown_rules.py
```
