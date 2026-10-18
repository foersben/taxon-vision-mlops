---
type: Guide
title: Codebase Explorer & Fast Routing Index
status: stable
stale_after: "2027-01-01T00:00:00Z"
version: 1.0
description: Interactive subpackage explorer widget, domain-to-test routing matrix, and cognitive complexity inspection.
tags: [codebase, routing, tests, architecture, modules, explorer]
generated: {by: "process:docs-librarian", at: "2026-10-18T12:00:00Z"}
verified: {by: "process:docs-librarian", at: "2026-10-18T12:00:00Z"}
---

# Codebase Explorer & Fast Routing Index

This interactive guide serves as the primary navigation index for software engineers, ML researchers, and autonomous AI agents. Use the interactive explorer below to filter and inspect any module across the 4 architectural tiers of `src/taxon_vision/`, reviewing responsibilities, class signatures, DVC stage consumers, and cognitive complexity scores.

---

## 1. Interactive Subpackage & Module Explorer

Filter by architectural tier or search dynamically by module, class, or method name:

<div id="taxon-codebase-explorer">
    <div style="padding: 2rem; text-align: center; opacity: 0.7;">
        Initializing Subpackage & Module Explorer...
    </div>
</div>

---

## 2. Fast Domain Routing Matrix

When modifying specific subsystems, read only the source files in your active domain and execute targeted test sweeps to maintain token efficiency:

| Subsystem Domain | Primary Source Files (`src/taxon_vision/`) | Targeted Test Suite (`tests/`) | AI Specialist Role |
| :--- | :--- | :--- | :--- |
| **Config & Types** | `config/{loader,schemas}.py`, `params.yaml`, `config/default_config.yaml` | `tests/unit/test_config.py` | `@orchestrator` |
| **Data & S3** | `data/{s3_streamer,inat_client,license_filter}.py` | `tests/unit/test_data_components.py` | `@mlops-data-engineer` |
| **Pipelines** | `pipelines/{ingest,audit_license,extract_features,train,export_onnx,calibrate,evaluate}.py` | `tests/unit/test_pipelines.py` | `@mlops-data-engineer` |
| **Models & Loss** | `models/{factory,head,loss}.py` | `tests/unit/test_models_components.py` | `@model-engineer` |
| **Training & HPO** | `models/training/{runner,loops,embeddings,tuning,tracking,checkpoint,types}.py` | `tests/unit/test_tuning.py`, `tests/unit/test_tracking.py` | `@model-engineer` |
| **ONNX & Inference** | `inference/{onnx_exporter,onnx_engine,quantizer,cam,calibration,cli}.py` | `tests/unit/test_onnx_exporter.py`, `tests/unit/test_quantizer.py` | `@inference-optimizer` |
| **OOD & Conformal** | `uncertainty/{ood_detector,conformal}.py` | `tests/unit/test_ood_energy.py`, `tests/unit/test_conformal_bounds.py` | `@uncertainty-safety-engineer` |
| **Active Learning** | `active_learning/{query,triage_queue}.py` | `tests/unit/test_active_learning.py` | `@active-learning-engineer` |
| **Monitoring** | `monitoring/{telemetry,drift}.py` | `tests/unit/test_monitoring.py` | `@observability-engineer` |
| **FastAPI & HTMX** | `service/{api,inference,schemas,storage}.py`, `service/routes/*.py` | `tests/integration/test_api_endpoints.py`, `tests/integration/test_views_and_routes.py` | `@api-and-ui-developer` |

---

## 3. Fast Targeted Verification Protocol

During local development, suppress global coverage checks to verify subsystem behavior in milliseconds:

```bash
# Verify Conformal Bounds & Empirical Coverage
pixi run -e ci-dev pytest tests/unit/test_conformal_bounds.py -q -o "addopts="

# Verify ONNX Exporter & INT8 Quantization
pixi run -e ci-dev pytest tests/unit/test_onnx_exporter.py -q -o "addopts="

# Verify S3 Streaming & Licensing Filters
pixi run -e ci-dev pytest tests/unit/test_data_components.py -q -o "addopts="
```

---

## 4. Comprehensive Quality Gate Protocol

Execute the mandatory quality gates across all 66 source files prior to creating a commit:

```bash
# 1. Strict Ruff formatting and linting
pixi run -e dev ruff check .

# 2. Strict MyPy static type analysis
pixi run -e dev mypy src scripts

# 3. Licensing compliance verification
pixi run -e dev python -m taxon_vision.pipelines.audit_license

# 4. Open Knowledge Format governance audit
pixi run -e dev python scripts/governance/validate_okf.py
```
