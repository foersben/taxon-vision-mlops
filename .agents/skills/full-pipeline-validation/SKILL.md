---
type: Agent Skill
name: full-pipeline-validation
title: Full Pipeline Validation
status: stable
stale_after: "2027-01-01T00:00:00Z"
version: 1.0
description: Coordinated pipeline executing lint, type checks, unit tests, coverage, and 7-stage DVC pipeline.
tags: [skill, automation, mlops, pipelines]
generated: {by: process:okf-updater, at: "2026-10-17T22:30:00Z"}
verified: {by: process:okf-updater, at: "2026-10-17T22:30:00Z"}
---

# Full Pipeline Validation

* **Description:** Coordinated quality assurance and DVC execution pipeline verifying code standards, static typing, test coverage, and complete 7-stage pipeline reproducibility under root `params.yaml`.
* **Invocation:** Triggered via `/validate-full-stack` or `/full-pipeline-validation` in the AI IDE.

## Step-by-Step Procedure

* Run pre-commit quality gates and type checks:

```bash
pixi run -e dev ruff check . && pixi run -e dev mypy src scripts
pixi run -e dev python scripts/governance/validate_okf.py
pixi run -e dev python -m taxon_vision.pipelines.audit_license
pixi run -e dev complexipy src
```

* Execute targeted and global unit tests ensuring code coverage meets or exceeds 80%:

```bash
pixi run -e dev pytest tests/ --cov=src/taxon_vision --cov-fail-under=80 -o "addopts="
```

* Execute the canonical 7-stage DVC pipeline driven by root `params.yaml`:

```bash
pixi run -e dev dvc repro
```

* Verify Epistemic Diamond artifacts across the workspace:
    * Raw and verified Parquet manifests: `data/manifests/raw_manifest.parquet` and `data/manifests/dataset_manifest.parquet`.
    * 4-way stratified embedding tensors: `data/embeddings/train_embeddings.pt`, `data/embeddings/val_embeddings.pt`, `data/embeddings/cal_embeddings.pt`, and `data/embeddings/test_embeddings.pt`.
    * Trained head checkpoint: `models/checkpoints/head.pt`.
    * Quantized production ONNX graph: `models/checkpoints/model_int8.onnx`.
    * Conformal parameters and metrics: `config/conformal_config.yaml` and `reports/conformal_metrics.json`.
    * Pareto evaluation profiles: `docs/assets/pareto_results.json` and `reports/pareto_evaluation.json`.

* Compile and strictly validate documentation:

```bash
pixi run -e dev python scripts/governance/visualize_okf.py
pixi run -e dev zensical build --strict
```
