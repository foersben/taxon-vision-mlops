---
type: Agent Memory
title: TaxonVision-MLOps Architecture State Map
status: stable
stale_after: "2027-01-01T00:00:00Z"
version: 1.0
description: Live architecture state detailing implemented pipeline stages, inference engines, conformal safety, and serving layers.
tags: [architecture, status, memory, mlops, taxon-vision]
generated: {by: process:okf-updater, at: "2026-10-30T00:00:00Z"}
verified: {by: process:okf-updater, at: "2026-10-30T00:00:00Z"}
---

# TaxonVision-MLOps Architecture State Map

## 1. Toolchain & Environment Invariants

* **Python Runtime:** Python 3.12 managed deterministically via Pixi (`pixi.lock`). Dual environment targets: `dev` (with CUDA 12.8 GPU support) and `ci-dev` (CPU-only for CI runners).
* **Command Dispatch:** All Python executions mediated via `pixi run -e dev` or `just`. Direct invocations of `python`, `pip`, or `.venv` are strictly banned.
* **Code Style & Type Checking:** Ruff (line length 120, Google docstrings, strict linting) and MyPy strict mode (`strict = true`).
* **Cognitive Complexity:** Monitored via `complexipy`, strictly ratcheted to `<= 15` per function via baseline watermark.

## 2. 7-Stage Epistemic Diamond DAG Pipeline

* **Stage 1 (Ingest):** S3 image streaming from iNaturalist Open Data bucket, validating CC0, CC-BY, and CC-BY-NC licenses while filtering non-compliant static domains.
* **Stage 2 (Audit License):** Verifies license compliance and photographic attribution retention, outputting `dataset_manifest.parquet`.
* **Stage 3 (Extract Features):** Batch inference over frozen foundation vision backbones (BioCLIP-2 / DINOv2 / DINOv3), caching normalized feature embeddings across 4 splits (`train`, `val`, `cal`, `test`).
* **Stage 4 (Train Head):** Class-balanced cosine classifier training with PyTorch, Optuna HPO, and MLflow run tracking.
* **Stage 5 (Export ONNX & Quantize):** Zero-allocation ONNX model export with INT8 Dynamic Quantization and FP16 support.
* **Stage 6 (Calibrate Conformal & OOD):** Split Conformal Prediction calculating empirical non-conformity quantile $\hat{q}$ to guarantee $1 - \alpha$ marginal coverage; Energy-based Out-of-Distribution (OOD) detector calibration.
* **Stage 7 (Evaluate & Pareto Benchmark):** Holdout evaluation calculating top-1/top-5 accuracy, conformal prediction set sizes, p95 latency benchmarks, and Grafana Prometheus metrics.

## 3. Serving & UI Architecture

* **FastAPI Backend:** Machine REST API (`/predict`, `/feedback`, `/health`) and Prometheus `/metrics`.
* **Asynchronous Non-Blocking Invariant:** Heavy CPU-bound inference and disk I/O offloaded via `await anyio.to_thread.run_sync`.
* **Jinja2 + HTMX Dashboard:** Operator interface for live drag-and-drop classification, Grad-CAM heatmap visualization, and conformal prediction set inspection.
* **Distributed Upload Cache:** Redis-backed deduplication and transient caching preventing duplicate S3 transfers and split-brain states across replicas.
