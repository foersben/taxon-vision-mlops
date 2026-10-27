# TaxonVision-MLOps: Automated Species Identification Service

[![CI & Deployment Pipeline](https://github.com/foersben/taxon-vision-mlops/actions/workflows/ci.yml/badge.svg)](https://github.com/foersben/taxon-vision-mlops/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Pixi](https://img.shields.io/badge/managed%20by-pixi-blue)](https://pixi.sh)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)

This repository provides a highly formalised, end-to-end MLOps pipeline and high-throughput production service for the purpose of species identification from citizen-science photographs (such as iNaturalist and GBIF DarwinCore). It has been engineered with rigorous attention to deterministic execution and state-of-the-art inference efficiency.

---

## Key Capabilities

* **Level 1 Closed Chain:** Ingestion of approximately 10 well-represented taxa, strict open license filtering (CC0, CC-BY, CC-BY-NC), attribution preservation, frozen feature extraction, and automated deployment.
* **Level 2 Scalability & Cost:** Automated multi-backbone Pareto benchmarking (`BioCLIP-2`, `DINOv3`, `DINOv2`, `MobileNetV4`, `EfficientNet`), ONNX Runtime export with dynamic INT8/FP16 quantisation, and rigorous DVC dataset versioning.
* **Level 3 Observability:** Automated feedback ingestion, taxonomic mutation reconciliation, and structured Prometheus telemetry.
* **Level 4 Robustness & Safety:** **Split Conformal Prediction** mathematically guaranteeing distribution-free $1 - \alpha$ coverage, energy-based OOD detection, active learning triage, Class-Balanced Loss for long-tail species, and low-latency Grad-CAM heatmaps (< 25 ms).

---

## Quickstart

### Option A: One-Command Microservices Deployment (Docker Compose)

Deploy the complete multi-service stack with a single command (requires only [Docker](https://docs.docker.com/get-docker/)):

```bash
# Clone the repository
git clone https://github.com/foersben/taxon-vision-mlops.git
cd taxon-vision-mlops

# Launch all 6 microservices (Traefik gateway, FastAPI app, SeaweedFS, Optuna, Prometheus, Grafana)
docker compose up -d --build
```

Access the operational interfaces:

* **Traefik Edge Ingress Gateway:** [http://localhost](http://localhost)
* **Traefik Management Dashboard:** [http://localhost:8081](http://localhost:8081)
* **TaxonVision Direct REST API:** [http://localhost:8000](http://localhost:8000)
* **Prometheus Telemetry Scraper:** [http://localhost:9090](http://localhost:9090)
* **Grafana Operational Dashboards:** [http://localhost:3000](http://localhost:3000) (credentials: `admin` / `admin`)
* **Optuna Bayesian Tuning Cockpit:** [http://localhost:8080](http://localhost:8080)
* **SeaweedFS S3 Storage UI:** [http://localhost:9333](http://localhost:9333)

To inspect logs or tear down the stack:

```bash
docker compose logs -f app
docker compose down
```

### Option B: Native Workstation Setup (Pixi & Just)

#### Prerequisites

* [Pixi](https://pixi.sh) installed (`curl -fsSL https://pixi.sh/install.sh | bash`)
* [Just](https://github.com/casey/just) installed (`cargo install just` or `pixi global install just`)

#### 1. Bootstrapping

```bash
# Clone the repository
git clone https://github.com/foersben/taxon-vision-mlops.git
cd taxon-vision-mlops

# Bootstrap Pixi dev environment, pre-commit hooks, and baseline weights
just setup --scratch
```

#### 2. Quality & Tests

```bash
# Run linting and type checks
just lint

# Run unit and integration tests
just test

# Validate Open Knowledge Format (OKF v0.2) frontmatter
just validate-okf
```

#### 3. Running the Service Locally

```bash
# Launch unified FastAPI + HTMX service on http://localhost:8000
just run
```

---

## Architectural Principles: The Epistemic Diamond DAG

TaxonVision enforces an **Epistemic Diamond DAG** topology across its unified 7-stage DVC pipeline, governed by the canonical root `params.yaml`:

```text
                       +-----------+
                       | ingestion |  (pipelines/ingest.py)
                       +-----------+
                             *
                             * (data/manifests/raw_manifest.parquet)
                             v
                    +---------------+
                    | audit_license |  (pipelines/audit_license.py) ---> [license_audit.json]
                    +---------------+
                             *
                             * (data/manifests/dataset_manifest.parquet)
                             v
                  +------------------+
                  | extract_features |  (pipelines/extract_features.py)
                  +------------------+
                   /        |         \
  (train & val.pt)/         |          \ (test_embeddings.pt)
                 /          |           \
                v           |            \
            +-------+       |             \
            | train |       |              \
            +-------+       | (cal.pt)      \
                *           |                \
         (head.pt)          |                 \
                *           |                  \
                v           |                   \
            +--------+      |                    \
            | export |      |                     \
            +--------+      |                      \
                *           |                       \
        (model_int8.onnx)   |                        \
                *           v                         \
                *====>+-----------+                    \
                *     | calibrate | (pipelines/        \
                *     +-----------+  calibrate.py)      \
                *           *                            \
                *   (conformal_config.yaml)               \
                *           *                              \
                *           v                               v
                *========>+------------------------------------+
                          |                eval                | (pipelines/evaluate.py)
                          +------------------------------------+
                                            |
                                            +---> [pareto_evaluation.json]
                                            +---> [docs/assets/pareto_results.json]
```

### The 7 Canonical DVC Stages

All stages are implemented as first-class CLI entrypoints under `src/taxon_vision/pipelines/`:

* **Stage 1 (Ingestion):** `taxon_vision.pipelines.ingest` fetches DarwinCore observation metadata from iNaturalist into `data/manifests/raw_manifest.parquet`.
* **Stage 2 (License Gatekeeper):** `taxon_vision.pipelines.audit_license` enforces strict open licensing (CC0, CC-BY, CC-BY-NC) and filters compliant records to `data/manifests/dataset_manifest.parquet`.
* **Stage 3 (Feature Extraction):** `taxon_vision.pipelines.extract_features` streams images from S3 and partitions frozen representations into 4 stratified tensors: `train_embeddings.pt`, `val_embeddings.pt`, `cal_embeddings.pt`, and `test_embeddings.pt`.
* **Stage 4 (Head Training):** `taxon_vision.pipelines.train` trains a classification head with Class-Balanced Loss on train and validation embeddings, producing `models/checkpoints/head.pt`.
* **Stage 5 (ONNX Export & Quantization):** `taxon_vision.pipelines.export_onnx` compiles PyTorch weights to ONNX with INT8 dynamic/static quantization into `models/checkpoints/model_int8.onnx`.
* **Stage 6 (Conformal & OOD Calibration):** `taxon_vision.pipelines.calibrate` calibrates Split Conformal quantiles ($\hat{q}$) and Helmholtz OOD energy thresholds ($\tau_{\text{ood}}$) on calibration embeddings, serializing to `config/conformal_config.yaml`.
* **Stage 7 (Pareto Benchmark & Evaluation):** `taxon_vision.pipelines.evaluate` benchmarks accuracy, latency, and conformal coverage guarantees ($1 - \alpha = 0.95$) on holdout test embeddings, exporting `reports/pareto_evaluation.json`.

---

## Repository Governance

The architecture is governed by a 12-role agent matrix defined within `.agents/AGENTS.md` and standardises upon the Google Open Knowledge Format (OKF v0.2) across the `docs/` directory. Configuration hyperparameters are canonically tracked in root `params.yaml`, with system runtime defaults defined in `config/default_config.yaml`.
