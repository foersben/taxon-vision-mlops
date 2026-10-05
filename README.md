# TaxonVision-MLOps: Automated Species Identification Service

[![CI & Deployment Pipeline](https://github.com/foersben/taxon-vision-mlops/actions/workflows/ci.yml/badge.svg)](https://github.com/foersben/taxon-vision-mlops/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Pixi](https://img.shields.io/badge/managed%20by-pixi-blue)](https://pixi.sh)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)

This repository provides a highly formalised, end-to-end MLOps pipeline and high-throughput production service for the purpose of species identification from citizen-science photographs (such as iNaturalist and GBIF DarwinCore). It has been engineered with rigorous attention to deterministic execution and state-of-the-art inference efficiency.

```text
    ┌────────────────────────────────────────────────────────────────────────────────────────┐
    │                                  TAXONVISION-MLOPS                                     │
    │         Automated End-to-End MLOps Pipeline for Species Identification from Photos     │
    │              Citizen Science (iNaturalist / GBIF) - Conformal Uncertainty              │
    │             Production-Grade Engineering - Dual-Target Pixi (GPU / CPU)                │
    └────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Key Capabilities

* **Level 1 Closed Chain:** Ingestion of approximately 10 well-represented taxa, strict open license filtering (CC0, CC-BY, CC-BY-NC), attribution preservation, frozen feature extraction, and automated deployment.
* **Level 2 Scalability & Cost:** Automated multi-backbone Pareto benchmarking (`BioCLIP-2`, `DINOv3`, `DINOv2`, `MobileNetV4`, `EfficientNet`), ONNX Runtime export with dynamic INT8/FP16 quantisation, and rigorous DVC dataset versioning.
* **Level 3 Observability:** Automated feedback ingestion, taxonomic mutation reconciliation, and structured Prometheus telemetry.
* **Level 4 Robustness & Safety:** **Split Conformal Prediction** mathematically guaranteeing distribution-free $1 - \alpha$ coverage, energy-based OOD detection, active learning triage, Class-Balanced Loss for long-tail species, and low-latency Grad-CAM heatmaps (< 25 ms).

---

## Quickstart

### Prerequisites

* [Pixi](https://pixi.sh) installed (`curl -fsSL https://pixi.sh/install.sh | bash`)
* [Just](https://github.com/casey/just) installed (`cargo install just` or `pixi global install just`)

### 1. Bootstrapping

```bash
# Clone the repository
git clone https://github.com/foersben/taxon-vision-mlops.git
cd taxon-vision-mlops

# Setup environment, pre-commit hooks, and scratch notebook space
just setup --scratch
```

### 2. Quality & Tests

```bash
# Run linting and type checks
just lint

# Run unit and integration tests
just test

# Validate Open Knowledge Format (OKF v0.2) frontmatter
just validate-okf
```

### 3. Running the Service

```bash
# Launch unified FastAPI + HTMX service on http://localhost:8000
just run
```

---

## Repository Governance

The architecture is governed by a 12-role agent matrix defined within `.agents/AGENTS.md` and standardises upon the Google Open Knowledge Format (OKF v0.2) across the `docs/` directory.
