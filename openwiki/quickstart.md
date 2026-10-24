---
type: guide
title: Quickstart & Developer Guide
description: Developer onboarding, environment bootstrapping with Pixi, Just recipe automation, repository layout, configuration hierarchy, and task routing for TaxonVision-MLOps.
tags: [quickstart, developer-guide, onboarding, pixi, justfile, workflow, configuration, task-routing]
verified:
  - by: openwiki/0.7.2
    at: 2026-10-15T19:46:36.219Z
sources:
  - id: openwiki-source-978a014826b061d298cf1c14
    resource: repo://config/default_config.yaml
  - id: openwiki-source-14e6a4e1ddc06973b783ff20
    resource: repo://Justfile
  - id: openwiki-source-05ccef8d4cf1698187f20464
    resource: repo://pyproject.toml
  - id: openwiki-source-12914d3b78b4a80693643804
    resource: repo://scripts/ci/local_ci.sh
  - id: openwiki-source-7079947a60e14bc71bc44a0e
    resource: repo://src/taxon_vision/config/__init__.py
  - id: openwiki-source-f0a6e7dc03522b2682f88655
    resource: repo://tests/conftest.py
  - id: openwiki-source-60b001159f253a7d5bdf9f0f
    resource: repo://tests/invariants/test_pure_pixi_invariant.py
generated: { by: "openwiki/0.7.2", at: "2026-10-15T19:46:36.219Z" }
---

# Quickstart & Developer Guide

TaxonVision-MLOps is an automated machine learning operations platform engineered for high-throughput biological species classification from photographic occurrences (such as iNaturalist and GBIF DarwinCore). Operating under strict reliability, open-science attribution, and biodiversity safety constraints, the system augments deep vision foundation representations with statistical conformal prediction sets and Helmholtz free-energy Out-of-Distribution (OOD) detection.

This guide provides developers and machine learning engineers with everything needed to bootstrap a local development environment, execute the inner development loop, run quality gates, train and optimize models, and navigate the repository architecture.

For architectural design and workflow deep-dives, see:

* [System Architecture & Design](architecture/system-overview.md)
* [Domain Entities & Taxonomy Perimeter](architecture/domain-and-taxa.md)
* [Data Ingestion & Dataset Versioning](workflows/data-ingestion-and-versioning.md)
* [Model Training & Experiment Tracking](workflows/model-training-and-tracking.md)
* [Inference Optimization & ONNX Runtime](workflows/inference-and-optimization.md)
* [Uncertainty Quantification & Safety](concepts/uncertainty-and-safety.md)
* [API Service & Web Interface](operations/service-and-api.md)
* [Deployment, Scheduling & Observability](operations/deployment-and-observability.md)
* [Testing Strategy & Quality Invariants](testing/test-strategy-and-invariants.md)

---

## Prerequisites & System Requirements

Before setting up the repository, ensure the following core tools are installed on your host system:

* **Pixi Package Manager**: Pixi is the single authoritative package, environment, and task manager for the repository. Install via:
    ```bash
    curl -fsSL https://pixi.sh/install.sh | bash
    ```
* **Just Command Runner**: Just serves as the automation entrypoint for all daily engineering tasks. Install via:
    ```bash
    cargo install just
    # or globally via Pixi:
    pixi global install just
    ```
* **Operating System**: Linux x86_64 (Ubuntu 22.04 LTS or newer recommended). macOS and Windows WSL2 are supported for CPU-only development and testing.
* **Hardware Acceleration (Optional for Dev, Recommended for Training)**: NVIDIA GPU with CUDA 12.x drivers. When CUDA is unavailable, Pixi automatically falls back to CPU-only PyTorch (`cpu` feature environment).
* **Linux Secret Service (Optional for Cloud Remote Sync)**: When pulling/pushing datasets to DagsHub via DVC or logging runs to remote MLflow, credentials can be automatically retrieved from `secret-tool` (KeePassXC or GNOME Keyring).

### Pure Pixi Workspace Invariant

TaxonVision strictly adheres to a **Pure Pixi Invariant**. The repository prohibits standard Python virtual environments (`.venv/`) and alternate package lockfiles (`uv.lock`). All Conda packages and PyPI wheels are declared in `pyproject.toml` and locked deterministically in `pixi.lock`. IDEs such as VSCode are configured to route directly to `.pixi/envs/dev/bin/python`. This invariant is formally enforced in CI and local test passes via `tests/invariants/test_pure_pixi_invariant.py` and `scripts/local_ci.sh`.

---

## Developer Onboarding & Bootstrapping

Follow these steps to set up a clean, reproducible development environment.

### 1. Clone the Repository

```bash
git clone https://github.com/foersben/taxon-vision-mlops.git
cd taxon-vision-mlops
```

### 2. Environment Bootstrapping

Bootstrap the primary development environment and configure Git pre-commit hooks:

```bash
# Core setup (Pixi dev environment + pre-commit hooks)
just setup
```

To include an isolated Jupyter interactive workspace for exploratory prototyping without dirtying the main Git repository:

```bash
# Core setup + isolated scratch workspace and ipykernel
just setup --scratch
```

#### What `just setup --scratch` Does

The `--scratch` setup recipe automates the following actions:

* Resolves and installs all Conda and PyPI dependencies into `.pixi/envs/dev`.
* Installs active Git pre-commit hooks (`pre-commit install`).
* Initializes an independent Git repository within `scratch/`.
* Configures `nbstripout` clean and smudge filters in `scratch/.gitattributes` to automatically strip output cells and metadata before commits.
* Registers a dedicated user Jupyter kernel named `taxon-scratch` ("TaxonVision (Scratch)") linked to the Pixi `dev` Python interpreter.

### 3. Entering the Development Shell

To spawn a subshell pre-configured with all project environment variables, library paths, and Pixi toolings:

```bash
just init
# Equivalent to: pixi shell --manifest-path ./pyproject.toml -e dev
```

### 4. Interactive Notebook Prototyping

Launch JupyterLab within the isolated `scratch/` workspace:

```bash
just lab
```

This launches JupyterLab on `http://127.0.0.1:8888` scoped strictly to `./scratch`, ensuring experimental notebooks remain completely separated from version-controlled source modules.

---

## Onboarding & Workflow Lifecycle

The following diagram illustrates the developer journey from initial workspace initialization through code verification, local CI gating, and model deployment:

<!-- openwiki: mermaid parse failed and this diagram was converted to a text fence so it does not break rendering. Fix the diagram source and restore the mermaid fence. Parser error: Heuristic: an unescaped angle bracket inside a label breaks rendering; rephrase the label. -->
```text
flowchart TD
    subgraph S_BOOT["1. Workspace Bootstrapping"]
        CLONE["git clone taxon-vision-mlops"]
        SETUP["just setup --scratch"]
        PIXI["pixi install -e dev<br/>pre-commit install"]
        SCRATCH["scratch/ git init<br/>nbstripout & ipykernel"]
        
        CLONE --> SETUP
        SETUP --> PIXI
        SETUP --> SCRATCH
    end

    subgraph S_DEV["2. Inner Development Loop"]
        SHELL["just init (pixi shell)"]
        EDIT["Author Code, Tests, & Config"]
        FMT["just lint (Ruff + MyPy)"]
        TEST["just test (Pytest & Invariants)"]
        
        PIXI --> SHELL
        SHELL --> EDIT
        EDIT --> FMT
        FMT --> TEST
    end

    subgraph S_CI["3. Verification & Gating"]
        LOC_CI["just ci (scripts/local_ci.sh)"]
        OKF["just validate-okf"]
        CHECK["just check (pre-commit all)"]
        
        TEST --> LOC_CI
        LOC_CI --> OKF
        LOC_CI --> CHECK
    end

    subgraph S_APP["4. Execution & Delivery"]
        SERV["just run (FastAPI & HTMX)"]
        TRAIN["just train (Class-Balanced Head)"]
        ONNX["just export-onnx"]
        BENCH["just pareto"]
        
        TEST --> SERV
        TEST --> TRAIN
        TRAIN --> ONNX
        ONNX --> BENCH
    end
```

Developer onboarding and daily execution workflow across bootstrapping, quality gating, testing, and service deployment.

---

## Repository Architecture & Directory Layout

The codebase is organized into decoupled domain modules, operational scripts, declarative configuration files, and deployment manifests:

```text
taxon-vision-mlops/
├── .agents/                 # Role definitions and multi-agent coordination governance
├── .pixi/                   # Pixi-managed isolated virtual environments and binary packages
├── config/                  # Single Source of Truth (SSOT) declarative configurations
│   ├── default_config.yaml  # Base application parameters and model hyperparameters
│   ├── taxa_catalog.yaml    # 10-taxa perimeter catalog with DarwinCore IDs and licenses
│   ├── params.yaml          # DVC and pipeline tuning hyperparameters
│   ├── conformal_config.yaml# Calibrated conformal non-conformity threshold (q_hat)
│   └── prometheus.yaml      # Telemetry scraping definitions for local observability
├── data/                    # Data artifacts (managed locally or tracked via DVC)
│   ├── manifests/           # Versioned Parquet observation manifests
│   ├── sample/              # Sample images for testing and static calibration
│   └── flagged_observations/# Triage storage for ambiguous observations and OOD samples
├── deploy/                  # Cloud-native deployment configurations
│   └── k8s/                 # Kubernetes manifests (FastAPI deployment, GPU time-slicing)
├── docs/                    # Open Knowledge Format (OKF v0.2) architecture documentation
├── models/                  # Checkpoints and serialized graphs
│   └── checkpoints/         # Trained PyTorch linear heads and quantized ONNX models
├── openwiki/                # OpenWiki engineering knowledge base
├── scratch/                 # Isolated workspace for interactive Jupyter experimentation
├── scripts/                 # Automation utilities (ingestion, calibration, benchmarks, audits)
│   ├── ingest_data.py       # Ingests observations from iNaturalist to Parquet
│   ├── download_samples.py  # Fetches verified open-license sample imagery
│   ├── calibrate_conformal.py# Calibrates split conformal prediction quantile q_hat
│   ├── benchmark_pareto.py  # Evaluates latency, accuracy, and cost trade-offs
│   ├── audit_license_compliance.py # Audits dataset manifests for open licenses
│   ├── validate_okf.py      # Validates OKF frontmatter and cross-document link integrity
│   ├── verify_markdown_rules.py    # Enforces strict Rule 04 Markdown formatting
│   ├── verify_deployment.sh # Validates live Kubernetes service health and inference
│   └── local_ci.sh          # Native host runner executing complete multi-pass CI checks
├── src/taxon_vision/        # Core application package
│   ├── active_learning/     # Uncertainty triage and diversity acquisition sampling
│   ├── config.py            # Pydantic BaseSettings models and SSOT YAML loader
│   ├── data/                # iNaturalist API client, license filter, and S3 streamer
│   ├── domain/              # DarwinCore entities, TaxonNode, Observation, and License
│   ├── inference/           # ONNX engine, Grad-CAM visualizer, and exporter
│   ├── models/              # Vision backbones, representation caching, and head trainer
│   ├── monitoring/          # Prometheus metrics exporter and drift telemetry
│   ├── pipelines/           # Training and inference orchestration pipelines
│   ├── service/             # FastAPI REST endpoints, HTMX dashboard, and file storage
│   └── uncertainty/         # Split conformal prediction and Helmholtz free energy OOD
├── tests/                   # Testing pyramid suites
│   ├── conftest.py          # Shared fixtures and offline deterministic isolation hooks
│   ├── unit/                # Fast unit tests mocking models and network calls
│   ├── integration/         # Multi-module integration tests and FastAPI endpoint tests
│   └── invariants/          # Mathematical coverage and architectural invariant suites
├── dvc.yaml                 # DVC pipeline stages (ingestion, train, export_onnx)
├── Justfile                 # Central command automation catalog
├── pixi.lock                # Deterministic cross-platform package lockfile
└── pyproject.toml           # Packaging manifest, Pixi environments, and tool configs
```

---

## Daily Developer Command Reference

The `Justfile` defines recipes grouped into logical engineering domains. Run `just` or `just --list` to view all available commands.

### Setup & Workspace Management

| Command | Description |
|---|---|
| `just setup` | Installs Pixi `dev` environment and installs Git pre-commit hooks. |
| `just setup --scratch` | Bootstraps dev environment, Git pre-commit hooks, isolated `scratch/` repo, and `taxon-scratch` ipykernel. |
| `just setup-scratch` | Alias for `just setup --scratch`. |
| `just init` | Launches an interactive Pixi subshell (`pixi shell -e dev`). |
| `just clean` | Cleans Python bytecode caches (`__pycache__`), test caches, build artifacts, and temporary directories. |

### Code Quality & Static Analysis

| Command | Description |
|---|---|
| `just lint` | Runs Ruff linter with autofix, Ruff code formatter, and MyPy strict type checking. |
| `just format` | Formats all Python codebase files using Ruff. |
| `just check` | Runs all Git pre-commit quality hooks across every file in the repository. |
| `just complexity` | Analyzes cyclomatic and cognitive complexity metrics via `complexipy`. |
| `just validate-okf` | Validates Open Knowledge Format (OKF v0.2) frontmatter, link topologies, and attestation rules. |
| `just lint-md` | Validates and fixes Markdown Rule 04 formatting invariants (ASCII hyphens, asterisk lists). |

### Testing & Invariant Verification

| Command | Description |
|---|---|
| `just test` | Runs the full test suite with Pytest and generates terminal and HTML coverage reports. |
| `just test-unit` | Runs fast unit tests only (`tests/unit/`). |
| `just test-integration` | Runs integration tests validating multi-component workflows and FastAPI routes (`tests/integration/`). |
| `just test-conformal` | Verifies mathematical $1 - \alpha$ finite-sample coverage guarantees on holdout distributions (`tests/invariants/test_conformal_coverage_invariant.py`). |
| `just ci [job]` | Executes local CI runner (`scripts/local_ci.sh`) without Docker container overhead. Options: `all`, `quality`, `tests`, `docs`. |
| `just ci-test` | Alias for `just ci tests`. |

### Data Ingestion & MLOps Pipelines

| Command | Description |
|---|---|
| `just ingest-data` | Ingests research-grade observation records from iNaturalist into versioned Parquet manifests. |
| `just fetch-sample` | Streams a sample batch of observation images from AWS Open Data S3 into `data/sample/`. |
| `just download-samples [per_taxon]` | Downloads verified open-license sample images for all 10 catalog taxa (defaults to 1 per taxon). |
| `just train [epochs] [extractor]` | Trains the classification head using frozen backbone representation caching and Class-Balanced Loss. |
| `just train-baseline [extractor]` | Quick baseline training run using `bioclip-2` for 5 epochs. |
| `just export-onnx` | Exports the PyTorch vision backbone and linear classification head to an optimized ONNX computational graph. |
| `just pareto` | Benchmarks Pareto frontier trade-offs (Accuracy vs Latency vs Cost) across candidate backbones. |
| `just calibrate` | Computes split conformal non-conformity quantile $\hat{q}$ on holdout validation data ($\alpha = 0.05$). |
| `just predict [image]` | Runs CLI inference, conformal set generation, and OOD energy evaluation on a target image file. |
| `just dvc-push` | Authenticates via Linux Secret Service and pushes versioned dataset manifests and models to DagsHub. |
| `just dvc-pull` | Pulls tracked dataset manifests and model checkpoints from DagsHub remote S3 storage. |

### Application Serving & UI

| Command | Description |
|---|---|
| `just run` | Starts the unified FastAPI web service and HTMX diagnostic dashboard with live reloading on `http://localhost:8000`. |
| `just run-api` | Direct invocation of Uvicorn server running `taxon_vision.service.api:app`. |
| `just verify-deployment [namespace]`| Runs the automated Kubernetes verification script (`scripts/verify_deployment.sh`) to test live endpoints and inference. |

### Documentation & Visualization

| Command | Description |
|---|---|
| `just docs` | Builds static documentation site using Zensical. |
| `just serve` | Builds and serves the Zensical documentation site locally on `http://localhost:9000`. |
| `just visualize-okf` | Generates an interactive D3/SVG knowledge graph visualization from OKF documentation frontmatter. |
| `just lab` | Starts JupyterLab inside the isolated `scratch/` workspace on `http://127.0.0.1:8888`. |

---

## Configuration Hierarchy & Runtime Invariants

TaxonVision maintains a deterministic configuration architecture preventing configuration drift between declarative defaults, pipeline stages, and deployed containers.

### Single Source of Truth (SSOT) Precedence

Configuration settings are resolved by `taxon_vision.config.get_settings()` through a strict six-tier hierarchy:

1. **Base Declarative Defaults**: `config/default_config.yaml` declares system defaults for web serving, vision backbones, calibration parameters, and OOD thresholds.
2. **Pipeline Hyperparameters**: Root `params.yaml` overlays training epochs, batch sizes, and learning rates (synchronized with DVC stages).
3. **Conformal Quantiles**: `config/conformal_config.yaml` overlays calibrated non-conformity threshold values ($\hat{q}$ and $k_{\max}$).
4. **Custom YAML Overrides**: Specified via the `TAXON_CONFIG_PATH` environment variable or an explicit filesystem argument.
5. **Direct Environment Overrides**: Targeted runtime environment variables for test execution and offline sandboxing:
    * `TAXON_ALLOW_UNTRAINED_FALLBACK=true`: Permits uninitialized model weights during offline test suites without network downloads.
    * `TAXON_FLAGGED_STORAGE_DIR=/path/to/dir`: Redirects OOD and ambiguous observation triage payloads.
6. **Nested Environment Variables**: Overrides any configuration field using the `TAXON_` prefix and double underscore `__` delimiter:
    ```bash
    export TAXON_MODEL__IMAGE_SIZE=336
    export TAXON_CONFORMAL__ALPHA=0.10
    export TAXON_SERVICE__PORT=8080
    ```

Every setting is strictly validated at runtime using Pydantic `BaseSettings` models (`ServiceConfig`, `ModelConfig`, `TrainingConfig`, `ConformalConfig`, `OODConfig`, `MLflowConfig`).

### Critical Runtime & Testing Invariants

When developing or modifying features, preserve these key architectural invariants:

* **Pure Pixi Workspace**: No `.venv/` directories or `uv.lock` files may exist in the workspace root. All dependencies must be managed via Pixi.
* **Deterministic Offline Test Isolation**: `tests/conftest.py` declares an autouse fixture setting `TAXON_ALLOW_UNTRAINED_FALLBACK="true"` and clearing Pydantic configuration caches (`get_settings.cache_clear()`). Tests must execute 100% offline without attempting to fetch Hugging Face or timm checkpoints.
* **Conformal Mathematical Coverage**: Any changes to `taxon_vision.uncertainty.conformal` must maintain finite-sample coverage guarantees ($P(Y \in C(X)) \ge 1 - \alpha$) verified by `tests/invariants/test_conformal_coverage_invariant.py`.
* **Markdown Formatting (Rule 04)**: All Markdown documentation must use standard ASCII hyphens (`-`), asterisk list markers (`*`) indented by multiples of 4 spaces, single blank lines around fences and headings, and zero trailing whitespace (`just lint-md`).

---

## Comprehensive Task-Routing Guide

Use this task-routing map to navigate directly to the appropriate documentation and source components for your development goals:

| Engineering Objective | Primary Wiki Guide | Key Files & Modules | Core Commands |
|---|---|---|---|
| **Taxonomy Perimeter & Species Catalog** | [Domain Entities & Taxonomy Perimeter](architecture/domain-and-taxa.md) | `config/taxa_catalog.yaml`<br/>`src/taxon_vision/domain/` | `just download-samples` |
| **System Architecture & Dual Pipeline** | [System Architecture & Design](architecture/system-overview.md) | `src/taxon_vision/config.py`<br/>`config/default_config.yaml` | `just init`<br/>`just clean` |
| **Data Ingestion, S3 Streaming, & DVC** | [Data Ingestion & Dataset Versioning](workflows/data-ingestion-and-versioning.md) | `scripts/ingest_data.py`<br/>`src/taxon_vision/data/`<br/>`dvc.yaml` | `just ingest-data`<br/>`just fetch-sample`<br/>`just dvc-push` |
| **Model Training & Experiment Tracking** | [Model Training & Experiment Tracking](workflows/model-training-and-tracking.md) | `src/taxon_vision/models/`<br/>`src/taxon_vision/pipelines/` | `just train`<br/>`just train-baseline` |
| **ONNX Export, Quantization, & Pareto** | [Inference Optimization & ONNX Runtime](workflows/inference-and-optimization.md) | `src/taxon_vision/inference/onnx_exporter.py`<br/>`scripts/benchmark_pareto.py` | `just export-onnx`<br/>`just pareto` |
| **Conformal Prediction & Safety Triage** | [Uncertainty Quantification & Safety](concepts/uncertainty-and-safety.md) | `src/taxon_vision/uncertainty/`<br/>`scripts/calibrate_conformal.py` | `just calibrate`<br/>`just test-conformal` |
| **FastAPI Service & HTMX Dashboard** | [API Service & Web Interface](operations/service-and-api.md) | `src/taxon_vision/service/`<br/>`src/taxon_vision/inference/` | `just run`<br/>`just predict` |
| **Kubernetes, Telemetry, & Runbooks** | [Deployment, Scheduling & Observability](operations/deployment-and-observability.md) | `deploy/k8s/`<br/>`src/taxon_vision/monitoring/` | `just verify-deployment` |
| **Quality Gates, Pytest, & Invariants** | [Testing Strategy & Quality Invariants](testing/test-strategy-and-invariants.md) | `tests/`<br/>`scripts/local_ci.sh`<br/>`scripts/governance/validate_okf.py` | `just test`<br/>`just lint`<br/>`just ci` |

### Detailed Task Routing

#### 1. Ingesting Biological Data & Updating Datasets
To modify the species perimeter, download new observations, or stream images from S3:
* Review biological entities and licensing attribution in [Domain Entities & Taxonomy Perimeter](architecture/domain-and-taxa.md).
* Follow the ingestion workflow in [Data Ingestion & Dataset Versioning](workflows/data-ingestion-and-versioning.md) to query iNaturalist API endpoints and generate immutable Parquet manifests.
* Track datasets with DVC and push to remote DagsHub storage using `just dvc-push`.

#### 2. Training Representation Heads & Tracking Experiments
To train classifier heads on top of frozen vision backbones (`BioCLIP-2`, `DINOv3`, `MobileNetV4`):
* Consult [Model Training & Experiment Tracking](workflows/model-training-and-tracking.md) for details on embedding caching and Class-Balanced Loss.
* Configure hyperparameters in root `params.yaml` and execute `just train epochs=10 extractor=dinov3`.
* Metrics, confusion matrices, and model artifacts are automatically tracked to MLflow on DagsHub.

#### 3. Optimizing Inference with ONNX Runtime & INT8 Quantization
To convert PyTorch weights to high-throughput ONNX computational graphs:
* Consult [Inference Optimization & ONNX Runtime](workflows/inference-and-optimization.md) for graph architecture details, dynamic vs static calibration, and dual-output heads (logits + Grad-CAM feature activations).
* Run `just export-onnx` to produce `models/checkpoints/model_int8.onnx`.
* Run `just pareto` to evaluate latency, throughput, and memory consumption across backbones.

#### 4. Calibrating Statistical Uncertainty & OOD Detection
To adjust conformal coverage error bounds or free-energy thresholds:
* Read [Uncertainty Quantification & Safety](concepts/uncertainty-and-safety.md) to understand non-conformity score calculation, adaptive prediction sets ($C(X)$), and energy-based out-of-distribution detection.
* Execute `just calibrate` to recalculate $\hat{q}$ across holdout validation splits.
* Verify coverage bounds with `just test-conformal`.

#### 5. Developing & Running the Web Service
To test the FastAPI REST API, live prediction endpoints, or the HTMX dashboard:
* See [API Service & Web Interface](operations/service-and-api.md) for OpenAPI specification details, multipart image upload formats, and Grad-CAM overlay generation.
* Run `just run` to launch the Uvicorn server with live reloading on `http://localhost:8000`.
* Test image classification directly from the terminal with `just predict image=data/sample/406185885_taxon_47120.jpg`.

#### 6. Deploying to Kubernetes & Monitoring Production
To inspect Kubernetes deployment topologies or set up observability:
* See [Deployment, Scheduling & Observability](operations/deployment-and-observability.md) for pod topologies, NVIDIA GPU time-slicing configs, Prometheus scraping, and operational runbooks.
* Verify live cluster deployments with `just verify-deployment namespace=taxon-vision`.
