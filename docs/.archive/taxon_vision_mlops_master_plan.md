# TaxonVision-MLOps: Master Repository Blueprint & Architecture Plan

## Executive Project Charter & Scientific Identity

```
    ┌────────────────────────────────────────────────────────────────────────────────────────┐
    │                                  TAXONVISION-MLOPS                                     │
    │         Automated End-to-End MLOps Pipeline for Species Identification from Photos     │
    │              Citizen Science (iNaturalist / GBIF) • Conformal Uncertainty             │
    │             Production-Grade Engineering • Dual-Target Pixi (GPU / CPU)               │
    └────────────────────────────────────────────────────────────────────────────────────────┘
```

* **Project Name:** `taxon-vision-mlops` (Alternative candidates: `species-id-mlops`, `inat-vision-ops`)
* **Target Domain:** Automated citizen science species identification pipeline using open-licensed photographs (iNaturalist, GBIF DarwinCore).
* **Course Context:** MLOps (Difficulty 08/10, Liora ex DataScientest).
* **Core Engineering Philosophy:** *"The project focuses on the production chain, not the raw performance of the model. A simple, versioned, monitored, deployed, and automatically retrainable model is infinitely better than a sophisticated model left in a notebook."*
* **Architecture Foundation:**
    * **Repository Blueprint & Governance:** Derived from [`foersben/PHIDS`](file:///home/benni/Documents/antigravity_workspace/PHIDS) (OKF v0.2 knowledge graph, 11 specialized agent roles, 6 canned autonomous personas, hardened pre-commit gates, two-pass CI/CD, and Zensical documentation).
    * **Environment & Package Management:** Derived from [`foersben/agentic-market-panic-sim`](file:///home/benni/Documents/antigravity_workspace/agentic-market-panic-sim) (Pixi multi-environment architecture with CUDA 12.x GPU and CPU solvers, PyTorch wheel segregation, scratch notebook hygiene, and Justfile automation).

---

## 1. End-to-End System Architecture Across Difficulty Levels

The repository is architected to address the 4 difficulty tiers specified in the project charter:

```mermaid
flowchart TD
    subgraph S1["Level 1: Ingestion & Closed Chain"]
        A1["iNaturalist API v2 / AWS S3 Open Data"] --> B1["License & Attribution Filter\n(CC0, CC-BY, CC-BY-NC vs Static)"]
        B1 --> C1["DarwinCore Observation Registry\n(Parquet / DuckDB / DVC)"]
        C1 --> D1["Timm Backbone Factory\n(Frozen BioCLIP-2 / DINOv2 / MobileNet)"]
        D1 --> E1["Trained Classification Head"]
        E1 --> F1["FastAPI Prediction Service"]
    end

    subgraph S2["Level 2: Scalability, Pareto & Cost"]
        G2["100+ Taxa Expansion"] --> H2["Pareto Benchmark Engine\n(Accuracy vs Latency vs $/Query)"]
        H2 --> I2["Inference Engine Optimization\n(ONNX Runtime / INT8 / FP16 / Zero-Alloc)"]
        I2 --> J2["DVC Large-Scale Dataset Versioning\n(S3 Remote / SHA256 Verification)"]
    end

    subgraph S3["Level 3: Observability & Feedback Loop"]
        K3["POST /api/v1/feedback\n(Validated Identifications)"] --> L3["Taxonomic Revision Engine\n(Lumping / Splitting / Ground Truth Mutations)"]
        F1 --> M3["Prometheus Telemetry & Evidently AI\n(Concept Drift / Embedding Drift / Latency)"]
        L3 --> N3["Automated Retraining Trigger"]
    end

    subgraph S4["Level 4: Robustness, Uncertainty & Cold Start"]
        F1 --> O4["Conformal Prediction Engine\n(Angelopoulos & Bates, 2021)"]
        O4 -->|Set Size <= k_max| P4["Calibrated Prediction Set\n(Guaranteed Error Rate 1 - alpha)"]
        O4 -->|Empty or Ambiguous| Q4["Human-in-the-Loop Review Queue"]
        F1 --> R4["Energy-Based OOD Detector\n(Empty Photos / Unusable Frames)"]
        R4 -->|OOD Flagged| Q4
        Q4 --> S4["Active Learning Selector\n(BADGE / CoreSet / Margin Sampling)"]
        S4 --> N3
        E1 --> T4["Class-Balanced Loss (Cui et al., 2019)\n(Long-Tail Rare Species)"]
        D1 --> U4["Few-Shot Centroid Cold Start\n(New Taxon Enrollment in <= 5 shots)"]
        I2 --> V4["Low-Latency Grad-CAM Heatmaps\n(XAI Visual Attribution < 25ms)"]
    end

    E1 -.-> I2
    I2 -.-> F1
    N3 -.-> D1
```

### Detailed Functional Mapping by Level

| Level | Scope | Architectural Implementation | Primary Module |
| --- | --- | --- | --- |
| **Level 1** | Restricted Perimeter & Full Chain | Ingestion of ~10 well-represented taxa; open-license filter (CC0, CC-BY, CC-BY-NC) retaining photographer attribution; fixed pre-trained extractor with simple head; closed loop deployment via FastAPI. | `src/taxon_vision/data/license_filter.py`<br>`src/taxon_vision/models/head.py`<br>`src/taxon_vision/service/api.py` |
| **Level 2** | Scalability & Cost of Service | Expansion to 100+ taxa; Pareto evaluation matrix relating accuracy to service cost ($/M queries, VRAM footprint, p95 latency); ONNX Runtime export with INT8 quantization; DVC dataset versioning on S3. | `src/taxon_vision/benchmarks/pareto.py`<br>`src/taxon_vision/inference/onnx_engine.py`<br>`src/taxon_vision/data/dvc_pipeline.py` |
| **Level 3** | Feedback Loop & Observability | Post-prediction consensus collection; label revision management (handling taxonomic mutations and species reclassifications without pipeline breakage); Prometheus metrics and embedding drift detection. | `src/taxon_vision/monitoring/drift.py`<br>`src/taxon_vision/feedback/reconciliation.py`<br>`src/taxon_vision/monitoring/telemetry.py` |
| **Level 4** | Robustness, Cold Start & Safety | Split Conformal Prediction for distribution-free error guarantees ($1 - \alpha$ coverage); energy-based Out-Of-Distribution (OOD) detector for empty/garbage images; automated human review referral; few-shot prototype enrollment for rare species; Class-Balanced Loss; low-latency Grad-CAM explanations. | `src/taxon_vision/uncertainty/conformal.py`<br>`src/taxon_vision/uncertainty/ood_detector.py`<br>`src/taxon_vision/active_learning/query.py`<br>`src/taxon_vision/inference/gradcam.py` |

---

## 2. Pixi Multi-Environment & Dependency Matrix

Incorporating the battle-tested configuration from `agentic-market-panic-sim`, the project utilizes Pixi with conda-forge and PyTorch wheel indexes, providing isolated environments for GPU development, CPU CI, and local experimentation.

### `pyproject.toml` Pixi Specification

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "taxon-vision-mlops"
version = "0.1.0"
description = "Automated end-to-end MLOps pipeline for species identification from photographs."
authors = [{ name = "Benjamin Förster", email = "mail@benjamin-foerster.de" }]
readme = "README.md"
requires-python = ">=3.12"
dependencies = [
    "fastapi>=0.115.0",
    "uvicorn[standard]>=0.34.0",
    "pydantic>=2.10.0",
    "pydantic-settings>=2.7.0",
    "timm>=1.0.14",
    "numpy>=1.26.0,<2.0.0",
    "pillow>=11.0.0",
    "polars>=1.20.0",
    "duckdb>=1.1.0",
    "scikit-learn>=1.6.0",
    "scipy>=1.14.0",
    "httpx>=0.28.0",
    "jinja2>=3.1.5",
    "python-multipart>=0.0.20",
    "dvc>=3.58.0",
    "dvc-s3>=3.2.0",
    "onnxruntime>=1.20.0",
    "prometheus-client>=0.21.0",
]

[dependency-groups]
dev = [
    "pytest>=8.3.0",
    "pytest-cov>=6.0.0",
    "pytest-asyncio>=0.25.0",
    "pytest-benchmark>=5.0.0",
    "pytest-mock>=3.14.0",
    "hypothesis>=6.120.0",
    "ruff>=0.9.0",
    "mypy>=1.14.0",
    "pre-commit>=4.0.0",
    "zensical>=0.0.65",
    "mkdocstrings[python]>=0.27.0",
    "mkdocs-htmlproofer-plugin>=1.5.0",
    "pyyaml>=6.0.2",
    "types-PyYAML>=6.0.12",
    "types-requests>=2.32.0",
    "nbstripout>=0.8.0",
    "complexipy>=8.0.1,<9",
    "evidently>=0.4.30",
]

[tool.pixi.workspace]
channels = ["conda-forge"]
platforms = ["linux-64"]

[tool.pixi.pypi-options]
index-strategy = "unsafe-best-match"
extra-index-urls = [
    "https://download.pytorch.org/whl/cu121",
    "https://download.pytorch.org/whl/cpu",
]

[tool.pixi.pypi-dependencies]
taxon-vision-mlops = { path = ".", editable = true }

[tool.pixi.dependencies]
python = "3.12.*"
just = ">=1.39.0"
jupyter = "*"
jupyterlab = "*"
ipykernel = "*"
pip = "*"
matplotlib = ">=3.9.0"
pandas = ">=2.2.0"
pillow = ">=11.0.0"
requests = ">=2.32.0"
scikit-learn = ">=1.6.0"
pynvml = ">=11.5.0"
optuna = ">=4.1.0"

[tool.pixi.feature.gpu.dependencies]
cuda-version = "12.*"
cuda-libraries = "12.*"
cuda-cupti = "12.*"
cuda-nvcc = "12.*"
cudnn = ">=9.0,<10"

[tool.pixi.feature.gpu.pypi-dependencies]
torch = { version = "==2.5.1", index = "https://download.pytorch.org/whl/cu121" }
torchvision = { version = "==0.20.1", index = "https://download.pytorch.org/whl/cu121" }
torchaudio = { version = "==2.5.1", index = "https://download.pytorch.org/whl/cu121" }
onnxruntime-gpu = ">=1.20.0"

[tool.pixi.feature.cpu.pypi-dependencies]
torch = { version = "==2.5.1", index = "https://download.pytorch.org/whl/cpu" }
torchvision = { version = "==0.20.1", index = "https://download.pytorch.org/whl/cpu" }
torchaudio = { version = "==2.5.1", index = "https://download.pytorch.org/whl/cpu" }

[tool.pixi.environments]
default = { features = ["gpu"], solve-group = "default" }
dev = { features = ["dev", "gpu"], solve-group = "default" }
ci = { features = ["cpu"], solve-group = "ci" }
ci-dev = { features = ["dev", "cpu"], solve-group = "ci" }

[tool.pixi.target.linux-64.activation.env]
LD_LIBRARY_PATH = "$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

[tool.pixi.tasks]
api = { cmd = "uvicorn taxon_vision.service.api:app --host 0.0.0.0 --port 8000 --reload" }
train = { cmd = "python -m taxon_vision.pipelines.train" }
pareto = { cmd = "python scripts/benchmark_pareto.py" }
calibrate = { cmd = "python scripts/calibrate_conformal.py" }
```

---

## 3. Detailed Repository Anatomy & File Layout

```
taxon-vision-mlops/
├── .agents/                                    # Master Multi-Agent Architecture
│   ├── AGENTS.md                               # Universal rules & constraints
│   ├── index.md                                # Algorithmic role routing decision matrix
│   ├── README.md                               # Agent ecosystem documentation
│   ├── PROMPT_TEMPLATE.md                      # Canned cloud persona prompts (Chisel, Bolt, Canon, etc.)
│   ├── log.md                                  # Chronological OKF change log
│   ├── mcp.json                                # Local MCP tool bindings
│   ├── roles/                                  # 11 Specialized Agent Roles
│   │   ├── 01-orchestrator.md                  # PM, milestones, OKF architecture
│   │   ├── 02-taxonomic-architect.md           # DarwinCore schema, iNaturalist taxonomy graph
│   │   ├── 03-mlops-data-engineer.md           # AWS S3 ingestion, DVC pipelines, license filter
│   │   ├── 04-model-engineer.md                # Timm extractors (BioCLIP-2, DINOv2), training loops
│   │   ├── 05-inference-optimizer.md           # ONNX Runtime, quantization, low-latency Grad-CAM
│   │   ├── 06-uncertainty-safety-engineer.md   # Conformal prediction sets, OOD detection, triage
│   │   ├── 07-active-learning-engineer.md      # Query strategies, human-in-the-loop triage queues
│   │   ├── 08-observability-engineer.md        # Prometheus metrics, drift detection, Pareto evaluation
│   │   ├── 09-api-and-ui-developer.md          # FastAPI service, async upload, Jinja2/HTMX UI
│   │   ├── 10-qa-automator.md                  # Deterministic fixtures, hypothesis tests, CI gates
│   │   ├── 11-docs-librarian.md                # Zensical documentation, LaTeX math, OKF metadata
│   │   └── 12-git-operator.md                  # GPG/SSH commit integrity, release workflows
│   ├── rules/                                  # Repository Constraints
│   │   ├── 00-python-modernization.md          # Strict types, ruff, pixi/just execution
│   │   ├── 01-mlops-reproducibility.md         # Deterministic PRNG seeds, DVC immutability
│   │   ├── 02-inference-latency-invariants.md  # Zero allocations on hot path, <50ms budgets
│   │   ├── 03-git-security-and-signing.md      # Mandatory GPG/SSH commit signing
│   │   ├── 04-markdown-formatting.md           # Hyphen standard, list spacing, no trailing whitespace
│   │   ├── 05-open-license-compliance.md       # CC-BY/CC0 validation, attribution retention
│   │   └── 06-conformal-coverage-invariants.md # Empirical coverage test >= 1 - alpha
│   ├── workflows/                              # Slash Command Automated Workflows
│   │   ├── full-pipeline-validation.md         # /validate-full-stack
│   │   ├── model-benchmark-pareto.md           # /benchmark-pareto
│   │   ├── conformal-uncertainty-audit.md      # /audit-conformal
│   │   ├── active-learning-cycle.md            # /active-learning-step
│   │   ├── doc-synchronization-pipeline.md     # /doc-synchronization-pipeline
│   │   └── jules-session-triage.md             # /jules-session-triage
│   └── skills/                                 # Deterministic Agent Skills
│       ├── validate-okf/SKILL.md               # OKF frontmatter validation
│       ├── visualize-okf/SKILL.md              # Interactive knowledge graph compiler
│       ├── audit-license-compliance/SKILL.md   # Open license & attribution auditing
│       ├── run-pareto-benchmarks/SKILL.md      # Extractor cost vs accuracy benchmarking
│       └── verify-conformal-coverage/SKILL.md  # Statistical coverage guarantee testing
│
├── .github/
│   └── workflows/
│       └── ci.yml                              # Hardened 2-pass CI/CD with Pixi and Pages deploy
│
├── config/
│   ├── default_config.yaml                     # Application hyperparameters & thresholds
│   ├── conformal_config.yaml                   # Significance level alpha, error rates, k_max
│   └── taxa_catalog.yaml                       # Level 1 (10 taxa) & Level 2 (100 taxa) taxonomy
│
├── docs/                                       # Zensical Multi-Audience Documentation
│   ├── index.md                                # Landing page & executive summary
│   ├── foundations/
│   │   ├── taxonomy_and_darwincore.md          # Taxonomic trees, GBIF schemas, open licenses
│   │   └── inaturalist_data_ecosystem.md       # API limits, AWS S3 buckets, attribution laws
│   ├── mlops_pipeline/
│   │   ├── ingestion_and_versioning.md         # DVC pipelines, S3 storage strategies
│   │   ├── backbone_benchmarking_pareto.md     # BioCLIP-2 vs DINOv2 vs MobileNet tradeoffs
│   │   └── onnx_optimization_and_latency.md    # INT8 quantization, memory footprints
│   ├── uncertainty_and_safety/
│   │   ├── conformal_prediction_theory.md      # Split conformal math & coverage proofs
│   │   ├── ood_and_empty_frame_detection.md    # Energy scoring, rejection protocols
│   │   └── active_learning_and_hitl.md         # Query strategies, human triage queues
│   ├── operations/
│   │   ├── api_reference.md                    # FastAPI endpoints, request/response models
│   │   ├── prometheus_and_observability.md     # Drift monitoring, Grafana dashboards
│   │   └── deployment_runbook.md               # Docker, Kubernetes, resource quotas
│   └── viz.html                                # Compiled OKF interactive knowledge graph
│
├── scratch/                                    # Local Isolated Experimentation Sandbox
│   ├── .gitignore                              # Prevents untracked pollution
│   ├── README.md                               # Scratch workspace guide
│   └── notebooks/
│       └── 00_quickstart.ipynb                 # Interactive extractor & inference testing
│
├── scripts/                                    # Automation Scripts & Gates
│   ├── audit_license_compliance.py             # Pre-commit hook: asserts CC0/CC-BY & attribution
│   ├── benchmark_pareto.py                     # Generates accuracy vs cost vs latency Pareto curve
│   ├── calibrate_conformal.py                  # Computes non-conformity quantiles on holdout set
│   ├── local_ci.sh                             # Host CI runner matching GitHub Actions
│   ├── run_ci_with_act.sh                      # Local Docker-based workflow rehearsal
│   ├── validate_okf.py                         # Strict OKF v0.2 frontmatter validator
│   └── visualize_okf.py                        # Compiles docs/viz.html knowledge graph
│
├── src/
│   └── taxon_vision/
│       ├── __init__.py
│       ├── domain/                             # Pure Domain Models (DarwinCore & Pydantic)
│       │   ├── observation.py                  # Observation metadata, geo-coordinates, dates
│       │   ├── taxonomy.py                     # Taxonomic rank tree (Kingdom -> Species)
│       │   └── license.py                      # Open license enumerations, attribution records
│       ├── data/                               # Ingestion & Versioning
│       │   ├── inat_client.py                  # iNaturalist API v2 rate-limited async client
│       │   ├── s3_streamer.py                  # Streaming reader for AWS Open Data S3 bucket
│       │   ├── license_filter.py               # Discards static domain, validates open licenses
│       │   └── dvc_pipeline.py                 # DVC automation for Parquet metadata & images
│       ├── models/                             # Backbone Architecture & Training
│       │   ├── factory.py                      # Timm model loader (BioCLIP-2, DINOv2, etc.)
│       │   ├── head.py                         # Frozen backbone + trainable classification head
│       │   ├── loss.py                         # Class-Balanced Loss (Cui et al., 2019)
│       │   └── trainer.py                      # Fast CPU/GPU head fine-tuning loop
│       ├── inference/                          # Low-Latency Runtime Engine
│       │   ├── onnx_exporter.py                # PyTorch -> ONNX dynamic shape export
│       │   ├── onnx_engine.py                  # Zero-allocation ONNX Runtime inference engine
│       │   ├── quantizer.py                    # Dynamic INT8 & FP16 quantization routines
│       │   └── gradcam.py                      # Low-latency visual attribution (<25ms)
│       ├── uncertainty/                        # Conformal Prediction & OOD
│       │   ├── conformal.py                    # Split conformal prediction set generator
│       │   ├── nonconformity.py                # Softmax & temperature-scaled score functions
│       │   └── ood_detector.py                 # Energy-based OOD / empty image detector
│       ├── active_learning/                    # Human-in-the-Loop & Query Engine
│       │   ├── query.py                        # BADGE, CoreSet, and Margin query strategies
│       │   └── triage_queue.py                 # Priority scoring for human expert review
│       ├── feedback/                           # Continuous Feedback Loop
│       │   ├── collector.py                    # Validated observation ingestion
│       │   └── reconciliation.py               # Taxonomic revision & label mutation handler
│       ├── monitoring/                         # Observability & Drift
│       │   ├── telemetry.py                    # Prometheus latency & confidence metrics
│       │   └── drift.py                        # Embedding drift & concept drift detection
│       ├── service/                            # Unified Production API & Web Service
│       │   ├── api.py                          # FastAPI application composition root
│       │   ├── schemas.py                      # Request/response contracts (Pydantic v2)
│       │   ├── routes/
│       │   │   ├── predict.py                  # POST /api/v1/predict (top-k + conformal set)
│       │   │   ├── explain.py                  # POST /api/v1/explain (Grad-CAM heatmap)
│       │   │   ├── feedback.py                 # POST /api/v1/feedback (citizen review submission)
│       │   │   ├── views.py                    # GET / (Jinja2+HTMX web interface & triage dashboard)
│       │   │   └── health.py                   # GET /health, GET /metrics
│       │   ├── templates/                      # Jinja2 SSR HTML Templates
│       │   │   ├── base.html                   # Responsive base layout with HTMX script
│       │   │   ├── index.html                  # Drag-and-drop image submission view
│       │   │   ├── partials/                   # HTMX dynamic swap fragments
│       │   │   │   ├── prediction_card.html    # Swaps in predictions, conformal sets, OOD warnings
│       │   │   │   ├── explain_modal.html      # Grad-CAM overlay heatmap fragment
│       │   │   │   └── feedback_confirm.html   # Review queue acknowledgement
│       │   └── static/                         # Lightweight static assets
│       │       ├── css/style.css               # Clean, modern vanilla CSS
│       │       └── js/htmx.min.js              # Vendor HTMX runtime (no npm build required)
│
├── tests/
│   ├── conftest.py                             # Pytest fixtures & miniature image loaders
│   ├── fixtures/                               # Static test fixtures (no heavy downloads)
│   │   ├── cub_subset/                         # Lightweight CUB-200 sample images
│   │   └── sample_metadata.json                # Synthetic DarwinCore observations
│   ├── unit/                                   # Fast Unit Tests (< 100ms)
│   │   ├── test_license_filter.py              # Asserts static domain rejection & CC retention
│   │   ├── test_taxonomy.py                    # Taxonomic rank resolution
│   │   ├── test_class_balanced_loss.py         # Cui et al. weighting equations
│   │   ├── test_conformal_bounds.py            # Quantile calculation mathematical sanity
│   │   └── test_ood_energy.py                  # Energy score thresholding
│   ├── integration/                            # Component Integration Tests
│   │   ├── test_onnx_parity.py                 # PyTorch vs ONNX numerical parity (< 1e-4)
│   │   ├── test_api_endpoints.py               # FastAPI test client validation
│   │   └── test_feedback_loop.py               # Feedback ingestion and drift logging
│   └── invariants/                             # Scientific & Production Invariants
│       ├── test_conformal_coverage_invariant.py# Empirical coverage >= 1 - alpha on holdout
│       └── test_zero_alloc_inference.py        # Invariant: ONNX engine performs no allocations
│
├── .gitignore
├── .gitattributes
├── .markdownlint.yaml
├── .pre-commit-config.yaml
├── Justfile
├── LICENSE
├── pyproject.toml
├── README.md
└── zensical.toml
```

---

## 4. Hardened Pre-Commit Pipeline Configuration

Preserves the PHIDS security and format enforcement while integrating the Pixi command dispatch from AMPS:

```yaml
minimum_pre_commit_version: "3.7.0"
default_stages: [pre-commit]
exclude: ^(site/|docs/viz\.html|\.cache/|\.pixi/|scratch/notebooks/)

repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v6.0.0
    hooks:
      - id: trailing-whitespace
        exclude: ^(\.agents/|docs/)
      - id: end-of-file-fixer
      - id: mixed-line-ending
        args: [--fix=lf]
      - id: check-yaml
        args: [--unsafe]
      - id: check-toml
      - id: check-added-large-files
        args: ['--maxkb=5000']
      - id: check-merge-conflict
      - id: check-case-conflict

  - repo: https://github.com/gitleaks/gitleaks
    rev: v8.30.1
    hooks:
      - id: gitleaks

  - repo: https://github.com/igorshubovych/markdownlint-cli
    rev: v0.43.0
    hooks:
      - id: markdownlint
        args: ["--fix"]

  - repo: https://github.com/codespell-project/codespell
    rev: v2.4.3
    hooks:
      - id: codespell
        additional_dependencies: [tomli]
        args: ["--skip", "*.lock,*.json,./site/*,./docs/viz.html", "-L", "taxa,taxon,inat,darwincore"]
        exclude: \.ipynb$

  - repo: https://github.com/kynan/nbstripout
    rev: 0.9.1
    hooks:
      - id: nbstripout
        files: ^(notebooks/|scratch/)

  - repo: local
    hooks:
      - id: ruff-check
        name: Ruff Linting & Quality Checks
        entry: pixi run -e dev ruff check --fix
        language: system
        types: [python]

      - id: ruff-format
        name: Ruff Formatting
        entry: pixi run -e dev ruff format
        language: system
        types: [python]

      - id: mypy
        name: Strict Mypy Type Checking
        entry: pixi run -e dev mypy src scripts
        language: system
        files: ^(src/|scripts/|pyproject\.toml)
        pass_filenames: false
        types: [python]

      - id: validate-okf
        name: Enforce Google Open Knowledge Format (OKF v0.2)
        entry: pixi run -e dev python scripts/validate_okf.py
        language: system
        files: ^(docs/|\.agents/).*\.md$
        pass_filenames: true

      - id: audit-license-compliance
        name: Audit iNaturalist Open License Compliance
        entry: pixi run -e dev python scripts/audit_license_compliance.py
        language: system
        files: ^src/taxon_vision/data/
        pass_filenames: false

      - id: enforce-author-identity
        name: Enforce GPG/SSH Signed Commits
        entry: bash -c '[ "$(git config --bool commit.gpgsign)" = "true" ]'
        language: system
        pass_filenames: false
```

---

## 5. Hardened CI/CD Workflow (`.github/workflows/ci.yml`)

A two-pass verification architecture using `prefix-dev/setup-pixi` running on the lightweight `ci-dev` (CPU) environment:

```yaml
name: CI & Deployment Pipeline

on:
  push:
    branches: [ "main", "develop" ]
  pull_request:
    branches: [ "main", "develop" ]

permissions:
  contents: read
  pages: write
  id-token: write

env:
  FORCE_JAVASCRIPT_ACTIONS_TO_NODE24: "true"

jobs:
  quality-gate:
    name: Code, Security & Type Quality
    runs-on: ubuntu-latest
    timeout-minutes: 25
    steps:
      - uses: actions/checkout@v7
        with:
          fetch-depth: 0

      - name: Setup Pixi Environment (CPU ci-dev)
        uses: prefix-dev/setup-pixi@v0.10.2
        with:
          environments: ci-dev

      - name: Pre-Commit Quality Checks
        run: SKIP=enforce-author-identity pixi run --frozen -e ci-dev pre-commit run --all-files

      - name: Pass 1 - Unit Tests & Coverage
        run: |
          pixi run --frozen -e ci-dev pytest tests/unit/ --cov=src/taxon_vision --cov-report=xml -o "addopts="
          pixi run --frozen -e ci-dev uvx diff-cover coverage.xml --fail-under=80

      - name: Pass 2 - Integration & Conformal Invariants
        run: |
          pixi run --frozen -e ci-dev pytest tests/integration/ tests/invariants/ -x -q -o "addopts="

  cognitive-complexity:
    name: Complexipy Cognitive Complexity
    runs-on: ubuntu-latest
    permissions:
      security-events: write
    steps:
      - uses: actions/checkout@v7

      - name: Setup Pixi Environment
        uses: prefix-dev/setup-pixi@v0.10.2
        with:
          environments: ci-dev

      - name: Run Complexipy Scan
        run: |
          [ -d .cache/.complexity_cache ] && mv .cache/.complexity_cache .complexipy_cache || true
          pixi run --frozen -e ci-dev complexipy src/ --output-format sarif --output metrics-results.sarif --ignore-complexity
          pixi run --frozen -e ci-dev complexipy src/ --failed || true

      - name: Upload Bottlenecks to GitHub Code Scanning
        if: "!env.ACT"
        uses: github/codeql-action/upload-sarif@v4
        with:
          sarif_file: metrics-results.sarif

  architectural-profiling:
    name: Architecture & Line Count Profile
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v7
      - name: Generate SCC Report
        run: docker run --rm -v "${{ github.workspace }}:/pwd:ro" ghcr.io/boyter/scc:master scc --exclude-dir examples,scratch /pwd

  deploy-docs:
    name: Build & Deploy Documentation
    needs: [quality-gate, cognitive-complexity]
    if: github.ref == 'refs/heads/main' && github.event_name == 'push'
    environment:
      name: github-pages
      url: ${{ steps.deployment.outputs.page_url }}
    runs-on: ubuntu-latest
    timeout-minutes: 15
    steps:
      - uses: actions/checkout@v7

      - name: Setup Pixi Environment
        uses: prefix-dev/setup-pixi@v0.10.2
        with:
          environments: ci-dev

      - name: Generate OKF Knowledge Graph Visualization
        run: pixi run --frozen -e ci-dev python scripts/visualize_okf.py

      - name: Build Docs with Zensical
        run: pixi run --frozen -e ci-dev zensical build --strict

      - name: Upload Pages Artifact
        if: ${{ !env.ACT }}
        uses: actions/upload-pages-artifact@v5
        with:
          path: ./site

      - name: Deploy to GitHub Pages
        if: ${{ !env.ACT }}
        id: deployment
        uses: actions/deploy-pages@v5
```

---

## 6. The `Justfile` Task Engine

```just
# ==============================================================================
# TaxonVision-MLOps Justfile Command Runner
# ==============================================================================
set shell := ["bash", "-uc"]

default:
 @just --list

# ── Setup & Bootstrapping ───────────────────────────────────────────────────

[group("setup")]
setup mode="":
 @if [ "{{mode}}" = "--scratch" ]; then \
  just _setup-core; \
  just _setup-scratch; \
 elif [ -n "{{mode}}" ]; then \
  echo "Unknown setup flag: {{mode}}. Supported: --scratch"; \
  exit 1; \
 else \
  just _setup-core; \
 fi

[group("setup")]
setup-scratch:
 @just setup --scratch

[private]
_setup-core:
 pixi install -e dev
 pixi run --frozen -e dev pre-commit install

[private]
_setup-scratch:
 @echo "==> Configuring scratch experimentation repository..."
 @mkdir -p scratch/notebooks
 @if [ ! -d "scratch/.git" ]; then \
  git -C scratch init; \
  printf "# TaxonVision Scratch & Experimentation\n" > scratch/README.md; \
  printf ".ipynb_checkpoints/\n__pycache__/\n*.pyc\n" > scratch/.gitignore; \
  echo "*.ipynb filter=nbstripout" > scratch/.gitattributes; \
  git -C scratch config filter.nbstripout.clean "pixi run -e dev nbstripout"; \
  git -C scratch config filter.nbstripout.smudge cat; \
 fi
 @pixi run --frozen -e dev python -m ipykernel install --user --name=taxon-scratch --display-name="TaxonVision (Scratch)"
 @echo "==> Scratch environment ready. Launch with: just lab"

# ── Quality & Gates ─────────────────────────────────────────────────────────

[group("quality")]
lint:
 pixi run --frozen -e dev ruff check --fix .
 pixi run --frozen -e dev ruff format .
 pixi run --frozen -e dev mypy src scripts

[group("quality")]
check:
 pixi run --frozen -e dev pre-commit run --all-files

[group("quality")]
format:
 pixi run --frozen -e dev ruff format .

[group("quality")]
complexity:
 pixi run --frozen -e dev complexipy src/ --failed

[group("quality")]
validate-okf:
 pixi run --frozen -e dev python scripts/validate_okf.py

# ── Testing ─────────────────────────────────────────────────────────────────

[group("testing")]
test:
 pixi run --frozen -e dev pytest

[group("testing")]
test-unit:
 pixi run --frozen -e dev pytest tests/unit/

[group("testing")]
test-conformal:
 pixi run --frozen -e dev pytest tests/invariants/test_conformal_coverage_invariant.py -v

[group("testing")]
ci-test:
 ./scripts/local_ci.sh tests

# ── Pipeline & Models ───────────────────────────────────────────────────────

[group("mlops")]
fetch-sample:
 pixi run --frozen -e dev python -m taxon_vision.data.s3_streamer --limit 100 --out data/sample/

[group("mlops")]
train-baseline extractor="bioclip-2":
 pixi run --frozen -e dev python -m taxon_vision.models.trainer --extractor {{extractor}} --epochs 5

[group("mlops")]
export-onnx:
 pixi run --frozen -e dev python -m taxon_vision.inference.onnx_exporter

[group("mlops")]
pareto:
 pixi run --frozen -e dev python scripts/benchmark_pareto.py

[group("mlops")]
calibrate:
 pixi run --frozen -e dev python scripts/calibrate_conformal.py --alpha 0.05

# ── Serving & Web UI ────────────────────────────────────────────────────────

[group("app")]
run-api:
 pixi run --frozen -e dev uvicorn taxon_vision.service.api:app --host 0.0.0.0 --port 8000 --reload

[group("app")]
run:
 @just run-api

# ── Documentation ───────────────────────────────────────────────────────────

[group("docs")]
docs:
 pixi run --frozen -e dev zensical build

[group("docs")]
serve:
 pixi run --frozen -e dev zensical build
 pixi run --frozen -e dev zensical serve -a localhost:9000

[group("docs")]
visualize-okf:
 pixi run --frozen -e dev python scripts/visualize_okf.py

# ── Utilities ───────────────────────────────────────────────────────────────

[group("utils")]
lab:
 pixi run --frozen -e dev jupyter lab --notebook-dir=scratch --ip=127.0.0.1 --port=8888

[group("utils")]
clean:
 find . -type d -name "__pycache__" -exec rm -rf {} +
 rm -rf .cache site build dist .pytest_cache .mypy_cache .ruff_cache htmlcov .pixi
```

---

## 7. Mathematical & Scientific Formulations for Documentation

In accordance with the PHIDS scientific documentation standard, the Zensical documentation will include rigorous mathematical equations rendered via KaTeX:

### Conformal Prediction Coverage Guarantee (Angelopoulos & Bates, 2021)

For a user-specified significance level $\alpha \in (0, 1)$ (e.g., $\alpha = 0.05$ for 95% marginal coverage), given calibration non-conformity scores $s_i = 1 - \hat{f}(X_i)_{Y_i}$, we compute the conformal quantile $\hat{q}$:
$$\hat{q} = \text{Quantile}\left( \frac{\lceil (n+1)(1-\alpha) \rceil}{n}; \; s_1, \dots, s_n \right)$$
The prediction set for a novel image $X_{\text{test}}$ is then dynamically constructed:
$$C(X_{\text{test}}) = \left\{ y \in \mathcal{Y} : 1 - \hat{f}(X_{\text{test}})_y \le \hat{q} \right\}$$
Providing the finite-sample distribution-free theoretical guarantee:
$$P\left( Y_{\text{test}} \in C(X_{\text{test}}) \right) \ge 1 - \alpha$$
*Human Referral Rule:* If $|C(X_{\text{test}})| > k_{\text{max}}$ (excessive ambiguity) or $|C(X_{\text{test}})| = 0$ (anomaly/OOD), the observation is automatically redirected to the expert review queue.

### Class-Balanced Loss for Long-Tail Ecological Data (Cui et al., 2019)

Given an extreme long-tail frequency distribution where species $y$ has $n_y$ training instances, the effective number of samples is defined as:
$$E_{n_y} = \frac{1 - \beta^{n_y}}{1 - \beta}, \quad \text{where } \beta = \frac{N - 1}{N}$$
The Class-Balanced Cross-Entropy loss weights each sample $(x, y)$ by the reciprocal of its effective volume:
$$\mathcal{L}_{\text{CB}}(x, y) = -\frac{1 - \beta}{1 - \beta^{n_y}} \log\left( \hat{p}_y \right)$$

---

## 8. Phased Scaffolding & Milestone Progression

The project progression follows a logical 5-phase evolution without rigid time bounds:

* **Phase 0: Scaffolding & Environment Bootstrap**
    * Repository initialization with Git & GPG signing integrity.
    * Pixi multi-environment specification (`default`, `dev`, `ci`, `ci-dev`) in `pyproject.toml`.
    * Hardened pre-commit gates (`.pre-commit-config.yaml`) and `Justfile` automation.
    * Scratch experimentation setup with `nbstripout` clean filter.

* **Phase 1: Agentic Ecosystem & OKF Engine**
    * Master Router `AGENTS.md` and `.agents/index.md` decision matrix.
    * 12 specialized agent roles and 6 canned cloud personas (`Chisel`, `Bolt`, `Canon`, etc.).
    * OKF validation (`scripts/validate_okf.py`) and knowledge graph generator (`scripts/visualize_okf.py`).
    * Host CI script (`scripts/local_ci.sh`) and Docker rehearsal (`scripts/run_ci_with_act.sh`).

* **Phase 2: Ingestion & DarwinCore Data Pipeline (Level 1)**
    * Pure domain models: DarwinCore observation schemas, taxonomic tree, open license enums.
    * License filter: CC0, CC-BY, CC-BY-NC validation and attribution retention; static domain rejection.
    * S3 streaming client for AWS Open Data and synthetic fixtures (CUB-200 / Oxford Flowers subset) for unit testing.
    * DVC pipeline configuration for reproducible data versioning.

* **Phase 3: Model Architecture & Inference Optimization (Level 1 & Level 2)**
    * `timm` backbone factory supporting `BioCLIP-2`, `DINOv2-small`, `MobileNetV4`, and `EfficientNet-B0`.
    * Frozen feature extractor with trainable classification head.
    * Class-Balanced Loss (Cui et al., 2019) for long-tail species distributions.
    * ONNX Runtime export with dynamic INT8 and FP16 quantization for zero-allocation hot paths.
    * Automated Pareto benchmark suite comparing Accuracy vs. Latency vs. \$/M queries.

* **Phase 4: Conformal Uncertainty, OOD Safety & Feedback Loop (Level 3 & Level 4)**
    * Split Conformal Prediction engine guaranteeing distribution-free $1 - \alpha$ coverage.
    * Energy-based OOD detector for empty/blurry/non-organism images.
    * Human-in-the-Loop review queue and active learning query strategies (BADGE, CoreSet, Margin).
    * Validated feedback ingestion endpoint (`POST /api/v1/feedback`) and taxonomic mutation handler.
    * Low-latency Grad-CAM visual explanation heatmaps (< 25ms budget).

* **Phase 5: Unified Service, HTMX UI & Documentation**
    * FastAPI application serving both OpenAPI REST endpoints and server-rendered Jinja2/HTMX dashboard.
    * Prometheus metrics (`/metrics`) and health checks (`/health`).
    * Zensical documentation site with KaTeX formulas and Mermaid diagrams.
    * Complete test suite execution, coverage gates, and initial signed commit.

---

## 9. Next Steps for Initial Execution

Upon user approval of this blueprint:

1. **Repository Creation:** Initialize the target directory (`/home/benni/Documents/antigravity_workspace/taxon-vision-mlops`).
2. **Environment Solve:** Run `pixi install -e dev` to generate `pixi.lock` with deterministic CUDA 12.1 and CPU solver states.
3. **Core Scaffolding:** Populate `Justfile`, `.pre-commit-config.yaml`, `.github/workflows/ci.yml`, and `zensical.toml`.
4. **Agentic System Bootstrap:** Write `.agents/` roles, rules, workflows, skills, and OKF scripts.
5. **Level 1 Baseline Verification:** Scaffold DarwinCore data schemas, license filter, BioCLIP/MobileNet backbone factory, and run test suite.
