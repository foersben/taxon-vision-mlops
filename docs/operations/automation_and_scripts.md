---
type: Reference
title: Automation & Scripts Reference
status: stable
stale_after: "2027-01-01T00:00:00Z"
version: 1.0
description: Comprehensive operational reference for CI/CD automation, K3s GPU training orchestration, MLflow maintenance, and governance verification scripts.
tags: [scripts, automation, ci-cd, jenkins, gpu, mlflow, okf, governance]
generated: {by: "process:docs-librarian", at: "2026-10-24T22:25:00Z"}
verified: {by: "process:docs-librarian", at: "2026-10-24T22:25:00Z"}
---

# Automation & Scripts Reference

The `scripts/` directory houses structured automation, continuous integration, governance, and maintenance tools for the TaxonVision-MLOps platform. These utilities bridge local developer environments, bare-metal Jenkins orchestration, and production Kubernetes deployments.

---

## 1. Canonical Pipeline CLI Entrypoints

All 7 stages of the Epistemic Diamond DAG are implemented as first-class CLI entrypoints within `src/taxon_vision/pipelines/`. Per Rule 00, all commands must execute via `pixi run -e dev`:

```bash
# Stage 1: Ingestion from iNaturalist
pixi run -e dev python -m taxon_vision.pipelines.ingest

# Stage 2: License compliance gatekeeper
pixi run -e dev python -m taxon_vision.pipelines.audit_license

# Stage 3: Zero-copy S3 streaming & 4-way embedding extraction
pixi run -e dev python -m taxon_vision.pipelines.extract_features

# Stage 4: Classification head training with Class-Balanced Loss
pixi run -e dev python -m taxon_vision.pipelines.train

# Stage 5: ONNX graph export & INT8 post-training quantization
pixi run -e dev python -m taxon_vision.pipelines.export_onnx

# Stage 6: Split Conformal & Helmholtz OOD energy calibration
pixi run -e dev python -m taxon_vision.pipelines.calibrate

# Stage 7: Pareto benchmark evaluation & coverage guarantee verification
pixi run -e dev python -m taxon_vision.pipelines.evaluate
```

---

## 2. Continuous Integration & Deployment (`scripts/ci/`)

### `scripts/ci/jenkins_gpu_train.sh`

The primary GPU training orchestration script executed on the bare-metal server.

#### Execution Contexts

* **Ephemeral K3s GPU Worker Pod (Production CI):** Invoked inside an ephemeral Kubernetes pod (`ghcr.io/foersben/taxon-vision-gpu:latest`, PyTorch 2.7 `cu128`) scheduled on the host NVIDIA GeForce RTX 5070 Ti 16 GB GDDR7 (Blackwell SM 120) using 4-way NVIDIA time-slicing. Binds to in-cluster SeaweedFS S3 (`http://seaweedfs.taxon-vision.svc.cluster.local:8333`), in-cluster Optuna SQLite WAL (`/workspace/optuna/optuna_studies.db`), and mounts the host NVMe cache volume at `/var/lib/taxon-vision`.
* **Local Workstation Simulation:** Can be executed directly on the bare-metal Linux host or within a development container with GPU passthrough to simulate the full training and deployment pipeline locally.

#### Step-by-Step Procedural Lifecycle

1. **Tooling & Environment Bootstrap:** Verifies `git` and `kubectl` availability, configures git safe directories, syncs the frozen Pixi development environment (`pixi install --frozen -e dev`), and configures local S3 credentials for in-cluster SeaweedFS.
2. **Host NVMe Cache Priming:** Inspects the local directory `/var/lib/taxon-vision/` on the host's NVMe drive. Copies pre-existing feature embeddings (`*.pt`), parquet manifests, and model checkpoints directly into the workspace to avoid redundant compute or remote network pulls.
3. **Multi-Remote DVC Artifact Hydration:** Checks for missing Epistemic Diamond artifacts (`raw_manifest.parquet`, `train_embeddings.pt`, `head.pt`, `model_int8.onnx`). Tries the in-cluster SeaweedFS S3 cache (`local-seaweedfs`, running locally inside the cluster on the same host) first over local cluster networking. If still missing, falls back to the remote DagsHub S3 cloud archive (`origin`). If `raw_manifest.parquet` cannot be found in either remote, automatically forces `INGEST_DATA=true` to unfreeze and regenerate.
4. **Dynamic Parameter Injection:** Safely parses and modifies `params.yaml` using PyYAML based on Jenkins pipeline parameters (`TUNE_HYPERPARAMETERS=true` enables Optuna HPO, `EXTRACTOR` dynamically switches the foundation backbone).
5. **Conditional Live Ingestion & Error Trap:** If `INGEST_DATA=true`, unfreezes DVC stage `ingest`, runs live queries against the iNaturalist API with an ERR trap that restores `dvc.yaml` on failure, and freezes the stage again upon completion.
6. **Unified Pipeline Execution (`dvc repro`):** Executes the 7-stage Epistemic Diamond DAG across GPU (feature extraction, head training, ONNX INT8 PTQ quantization, conformal calibration, and Pareto evaluation).
7. **Host NVMe Cache Persistence (Atomic Staging):** Copies newly generated `.pt` embeddings and `.parquet` manifests back to local host storage `/var/lib/taxon-vision/`. For model checkpoints (`models/checkpoints/*`), executes atomic staging (`.tmp` write, `chmod a+r`, `mv`) to guarantee that zero-downtime serving pods mounting the directory never read partially written checkpoint files.
8. **Dual-Remote DVC Synchronization:** Pushes newly created DVC artifacts to the in-cluster SeaweedFS S3 cache (`local-seaweedfs`), then pushes to the remote DagsHub S3 cloud archive (`origin`).
9. **Git Artifact Commit & Push:** Adds updated `params.yaml`, `dvc.lock`, `config/conformal_config.yaml`, `docs/assets/pareto_results.json`, and `reports/*.json`, creates a `[skip ci]` commit, and pushes to the active branch.

#### Environment Variables

| Variable | Default | Purpose |
| :--- | :--- | :--- |
| `INGEST_DATA` | `false` | When `true`, unfreezes stage 1 to query fresh iNaturalist observations. |
| `TUNE_HYPERPARAMETERS` | `false` | When `true`, enables Optuna Bayesian sweeps with ASHA pruning. |
| `EXTRACTOR` | `""` | Overrides the backbone architecture in `params.yaml` (e.g. `bioclip-2`). |
| `BRANCH_NAME` | `develop` | Target Git branch for committing updated `dvc.lock` and evaluation reports. |
| `TAXON_OPTUNA_STORAGE_URI` | `sqlite:////workspace/optuna/optuna_studies.db` | Persistent database connection for Optuna HPO trials. |

---

### `scripts/ci/verify_deployment.sh`

Automated post-deployment verification script validating Kubernetes service health, readiness, and metrics counters.

#### Execution Context

Invoked by the Jenkins continuous deployment pipeline immediately after applying `deploy/k8s/api-deployment.yaml` and issuing `kubectl rollout restart deployment/taxon-vision-serving -n taxon-vision`.

#### 5-Step Verification Process

1. **Rollout Status Check:** Executes `kubectl rollout status deployment/taxon-vision-serving -n taxon-vision --timeout=60s` to confirm all updated replicas have achieved ready status.
2. **Pod Readiness & Drain Inspection:** Queries active pod status and waits up to 30 seconds for terminating replicas to completely drain from EndpointSlices before probing.
3. **Service Endpoint Resolution:** Queries Kubernetes API to discover the internal ClusterIP and port of `service/taxon-vision-serving`.
4. **HTTP Health & Readiness Probes:** Uses `curl` with exponential retries (`--retry 5 --retry-connrefused`) to probe `/health` (liveness) and `/health/ready` (readiness, asserting ONNX model and taxonomy catalog are loaded in memory).
5. **Live Inference & Telemetry Verification:** Submits a synthetic probe image via multipart POST to `/api/v1/predict` and verifies that Prometheus metrics scraped from `/metrics` confirm the deterministic increment of `taxon_predictions_total`.

```bash
# Manual execution
./scripts/ci/verify_deployment.sh [namespace]
```

---

### `scripts/ci/local_ci.sh`

Local developer validation harness reproducing the GitHub Actions CI pipeline without Docker overhead.

#### Workflow Targets

* `quality`: Enforces the Pure Pixi environment invariant (asserts no `.venv` or `uv.lock` exist), executes Markdown Rule 04 verification, audits OKF frontmatter, and runs Ruff linting, formatting, and MyPy type checks.
* `tests`: Executes the complete pytest test suite with coverage assertions ($\ge 80\%$), integration and invariant checks, and license compliance audits.
* `docs`: Audits OKF frontmatter and compiles documentation using Zensical in strict mode (`zensical build --strict`).
* `all`: Executes all three passes sequentially (default).

```bash
# Run all CI passes locally
./scripts/ci/local_ci.sh all

# Skip dependency re-sync
TAXON_SKIP_SYNC=1 ./scripts/ci/local_ci.sh quality
```

---

## 3. Maintenance & Model Registry (`scripts/maintenance/`)

### `scripts/maintenance/cleanup_mlflow_registry.py`

Maintenance utility for managing the remote MLflow Model Registry, purging stale trial runs while safeguarding production models.

#### Operational Safety Invariants

* **Active Alias Protection:** Model versions tagged with deployment aliases (`@Production`, `@Challenger`, `@Archived`) are strictly protected from deletion.
* **Retention Horizons:** Unconditionally preserves the $N$ most recent model versions (`--keep-last`, default 10).
* **Metric-Based Ranking:** Evaluates validation metrics (`--metric-key val_pr_auc`) across all candidate versions, preserving the top-$k$ highest-scoring models (`--keep-top-k`, default 5).
* **Dry-Run by Default:** Requires explicit `--execute` flag to execute destructive API calls. Rate-limiting delays (`--request-delay 0.15s`) prevent 429 throttling against remote endpoints.

```bash
# Dry-run audit (safe)
pixi run -e dev python scripts/maintenance/cleanup_mlflow_registry.py

# Execute pruning on remote DagsHub MLflow registry
pixi run -e dev python scripts/maintenance/cleanup_mlflow_registry.py --execute --keep-last 10 --keep-top-k 5
```

---

## 4. Governance & Quality Gates (`scripts/governance/`)

### `scripts/governance/validate_okf.py`

Automated gatekeeper enforcing Google's Open Knowledge Format (OKF v0.2) across all markdown documentation in `docs/`, `.agents/`, and `openwiki/`.

#### Audit Capabilities

* **Frontmatter Schema:** Validates mandatory `type`, lifecycle `status` (`draft`, `stable`, `deprecated`), strict ISO-8601 UTC datetimes, and actor naming syntax (`human:<id>`, `process:<id>`).
* **Attested Computation:** Confirms runtime verification receipts on generated concept artifacts.
* **Link Topology:** Traverses all internal document links to guarantee zero broken references across the knowledge graph.
* **Trust & Freshness Metrics:** Categorizes documents into trust tiers (Human-Reviewed vs Machine-Confirmed) and identifies stale documents exceeding `stale_after` bounds.

```bash
pixi run -e dev python scripts/governance/validate_okf.py
```

---

### `scripts/governance/visualize_okf.py`

Knowledge graph compiler transforming OKF concept documents and semantic links into an interactive Cytoscape.js web application (`docs/viz.html`).

#### Features

* **Deterministic Layout Engines:** Provides interactive switching between CoSE (force-directed physics), Dagre (hierarchical DAG), and Circle topologies.
* **Domain Filtering & Real-Time Search:** Allows operators to isolate subdomains (`architecture`, `mlops`, `safety`, `operations`) and execute instant keyword matching across node labels and descriptions.
* **Inspector Panel:** Displays detailed frontmatter metadata, verified code claims, trust tiers, and bidirectional links upon node selection.

```bash
# Regenerate documentation knowledge graph
pixi run -e dev python scripts/governance/visualize_okf.py
```

---

### `scripts/governance/verify_markdown_rules.py`

Syntax and formatting linter enforcing project-wide Markdown Rule 04 invariants.

#### Enforced Invariants

* **Standard ASCII Hyphens:** Bans unicode en-dash (U+2013), em-dash (U+2014), and horizontal bars (U+2015), enforcing standard hyphen (`-`) exclusively.
* **Asterisk Lists:** Requires asterisk (`*`) for all unordered lists with exactly 1 space and 4-space indentation multiples.
* **Blank Line Spacing:** Enforces exactly 1 blank line before and after lists, code blocks, and headings.
* **Whitespace Hygiene:** Automatically detects and strips trailing whitespace across all lines.

```bash
# Check formatting
pixi run -e dev python scripts/governance/verify_markdown_rules.py

# Automatically fix non-compliant formatting
pixi run -e dev python scripts/governance/verify_markdown_rules.py --fix
```

---

## 5. Data Utilities & System Decryption

### `scripts/data/download_samples.py`

Calibration sample fetcher querying the iNaturalist REST API for verified research-grade imagery.

* Queries catalog taxa defined in `config/taxa_catalog.yaml`.
* Validates open licensing (CC0, CC-BY, CC-BY-NC) and extracts photographer attribution metadata.
* Writes downloaded images to `data/sample/` with a corresponding `metadata.json` to provide representative inputs for INT8 post-training quantization.

```bash
pixi run -e dev python scripts/data/download_samples.py --per-taxon 2
```

---

### `scripts/unlock_taxon.sh`

Host decryption helper for unattended bare-metal maintenance.

Dispatches the LUKS root disk unlock passphrase to the initramfs Dropbear SSH daemon via `unlock-server taxon`, enabling zero-touch host reboots with full-disk encryption active.

```bash
./scripts/unlock_taxon.sh
```

---

### Top-Level CLI Forwarders (`scripts/validate_okf.py` & `scripts/visualize_okf.py`)

Convenience shims located directly under `scripts/`. They utilize Python's `runpy.run_path()` to dynamically resolve and invoke the canonical implementations in `scripts/governance/`, maintaining backward compatibility for legacy CI jobs and operator habit.
