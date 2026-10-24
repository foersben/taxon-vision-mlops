# TaxonVision-MLOps Automation & Verification Scripts

This directory houses structured automation, verification, and governance scripts for the TaxonVision-MLOps platform. For full architectural specifications, parameter options, and failure-handling runbooks, see the [Automation & Scripts Reference](file:///home/benni/Documents/antigravity_workspace/taxon-vision-mlops/docs/operations/automation_and_scripts.md) in the documentation portal.

---

## Canonical Pipeline Entrypoints

All 7 unified pipeline stages are executed directly via their first-class CLI entrypoints in `src/taxon_vision/pipelines/`. Per Rule 00, all commands must use `pixi run -e dev`:

```bash
# Stage 1: Ingestion from iNaturalist API
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

## Script Index & Explanations

### Continuous Integration & Deployment (`scripts/ci/`)

#### `scripts/ci/jenkins_gpu_train.sh`

This script serves as the primary headless GPU training orchestrator, designed to execute inside an ephemeral K3s GPU worker pod (`ghcr.io/foersben/taxon-vision-gpu:latest`, PyTorch 2.7 `cu128`) scheduled on the bare-metal host's NVIDIA GeForce RTX 5070 Ti 16 GB GDDR7 (Blackwell SM 120) using 4-way NVIDIA time-slicing (or directly on a developer workstation with GPU passthrough). Its procedural lifecycle spans nine automated stages: (1) environment and Pixi bootstrap, (2) host-tier cache priming from `/var/lib/taxon-vision/` on host NVMe to minimize network I/O, (3) multi-tier DVC artifact hydration prioritizing the in-cluster SeaweedFS edge cache before falling back to the DagsHub cloud origin, (4) dynamic PyYAML injection of pipeline hyperparameters (`TUNE_HYPERPARAMETERS=true`, backbone selector `EXTRACTOR`) into root `params.yaml`, (5) conditional unfreezing of the iNaturalist ingestion stage with an automatic ERR trap to preserve lockfile integrity on API failures, (6) unified execution of the 7-stage Epistemic Diamond DAG via `dvc repro`, (7) atomic staging of newly minted checkpoints (`.tmp` write, permissions check, and atomic rename) to prevent live serving pods from loading partially written weights, (8) multi-tier DVC push synchronization to both edge SeaweedFS and cloud DagsHub remotes, and (9) Git staging and pushing of updated `dvc.lock`, `conformal_config.yaml`, and evaluation metrics with `[skip ci]` tags.

#### `scripts/ci/verify_deployment.sh`

This script provides automated post-deployment smoke testing and observability verification, executed by Jenkins or operators immediately following a zero-downtime rolling restart of `deploy/k8s/api-deployment.yaml`. It performs a deterministic 5-step validation procedure: (1) waits up to 60 seconds for `kubectl rollout status deployment/taxon-vision-serving -n taxon-vision` to confirm all pods are healthy, (2) inspects active replica states and drains terminating pods from EndpointSlices, (3) dynamically discovers the internal Kubernetes ClusterIP and port of `service/taxon-vision-serving`, (4) probes HTTP `/health` (liveness) and `/health/ready` (readiness, asserting ONNX model and taxonomy catalog are loaded in memory) using exponential curl retries, and (5) issues a synthetic multipart image classification request to `/api/v1/predict` and scrapes `/metrics` to verify that the Prometheus counter `taxon_predictions_total` increments deterministically.

#### `scripts/ci/local_ci.sh`

This developer utility replicates the complete GitHub Actions CI workflow locally on the native workstation without the overhead of Docker or `act`. It accepts target arguments (`quality`, `tests`, `docs`, `all`) and sequentially enforces the Pure Pixi environment invariant (verifying that no rogue `.venv/` or `uv.lock` files exist), runs Markdown Rule 04 verification, audits OKF frontmatter, executes Ruff linting and formatting, runs MyPy strict type checking, triggers the full pytest regression suite with coverage requirements ($\ge 80\%$), audits open-license compliance, and builds documentation in strict mode with Zensical.

---

### Maintenance & Model Registry (`scripts/maintenance/`)

#### `scripts/maintenance/cleanup_mlflow_registry.py`

This maintenance utility manages the remote DagsHub MLflow Model Registry (`taxon_vision_classifier`), purging stale trial versions generated by automated CI/CD runs while safeguarding production deployments and high-performing candidates. It operates under strict safety invariants: (1) model versions carrying active deployment aliases (`@Production`, `@Challenger`, `@Archived`) are permanently immune to deletion, (2) the $N$ most recent model versions (`--keep-last`, default 10) are retained unconditionally to preserve recent rollback history, (3) candidate models are ranked by their validation performance (`--metric-key val_pr_auc`) to preserve the top-$k$ versions (`--keep-top-k`, default 5), and (4) destructive deletions require an explicit `--execute` flag (dry-run by default) and apply polite request pacing (`--request-delay 0.15s`) to prevent rate-limit throttling against remote APIs.

---

### Governance & Quality Gates (`scripts/governance/`)

#### `scripts/governance/validate_okf.py`

This hardened validation engine enforces Google's Open Knowledge Format (OKF v0.2) across all documentation bundles in `docs/`, `.agents/`, and `openwiki/`. It performs deep semantic audits: verifying mandatory YAML `type` declarations, lifecycle statuses (`draft`, `stable`, `deprecated`), strict ISO-8601 UTC timestamp formatting on `stale_after` and `verified.at`, actor naming syntax (`human:<id>`, `process:<id>`), Attested Computation runtime declarations, and full internal link graph integrity to ensure zero broken cross-document links. It outputs structured trust-tier distributions (Human-Reviewed vs. Machine-Confirmed) and identifies stale knowledge assets requiring re-verification.

#### `scripts/governance/visualize_okf.py`

This tool parses the verified OKF documentation graph and compiles it into an interactive, standalone browser visualizer (`docs/viz.html`) powered by Cytoscape.js. It features three deterministic layout engines (CoSE force-directed physics, Dagre hierarchical DAG, and Circle layout), domain-filtering chips to isolate subdomains (`architecture`, `mlops`, `safety`, `operations`), real-time client-side keyword search, and an interactive metadata sidebar displaying frontmatter attributes, verified code claims, trust tiers, and bidirectional incoming and outgoing links.

#### `scripts/governance/verify_markdown_rules.py`

This script enforces project-wide Markdown Rule 04 formatting invariants across all markdown files. It validates that standard ASCII hyphens (`-`) are used exclusively (banning unicode en-dashes `–`, em-dashes `—`, and minus signs `−`), mandates asterisk (`*`) unordered lists with exactly 1 space and 4-space indentation multiples, requires exactly 1 blank line before and after lists, code blocks, and headings, and strips all trailing whitespace. It can be executed with `--fix` to automatically repair non-compliant markdown documents across the workspace.

---

### Data Utilities & Host Decryption

#### `scripts/data/download_samples.py`

This data acquisition utility queries the iNaturalist REST API to download verified research-grade citizen-science observations for the catalog taxa defined in `config/taxa_catalog.yaml`. It enforces strict open-licensing criteria (CC0, CC-BY, CC-BY-NC), preserves complete photographer attribution metadata in `data/sample/metadata.json`, and downloads sample imagery to `data/sample/` to serve as representative calibration datasets for static INT8 post-training quantization (PTQ).

#### `scripts/unlock_taxon.sh`

This decryption helper automates remote host disk decryption following bare-metal server reboots. It dispatches the LUKS root disk unlock passphrase to the initramfs Dropbear SSH daemon via `unlock-server taxon`, enabling zero-touch unattended maintenance while preserving full-disk encryption security.

#### Root CLI Forwarders (`scripts/validate_okf.py` & `scripts/visualize_okf.py`)

These top-level scripts act as backward-compatible CLI entrypoint wrappers. Utilizing Python's `runpy.run_path()`, they dynamically resolve and delegate execution to the canonical implementations in `scripts/governance/`, allowing developers and CI workflows to invoke the validators from the repository root without breaking path expectations.
