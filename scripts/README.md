# TaxonVision-MLOps Automation & Verification Scripts

This directory houses structured automation, verification, and governance scripts for the TaxonVision-MLOps platform.

## Canonical Pipeline Entrypoints

All 7 unified pipeline stages are executed directly via their first-class CLI entrypoints in `src/taxon_vision/pipelines/`:

* `python -m taxon_vision.pipelines.ingest` (Stage 1: Ingestion)
* `python -m taxon_vision.pipelines.audit_license` (Stage 2: Active License Gatekeeper)
* `python -m taxon_vision.pipelines.extract_features` (Stage 3: Feature Extraction & 4-Way Splitting)
* `python -m taxon_vision.pipelines.train` (Stage 4: Head Training with Class-Balanced Loss)
* `python -m taxon_vision.pipelines.export_onnx` (Stage 5: ONNX Model Export & Quantization)
* `python -m taxon_vision.pipelines.calibrate` (Stage 6: Conformal & OOD Calibration)
* `python -m taxon_vision.pipelines.evaluate` (Stage 7: Pareto Benchmark & Evaluation)

## Directory Structure & Scripts

### 1. Continuous Integration & Deployment (`scripts/ci/`)

* `scripts/ci/local_ci.sh`: Reproduces the complete GitHub Actions CI pipeline locally on the native host without Docker overhead.
* `scripts/ci/verify_deployment.sh`: Automated post-deployment smoke test verifying Kubernetes pod readiness and Prometheus metrics counters.
* `scripts/ci/jenkins_gpu_train.sh`: Modular GPU training execution script invoked within the Jenkins GPU agent pod.

### 2. Data Utilities (`scripts/data/`)

* `scripts/data/download_samples.py`: Fetches verified open-license sample images for catalog taxa from iNaturalist.

### 3. Governance & Quality Gates (`scripts/governance/`)

* `scripts/governance/validate_okf.py`: Google Open Knowledge Format (OKF v0.2) validator verifying frontmatter, trust tiers, freshness, and link topology across `docs/` and `.agents/`.
* `scripts/governance/visualize_okf.py`: Generates the interactive D3/SVG knowledge graph visualization (`docs/viz.html`).
* `scripts/governance/verify_markdown_rules.py`: Enforces strict Markdown Rule 04 formatting invariants across all markdown files.
