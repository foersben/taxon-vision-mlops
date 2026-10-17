# TaxonVision-MLOps Automation & Verification Scripts

This directory houses deterministic verification scripts, quality gates, and backwards-compatible CLI shims.

## Canonical Pipeline Location

The authoritative home for the 7 unified DVC pipeline stages is **`src/taxon_vision/pipelines/`**:

* `src/taxon_vision/pipelines/ingest.py` (Stage 1: Ingestion)
* `src/taxon_vision/pipelines/audit_license.py` (Stage 2: Active License Gatekeeper)
* `src/taxon_vision/pipelines/extract_features.py` (Stage 3: Feature Extraction & 4-Way Splitting)
* `src/taxon_vision/pipelines/train.py` (Stage 4: Head Training with Class-Balanced Loss)
* `src/taxon_vision/pipelines/export_onnx.py` (Stage 5: ONNX Model Export & Quantization)
* `src/taxon_vision/pipelines/calibrate.py` (Stage 6: Conformal & OOD Calibration)
* `src/taxon_vision/pipelines/evaluate.py` (Stage 7: Pareto Benchmark & Evaluation)

## Backwards-Compatible Pipeline CLI Shims

The following scripts serve as clean, documented entrypoint shims delegating to `taxon_vision.pipelines.*`:

* `ingest_data.py`: CLI shim delegating to `taxon_vision.pipelines.ingest`.
* `audit_license_compliance.py`: CLI shim delegating to `taxon_vision.pipelines.audit_license`.
* `calibrate_conformal.py`: CLI shim delegating to `taxon_vision.pipelines.calibrate`.
* `benchmark_pareto.py`: CLI shim delegating to `taxon_vision.pipelines.evaluate`.

## Quality & Automation Scripts

* `validate_okf.py`: Google Open Knowledge Format (OKF v0.2) validator verifying frontmatter, trust tiers, freshness, and link topology across `docs/` and `.agents/`.
* `visualize_okf.py`: Generates the interactive D3/SVG knowledge graph visualization (`docs/viz.html`).
* `download_samples.py`: Fetches verified open-license sample images for catalog taxa from iNaturalist.
* `local_ci.sh`: Reproduces the complete GitHub Actions CI pipeline locally on the native host without Docker overhead.
* `verify_deployment.sh`: Automated post-deployment smoke test verifying Kubernetes pod readiness and Prometheus metrics counters.
