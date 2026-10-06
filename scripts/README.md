# TaxonVision-MLOps Automation & Verification Scripts

This directory houses the deterministic verification scripts, quality gates, and evaluation runners:

* `validate_okf.py`: Google Open Knowledge Format (OKF v0.2) validator verifying frontmatter, trust tiers, freshness, and link topology across `docs/` and `.agents/`.
* `visualize_okf.py`: Generates the interactive D3/SVG knowledge graph visualization (`docs/viz.html`).
* `audit_license_compliance.py`: Asserts open-license compliance (CC0, CC-BY, CC-BY-NC) and photographer attribution preservation.
* `benchmark_pareto.py`: Benchmarks candidate backbones (BioCLIP-2, DINOv3, DINOv2, MobileNetV4, EfficientNet) evaluating Accuracy vs. Latency vs. \$/M queries.
* `calibrate_conformal.py`: Calibrates Split Conformal Prediction non-conformity score quantiles on holdout data.
* `local_ci.sh`: Reproduces the complete GitHub Actions CI pipeline locally on the native host without Docker overhead.
