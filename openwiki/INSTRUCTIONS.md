# OpenWiki Generation & Update Instructions

These instructions govern automated OpenWiki maintenance across the TaxonVision-MLOps repository.

When executing `openwiki code --update`, prioritize the following architectural layers and epistemic invariants:

* Core MLOps dual-pipeline architecture (`src/taxon_vision/pipelines/`, `src/taxon_vision/data/`, `src/taxon_vision/models/`):
    * S3 image streaming with strict open-license filtering (CC0, CC-BY, CC-BY-NC).
    * PyTorch foundation vision backbones (DINOv3, BioCLIP-2, DINOv2, MobileNetV4).
    * Class-balanced loss and multi-stage fine-tuning pipelines.

* Epistemic safety invariants:
    * Split Conformal Prediction (`src/taxon_vision/uncertainty/conformal.py`): finite-sample statistical coverage guarantee ($1 - \alpha$).
    * Helmholtz energy and predictive entropy Out-of-Distribution (OOD) detection (`src/taxon_vision/uncertainty/ood_detector.py`).
    * Ambiguity triage and conformal confidence threshold routing ($\tau_{\text{conf}}$).

* ONNX Runtime quantization and latency invariants (`src/taxon_vision/inference/`):
    * INT8 post-training quantization and ONNX session optimizations.
    * Zero-allocation inference execution paths with strict latency budgets ($p95 \le 50\text{ ms}$).

* FastAPI + HTMX operator service and telemetry (`src/taxon_vision/service/`, `src/taxon_vision/monitoring/`):
    * Unified REST inference API and server-rendered operator triage dashboard.
    * Prometheus metrics reporting, Kolmogorov-Smirnov distribution drift, and embedding drift monitoring.

* Agentic governance & OKF rules (`.agents/`):
    * Pure-Pixi environment discipline (forbidding bare pip, uv, and virtualenvs).
    * Role delegation protocols and GPG/SSH signed commit security invariants.
