---
type: Specification
title: Out-Of-Distribution & Empty Frame Detection
status: stable
stale_after: "2027-01-01T00:00:00Z"
version: 1.0
description: Energy-based scoring for identifying non-organism and uninformative images.
tags: [ood, energy-score, safety]
generated: {by: process:scaffold-init, at: "2026-10-02T10:00:00Z"}
verified: {by: process:scaffold-init, at: "2026-10-02T10:00:00Z"}
sources:
  - resource: "src/taxon_vision/pipelines/calibrate.py"
  - resource: "src/taxon_vision/pipelines/evaluate.py"
  - resource: "src/taxon_vision/uncertainty/ood_detector.py"
---

# Out-Of-Distribution & Empty Frame Detection

The system computes Helmholtz free energy over model logits to measure the energy-based Out-Of-Distribution (OOD) score:

$$E(x; f) = -T \cdot \log \sum_{k=1}^K e^{f_k(x)/T}$$

Observations where $E(x; f) > \tau_{\text{energy}}$ indicate absent organisms or extreme noise, triggering automated human referral.

## Distinguishing OOD vs. Ambiguous Taxa

Energy scores successfully distinguish between in-distribution ambiguous taxa and true out-of-distribution instances, such as camera trap blanks (e.g., wind-blown foliage, lens flares, non-biological objects). Ambiguous taxa typically produce diffuse confidence scores spread across multiple related classes, but maintain a low overall energy due to matching biological features. Conversely, empty frames and non-biological artifacts produce low activations across all classes, resulting in a high free energy score that cleanly separates them from the valid taxonomic distribution.

## Energy Threshold Calibration

In the Epistemic Diamond DAG, the Helmholtz free energy threshold $\tau_{\text{energy}}$ is calibrated by [`src/taxon_vision/pipelines/calibrate.py`](file:///home/benni/Documents/antigravity_workspace/taxon-vision-mlops/src/taxon_vision/pipelines/calibrate.py):

* **Calibration Split (`cal_embeddings.pt`):** Precomputed representations are evaluated against the quantized ONNX runtime (`models/checkpoints/model_int8.onnx`).
* **95% In-Distribution Retention:** $\tau_{\text{energy}}$ is set to the 95th percentile of in-distribution calibration energy scores, ensuring a nominal 95% True Positive Rate (TPR) on valid biological observations.
* **Serialization:** The calibrated threshold and temperature $T$ are stored in `config/conformal_config.yaml` under `ood.energy_threshold` and `ood.temperature`.
* **Inference Gating:** At runtime, [`EnergyOODDetector`](file:///home/benni/Documents/antigravity_workspace/taxon-vision-mlops/src/taxon_vision/uncertainty/ood_detector.py) flags any input exceeding $\tau_{\text{energy}}$, routing it immediately to human triage before conformal set construction.
