---
type: Concept
title: Epistemic Safety & Vision-RAG
status: stable
stale_after: "2027-01-01T00:00:00Z"
version: "1.0"
description: Epistemic uncertainty quantification, Split Conformal coverage guarantees, thermodynamic OOD detection, and Vision-RAG multimodal retrieval.
tags: [safety, conformal-prediction, uncertainty, ood, vision-rag, active-learning]
generated: {by: "process:docs-librarian", at: "2026-10-18T18:00:00Z"}
verified: {by: "process:docs-librarian", at: "2026-10-18T18:00:00Z"}
---

# Epistemic Safety & Vision-RAG

Uncalibrated softmax probabilities are mathematically deceptive and pose an unacceptable epistemic risk in biological monitoring. This section outlines the rigorous safety mechanisms that guarantee the reliability of TaxonVision-MLOps predictions.

We document the Split Conformal Prediction theory that provides mathematical coverage guarantees, ensuring that the true species is captured within a dynamic prediction set. We also detail Energy-Based Out-Of-Distribution (OOD) detection systems that filter out anomalous or empty frames natively in the feature space. When ambiguity is detected, the system safely triggers active learning pipelines and human-in-the-loop (HITL) triage to incrementally improve the taxonomic knowledge graph.

## Thematic Sections

* **[Conformal Prediction Theory](conformal_prediction_theory.md)**: Establishing distribution-free mathematical coverage bounds.
* **[Calibration Layers Deep Dive](calibration_layers_deep_dive.md)**: Four-layer calibration hierarchy from dynamic range to temperature scaling.
* **[OOD & Empty Frame Detection](ood_and_empty_frame_detection.md)**: Utilizing energy-based scoring for robust anomaly rejection.
* **[Interactive Conformal Simulator](interactive_simulator.md)**: Live simulation workbench for conformal bounds and OOD gating.
* **[Active Learning & Human Review](active_learning_and_hitl.md)**: Query strategies and triage routing for ambiguous predictions.
* **[Vision-RAG & Multimodal KG](mmkg_and_vision_rag.md)**: Lazy differential diagnosis and multimodal knowledge retrieval.
