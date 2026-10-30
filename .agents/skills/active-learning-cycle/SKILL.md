---
type: Agent Skill
name: active-learning-cycle
title: Active Learning Cycle
status: stable
stale_after: "2027-01-01T00:00:00Z"
version: 1.0
description: Executes sample query ranking, human annotation ingestion, and retraining trigger.
tags: [skill, active-learning, triage, retraining]
generated: {by: process:okf-updater, at: "2026-10-17T22:30:00Z"}
verified: {by: process:okf-updater, at: "2026-10-17T22:30:00Z"}
---

# Active Learning Cycle

* **Description:** Continuous epistemic triage and model adaptation cycle that routes high-uncertainty observations (empty conformal sets or high OOD energy) to expert review, admits verified open-licensed labels, and triggers pipeline retraining.
* **Invocation:** Triggered via `/active-learning-cycle` or `/active-learning-step` in the AI IDE.

## Step-by-Step Procedure

* Query flagged observations from the triage queue exceeding uncertainty thresholds:
    * Predictions where the conformal prediction set size is empty or exceeds `max_set_size_for_auto_accept`.
    * Observations where the logit Helmholtz free energy exceeds `ood_energy_threshold`.

* Ingest newly verified expert annotations into the raw ingestion manifest:
    * Append verified records with required DarwinCore fields to `data/manifests/raw_manifest.parquet`.

* Run the active license gatekeeper pipeline to filter unverified or non-compliant licenses:

```bash
pixi run -e dev python -m taxon_vision.pipelines.audit_license \
  --manifest data/manifests/raw_manifest.parquet \
  --verified-manifest data/manifests/dataset_manifest.parquet \
  --report reports/license_audit.json
```

* Trigger downstream feature extraction and head retraining:

```bash
pixi run -e dev dvc repro
```

* Verify regenerated model artifacts and statistical coverage:
    * 4-way stratified embeddings: `data/embeddings/{train,val,cal,test}_embeddings.pt`.
    * Retrained classification head: `models/checkpoints/head.pt`.
    * Quantized production ONNX graph: `models/checkpoints/model_int8.onnx`.
    * Updated conformal calibration parameters: `config/conformal_config.yaml`.
    * Holdout Pareto evaluation metrics: `reports/pareto_evaluation.json`.

* Execute active learning unit tests:

```bash
pixi run -e dev pytest tests/unit/test_active_learning.py -q -o "addopts="
```
