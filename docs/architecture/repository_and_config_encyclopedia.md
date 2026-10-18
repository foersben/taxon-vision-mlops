---
type: Reference
title: Repository & Config Encyclopedia
status: stable
stale_after: "2027-01-01T00:00:00Z"
version: 1.0
description: Comprehensive encyclopedia of repository topology, configuration files, environment variable overrides, and typed Pydantic models.
tags: [configuration, pydantic, schemas, params, environment-variables]
generated: {by: process:docs-librarian, at: "2026-10-18T12:00:00Z"}
verified: {by: process:docs-librarian, at: "2026-10-18T12:00:00Z"}
---

# Repository & Config Encyclopedia

The TaxonVision-MLOps configuration engine enforces Rule 00: **zero hardcoded magic numbers**. All operational limits, neural dimensions, batch sizes, and significance thresholds reside in declarative YAML manifests, validated at runtime by typed Pydantic v2 schemas in `src/taxon_vision/config.py`.

---

## 1. Declarative Configuration Files

The platform maintains four primary configuration manifests in the workspace:

### `params.yaml` (Root Pipeline Governance)

Governs the 7-stage DVC execution graph. DVC hashes individual keys in this file, ensuring that changing downstream training parameters does not trigger redundant upstream feature extraction:

```yaml
ingestion:
  target_taxa: "config/taxa_catalog.yaml"
  per_page: 100
  taxa_to_fetch: 20

model:
  extractor: "dinov3"
  image_size: 224

training:
  epochs: 10
  lr: 0.001
  batch_size: 64
  cb_beta: 0.999
  tune: false
  val_split: 0.10
  cal_split: 0.10
  test_split: 0.10

calibration:
  alpha: 0.05
  ood_tpr: 0.95

evaluation:
  latency_sla_ms: 50.0
```

### `config/default_config.yaml` (Application Defaults)

Provides baseline runtime configurations for standalone execution, FastAPI services, and active learning queues:

* **Inference Defaults:** Specifies default model path (`models/checkpoints/model_int8.onnx`), device preference (`cpu`/`cuda`), and IOBinding buffer sizes.
* **Telemetry Defaults:** Configures Prometheus scrape port (`8000`), histogram latency buckets, and drift threshold $\tau_W = 0.15$.
* **Storage Defaults:** Sets local SeaweedFS endpoints and S3 credentials.

### `config/taxa_catalog.yaml` (Biological Catalog)

Contains the canonical taxonomic definitions, GBIF taxon keys, and scientific names targeted by the ingestion pipeline:

```yaml
taxa:
  - taxon_id: 104
    scientific_name: "Danaus plexippus"
    common_name: "Monarch Butterfly"
    rank: "species"
  - taxon_id: 52
    scientific_name: "Apis mellifera"
    common_name: "Western Honey Bee"
    rank: "species"
```

### `config/conformal_config.yaml` (Runtime Safety Thresholds)

Committed to Git after Stage 6 (`pipelines.calibrate`) executes. Contains empirical calibration cutoffs evaluated by live FastAPI serving pods:

```yaml
alpha: 0.05
quantile_q_hat: 0.8421
tau_ood_energy: -11.4582
num_cal_samples: 240
```

---

## 2. Dynamic Environment Variable Overrides

The configuration loader in `src/taxon_vision/config.py` supports layered environment variable injection using double-underscore (`__`) notation:

```bash
# Override training learning rate
export TAXON__TRAINING__LR="0.0005"

# Switch model extractor trunk
export TAXON__MODEL__EXTRACTOR="bioclip-2"

# Enable Optuna Bayesian tuning
export TAXON__TRAINING__TUNE="true"
```

### Type Validation Mechanics

* All incoming environment variables undergo strict Pydantic v2 coercion and validation.
* Passing an invalid type (such as a string for a float field) causes the process to fail fast during startup with an informative schema error.
* Unknown parameters are rejected to prevent silent configuration typos.
