---
type: Reference
title: Repository & Configuration Encyclopedia
status: stable
stale_after: "2027-01-01T00:00:00Z"
version: 1.1
description: Comprehensive specification of the 3-Tier Configuration Precedence Pyramid, Pydantic schema hierarchy, root params.yaml keys, and calibrated YAML artefacts.
tags: [configuration, settings, params, yaml, pydantic, precedence, drift, promotion, mlflow, telemetry]
generated: {by: "process:docs-librarian", at: "2026-11-08T17:18:00Z"}
verified: {by: "process:docs-librarian", at: "2026-11-08T17:18:00Z"}
sources:
  - resource: "config/default_config.yaml"
  - resource: "params.yaml"
  - resource: "config/conformal_config.yaml"
  - resource: "config/drift_config.yaml"
  - resource: "src/taxon_vision/config/schemas.py"
  - resource: "src/taxon_vision/config/loader.py"
---

# Repository & Configuration Encyclopedia

## 1. Conceptual Motivation: Why MLOps Configuration Requires a Precedence Pyramid

In traditional web services, configuration management is straightforward: application settings live in a YAML file and are overridden by container environment variables at deployment time. Machine learning operations (MLOps) introduce two additional sources of state that complicate this picture:

1. **DVC-Tracked Experiment Hyperparameters (`params.yaml`):** Data Version Control (DVC) tracks individual YAML keys (`training.learning_rate`, `conformal.alpha`, `model.extractor`) to compute pipeline invalidation graphs. If runtime service settings (such as HTTP ports or PostgreSQL pool sizes) were mixed into `params.yaml`, changing a web port would needlessly invalidate cached GPU feature embeddings or trigger model retraining.
2. **Empirically Calibrated Statistical Artefacts (`conformal_config.yaml` & `drift_config.yaml`):** Neither human operators nor static defaults can guess the exact finite-sample conformal quantile ($\hat{q}_\alpha = 0.8405$), the negative Helmholtz free energy cutoff ($\tau_{\text{OOD}} = -2.2792$), or the 99th-percentile null bootstrap Wasserstein-1 drift ceiling ($\tau_{W_1} = 0.12812$). These values are *outputs* of calibration stages that subsequently become *inputs* to the production inference and monitoring services.

To reconcile static system defaults, DVC experiment hyperparameters, calibrated statistical thresholds, and containerised runtime overrides without hardcoded magic numbers (Rule 00), TaxonVision-MLOps implements a deterministic **3-Tier Configuration Precedence Pyramid** in [`src/taxon_vision/config/loader.py`](file:///home/benni/Documents/antigravity_workspace/taxon-vision-mlops/src/taxon_vision/config/loader.py) validated by strict Pydantic v2 models in [`src/taxon_vision/config/schemas.py`](file:///home/benni/Documents/antigravity_workspace/taxon-vision-mlops/src/taxon_vision/config/schemas.py).

```mermaid
flowchart TD
    subgraph Pyramid["3-Tier Configuration Precedence Pyramid"]
        direction TB
        T3["Tier 3: Runtime Environment Overrides<br/>(TAXON__* Nested & Direct Env Variables)"]
        T2["Tier 2: DVC Hyperparameters & Calibrated Thresholds<br/>(params.yaml & config/conformal_config.yaml)"]
        T1["Tier 1: System Baseline Defaults<br/>(config/default_config.yaml)"]
    end

    T1 -->|Base Dictionary| M1["_deep_merge()"]
    T2 -->|Overlay Hyperparameters & Quantiles| M1
    M1 -->|Merged Dictionary| M2["_apply_direct_env_overrides()"]
    T3 -->|Inject Container Runtime Env| M2
    M2 -->|Instantiate & Validate| PYD["Settings (Pydantic BaseSettings)<br/>src/taxon_vision/config/schemas.py"]
```

---

## 2. The 3-Tier Configuration Precedence Pyramid

The configuration loader [`get_settings()`](file:///home/benni/Documents/antigravity_workspace/taxon-vision-mlops/src/taxon_vision/config/loader.py#L203-L257) resolves the workspace root, merges configuration layers in strict ascending precedence, and caches the validated [`Settings`](file:///home/benni/Documents/antigravity_workspace/taxon-vision-mlops/src/taxon_vision/config/schemas.py#L348-L377) singleton via `@lru_cache(maxsize=1)`:

### Tier 1: System Baseline Defaults (`config/default_config.yaml`)

* Serves as the foundational baseline for all 13 subsystem schemas: `service`, `model`, `training`, `ingestion`, `mlflow`, `conformal`, `ood`, `streaming`, `optuna`, `telemetry`, `drift`, `evaluation`, and `promotion`.
* Defines operational constants, connection pool bounds, ImageNet channel normalisation vectors, backbone registry mappings (`dinov3`, `bioclip-2`, `dinov2_vits14`, `mobilenetv4_conv_small`), and four-gate promotion tolerances.

### Tier 2: DVC Pipeline Hyperparameters & Calibrated Thresholds (`params.yaml` & `config/conformal_config.yaml`)

* **Root `params.yaml`:** Single source of truth for DVC-tracked machine learning hyperparameters (`ingestion`, `model`, `feature_extraction`, `training`, `export`, `conformal`, `evaluation`). Overlaid onto Tier 1 via recursive dictionary merging (`_deep_merge()`).
* **Calibrated Conformal Override (`config/conformal_config.yaml`):** Generated by Stage 6 (`calibrate`) and parsed by [`_load_conformal_override()`](file:///home/benni/Documents/antigravity_workspace/taxon-vision-mlops/src/taxon_vision/config/loader.py#L52-L86) to map empirical `q_hat`, `max_set_size_for_auto_accept` (`k_max`), and `nominal_coverage` directly into `Settings.conformal`.
* **Calibrated Drift Thresholds (`config/drift_config.yaml`):** Generated by [`src/taxon_vision/pipelines/calibrate_drift.py`](file:///home/benni/Documents/antigravity_workspace/taxon-vision-mlops/src/taxon_vision/pipelines/calibrate_drift.py) using 1,000 bootstrap iterations between `cal_embeddings.pt` and `test_embeddings.pt` (`tau_w1: 0.12812`, `alpha: 0.01`).
    * *Implementation Nuance:* Stage 3.5 (`src/taxon_vision/pipelines/data_drift.py`) explicitly parses `config/drift_config.yaml` via `_resolve_drift_thresholds(config_path=Path(settings.drift.config_path))`, evaluating training drift against the calibrated $\tau_{W_1} = 0.12812$. Conversely, the online serving drift evaluator (`src/taxon_vision/monitoring/drift_runner.py`) currently reads `resolved_settings.drift.tau_w1` directly from `Settings`, which resolves to the `0.15` baseline in `config/default_config.yaml` unless overridden by `TAXON_DRIFT__TAU_W1=0.12812` because `config/loader.py` does not yet auto-overlay `drift_config.yaml`.

### Tier 3: Runtime Environment Overrides (`TAXON__*` & Direct Variables)

* Highest precedence layer, designed for Kubernetes pods, Docker Compose services, and CI test isolation.
* **Nested Pydantic Delimiters (`__`):** Any field in `Settings` can be overridden via `TAXON_<SECTION>__<KEY>` (for example, `TAXON_SERVICE__PORT=8080` or `TAXON_TELEMETRY__DATABASE_URL=...`).
* **Direct Operational Environment Variables:** Handled explicitly by [`_apply_direct_env_overrides()`](file:///home/benni/Documents/antigravity_workspace/taxon-vision-mlops/src/taxon_vision/config/loader.py#L106-L161) and route security guards:

| Environment Variable | Target Subsystem / Field | Operational Purpose |
| :--- | :--- | :--- |
| `TAXON_CONFIG_PATH` | `loader.get_settings()` | Custom filesystem path to an external YAML override file. |
| `TAXON_ALLOW_UNTRAINED_FALLBACK` | `model.allow_untrained_fallback` | Permits uninitialised fallback weights during cold CI smoke tests (`true` / `false`). |
| `TAXON_FLAGGED_STORAGE_DIR` | `model.flagged_storage_dir` | Filesystem directory for persisting flagged OOD and ambiguous observations. |
| `TAXON_OPTUNA_STORAGE_URI` | `optuna.optuna_storage_uri` | SQLAlchemy URI for Optuna study persistence (`postgresql+psycopg2://...` or `sqlite:///...`). |
| `MLFLOW_TRACKING_URI` / `TAXON_MLFLOW_TRACKING_URI` | `mlflow.tracking_uri` | Tier 1 priority override for the MLflow 3.16.0 tracking server URI (`http://mlflow-server:5000`). |
| `TAXON_TELEMETRY_DATABASE_URL` | `telemetry.database_url` | PostgreSQL connection string for prediction telemetry and drift run persistence. |
| `TAXON_TELEMETRY_ALLOW_IN_MEMORY_FALLBACK` | `telemetry.allow_in_memory_fallback` | Permits `BoundedInMemoryTelemetryStore` ring buffer (`maxlen=500`) when PostgreSQL is offline. |
| `TAXON_INTERNAL_API_TOKEN` | `service/routes/drift.py` | Bearer token verified via `hmac.compare_digest` on `POST /api/v1/drift/evaluate`. |
| `TAXON_RETRAIN_WEBHOOK_SECRET` | `service/routes/retrain.py` | Bearer secret verified via `hmac.compare_digest` on `POST /api/v1/retrain/webhook`. |
| `TAXON_RETRAIN_COOLDOWN_SECONDS` | `service/routes/retrain.py` | Minimum interval in seconds (`21600.0` = 6 hours default) between webhook-triggered retraining runs. |

---

## 3. Pydantic Schema Hierarchy (`src/taxon_vision/config/schemas.py`)

The root [`Settings`](file:///home/benni/Documents/antigravity_workspace/taxon-vision-mlops/src/taxon_vision/config/schemas.py#L348-L377) model aggregates 13 strictly validated Pydantic sub-models. Below are the key MLOps, telemetry, drift, and governance schemas introduced alongside the core service and model configurations:

| Pydantic Schema Class | Root Key in `Settings` | Key Validated Fields & Defaults (`config/default_config.yaml`) | Architectural Responsibility |
| :--- | :--- | :--- | :--- |
| [`ServiceConfig`](file:///home/benni/Documents/antigravity_workspace/taxon-vision-mlops/src/taxon_vision/config/schemas.py#L13-L40) | `settings.service` | `port: 8000`, `max_image_upload_bytes: 15728640`, `image_cache_ttl_seconds: 900.0`, `redis_url: null` | FastAPI HTTP binding, MIME whitelist, and distributed Redis CAM cache parameters. |
| [`ModelConfig`](file:///home/benni/Documents/antigravity_workspace/taxon-vision-mlops/src/taxon_vision/config/schemas.py#L42-L96) | `settings.model` | `default_extractor: "dinov3"`, `onnx_model_path: "models/checkpoints/model_int8.onnx"`, `allow_untrained_fallback: false` | Vision backbone registry, normalization constants, ONNX opset 17, and OOD temperature. |
| [`TrainingConfig`](file:///home/benni/Documents/antigravity_workspace/taxon-vision-mlops/src/taxon_vision/config/schemas.py#L98-L136) | `settings.training` | `epochs: 30`, `batch_size: 32`, `learning_rate: 0.001`, `cb_beta: 0.9999`, `random_seed: 42`, `tune: false` | Linear probe / cosine head hyperparameters, Class-Balanced Loss $\beta$, and deterministic PRNG seed. |
| [`MLflowConfig`](file:///home/benni/Documents/antigravity_workspace/taxon-vision-mlops/src/taxon_vision/config/schemas.py#L150-L172) | `settings.mlflow` | `tracking_uri: "http://127.0.0.1:5000"`, `fallback_sqlite_uri: "sqlite:///data/mlflow/mlflow.db"`, `experiment_name: "taxon-vision-head-training"`, `drift_experiment_name: "taxon-vision-drift"`, `model_registry_name: "taxon_vision_classifier"`, `onnx_registry_name: "taxon_vision_onnx"`, `health_check_timeout_seconds: 1.0` | Three-tier MLflow 3.x URI resolution, experiment naming, and dual registry names. |
| [`ConformalConfig`](file:///home/benni/Documents/antigravity_workspace/taxon-vision-mlops/src/taxon_vision/config/schemas.py#L174-L200) | `settings.conformal` | `alpha: 0.05`, `q_hat: 0.85`, `k_max: 3`, `nominal_coverage: 0.95`, `min_confidence_threshold: 0.60`, `ood_tpr: 0.95` | Split Conformal coverage guarantee ($1-\alpha$), ambiguity referral bound $k_{\text{max}}$, and OOD TPR anchor. |
| [`OptunaConfig`](file:///home/benni/Documents/antigravity_workspace/taxon-vision-mlops/src/taxon_vision/config/schemas.py#L240-L264) | `settings.optuna` | `optuna_storage_uri: "sqlite:///data/optuna/optuna_studies.db"`, `n_trials: 30`, `min_resource: 3`, `reduction_factor: 2` | Bayesian TPE study storage backend and ASHA successive halving pruning bounds. |
| [`TelemetryConfig`](file:///home/benni/Documents/antigravity_workspace/taxon-vision-mlops/src/taxon_vision/config/schemas.py#L266-L285) | `settings.telemetry` | `database_url: "postgresql+psycopg://telemetry:...@localhost:5432/telemetry"`, `pool_min_size: 2`, `pool_max_size: 10`, `timeout_seconds: 5.0`, `allow_in_memory_fallback: false`, `fallback_buffer_size: 500` | PostgreSQL `psycopg_pool` connection limits and fail-fast vs ring-buffer fallback control. |
| [`DriftConfig`](file:///home/benni/Documents/antigravity_workspace/taxon-vision-mlops/src/taxon_vision/config/schemas.py#L287-L309) | `settings.drift` | `window_size: 500`, `min_samples: 50`, `tau_w1: 0.15`, `mmd_max_samples: 500`, `mmd_permutations: 200`, `alpha: 0.01`, `config_path: "config/drift_config.yaml"`, `reference_embeddings_path: "data/embeddings/cal_embeddings.pt"` | Sliding window dimensions, Tier 1 $W_1$ norm ceiling, Tier 2 RBF MMD² permutation count, and reference tensor path. |
| [`EvaluationConfig`](file:///home/benni/Documents/antigravity_workspace/taxon-vision-mlops/src/taxon_vision/config/schemas.py#L311-L321) | `settings.evaluation` | `latency_sla_ms: 120.0` (overridden by `params.yaml` when specified) | Operational latency ceiling enforced during Stage 7 (`eval`). |
| [`PromotionConfig`](file:///home/benni/Documents/antigravity_workspace/taxon-vision-mlops/src/taxon_vision/config/schemas.py#L324-L346) | `settings.promotion` | `quantization_drop_tol: 0.03`, `coverage_tolerance: 0.05`, `latency_slo_ms: 120.0`, `min_telemetry_samples: 50`, `drift_alpha: 0.01` | Thresholds governing the Four-Gate Model Promotion audit in `src/taxon_vision/models/training/promotion.py`. |

---

## 4. Comprehensive `params.yaml` Parameter Encyclopedia

Every parameter declared in root [`params.yaml`](file:///home/benni/Documents/antigravity_workspace/taxon-vision-mlops/params.yaml) is mapped to its consuming DVC pipeline stage and mathematical purpose:

| Section & Parameter Key | Data Type & Default | Consuming DVC Stage | Mathematical & Engineering Purpose |
| :--- | :--- | :--- | :--- |
| `ingestion.taxa_to_fetch` | `list[int]` (10 Taxa IDs) | `ingestion` | Array of iNaturalist taxon identifiers defining the Level 1 perimeter across 10 biological species. |
| `ingestion.per_page` | `int: 200` | `ingestion` | Page size for paginated HTTP requests to the iNaturalist REST API (`v1/observations`). |
| `model.extractor` | `str: "dinov3"` | `extract_features`, `train`, `export` | Name of the frozen foundation backbone used for embedding extraction (`dinov3`, `bioclip-2`, `mobilenetv4_conv_small`). |
| `model.image_size` | `int: 224` | `extract_features`, `export` | Pixel resolution for bicubic interpolation and square centre cropping ($224 \times 224$). |
| `feature_extraction.batch_size` | `int: 32` | `extract_features` | Mini-batch size passed through the PyTorch backbone during inference feature extraction. |
| `feature_extraction.val_split` | `float: 0.10` | `extract_features` | Holdout ratio allocated to validation embeddings for monitoring loss during probe training ($10\%$). |
| `feature_extraction.cal_split` | `float: 0.10` | `extract_features` | Holdout ratio allocated exclusively to the Split Conformal Prediction calibration set ($10\%$). |
| `feature_extraction.test_split` | `float: 0.10` | `extract_features` | Holdout ratio allocated to the final evaluation test split for empirical coverage verification ($10\%$). |
| `training.epochs` | `int: 5` | `train` | Total optimisation epochs over the pre-extracted training embeddings tensor. |
| `training.batch_size` | `int: 32` | `train` | Batch size fed to the cosine classifier head during gradient descent. |
| `training.learning_rate` | `float: 0.001` | `train` | Initial learning rate for the AdamW optimiser ($10^{-3}$). |
| `training.weight_decay` | `float: 0.0001` | `train` | $L_2$ weight regularisation factor ($10^{-4}$) to prevent overfitting on extracted embeddings. |
| `training.cb_beta` | `float: 0.9999` | `train` | Hyperparameter $\beta$ for Class-Balanced Loss effective sample weighting $E_n = \frac{1 - \beta^n}{1 - \beta}$. |
| `training.tune` | `bool: false` | `train` | Flag indicating whether to trigger an Optuna Bayesian hyperparameter search prior to final training. |
| `export.quantize_int8` | `bool: true` | `export` | When enabled, runs ONNX Runtime INT8 quantisation to produce `model_int8.onnx`. |
| `conformal.alpha` | `float: 0.05` | `calibrate`, `eval` | Significance level $\alpha$ bounding marginal error rate to at most $5\%$ (target coverage $1 - \alpha = 95\%$). |
| `conformal.min_confidence` | `float: 0.60` | `calibrate` | Minimum top-1 probability score required to bypass human review when the conformal set contains multiple candidates. |
| `conformal.ood_tpr` | `float: 0.95` | `calibrate` | In-distribution True Positive Rate ($95\%$) anchor used to calibrate the Helmholtz free energy cutoff $\tau_{\text{OOD}}$. |
| `evaluation.latency_sla_ms` | `float: 120.0` | `eval` | Maximum permissible p95 inference latency in milliseconds enforced during Stage 7 holdout evaluation. |

---

## 5. Live Configuration Artefact Showcase

### Empirical Conformal Calibration Output (`config/conformal_config.yaml`)

```yaml
calibration:
  method: split-conformal
  nominal_coverage: 0.95
  alpha: 0.05
  q_hat: 0.8405
  tau_ood: -2.2792
  ood_energy_threshold: -2.2792
  max_set_size_for_auto_accept: 3
  empty_set_action: route_to_human_triage
  min_confidence_threshold: 0.6
  num_calibration_samples: 169
```

### Empirical Bootstrap Drift Calibration Output (`config/drift_config.yaml`)

```yaml
drift:
  tau_w1: 0.12812
  alpha: 0.01
  window_size: 169
  n_bootstraps: 1000
  percentile: 99.0
  null_median_w1: 0.071163
  null_max_w1: 0.163083
  reference_source: data/embeddings/cal_embeddings.pt
  reference_sha256: 0a676f8ac95edb1a255906ed02b7f8bb1d143493056b3abd0de0659a2af1583f
  test_source: data/embeddings/test_embeddings.pt
  calibrated_at: '2026-11-02T12:16:47.473824+00:00'
```

### Taxonomic Perimeter Definition (`config/taxa_catalog.yaml`)

```yaml
taxa:
  - id: 1
    scientific_name: "Danaus plexippus"
    common_name: "Monarch Butterfly"
    kingdom: "Animalia"
    class: "Insecta"
    inat_taxon_id: 48662
  - id: 2
    scientific_name: "Apis mellifera"
    common_name: "Western Honey Bee"
    kingdom: "Animalia"
    class: "Insecta"
    inat_taxon_id: 47219
    inat_aliases:
      - 47120  # Apis (Honey bees - genus)
      - 47119  # Bombus (Bumble bees)
      - 47114  # Vespidae (Yellowjackets & Hornets)
  - id: 3
    scientific_name: "Turdus migratorius"
    common_name: "American Robin"
    kingdom: "Animalia"
    class: "Aves"
    inat_taxon_id: 12727
    inat_aliases:
      - 14704  # Turdus (genus)
```
