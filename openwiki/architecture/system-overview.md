---
type: architecture
title: System Architecture & Design
description: High-level architectural overview, dual-pipeline pattern, component topology, and single-source-of-truth configuration hierarchy for TaxonVision.
tags: [architecture, system-overview, dual-pipeline, configuration, pydantic, kubernetes, onnx, mlops]
verified:
  - by: openwiki/0.7.2
    at: 2026-10-15T19:46:36.219Z
sources:
  - id: openwiki-source-978a014826b061d298cf1c14
    resource: repo://config/default_config.yaml
  - id: openwiki-source-c53db74dca0bf49162c87f32
    resource: repo://config/taxa_catalog.yaml
  - id: openwiki-source-67ce6221cdcbd3b0ca969d34
    resource: repo://deploy/k8s/api-deployment.yaml
  - id: openwiki-source-b9df81af2f7f5ce78210bf26
    resource: repo://deploy/k8s/jenkins-agent-rbac.yaml
  - id: openwiki-source-948b3861f8fd9364bb8fd9eb
    resource: repo://deploy/k8s/nvidia-time-slicing-config.yaml
  - id: openwiki-source-cfa03d5d7ac7270f0bb1bf28
    resource: repo://dvc.yaml
  - id: openwiki-source-7079947a60e14bc71bc44a0e
    resource: repo://src/taxon_vision/config.py
  - id: openwiki-source-ac488275ac5aaa7af4bd13ca
    resource: repo://src/taxon_vision/inference/onnx_exporter.py
  - id: openwiki-source-e4ad8e996f2c9027ac79d217
    resource: repo://src/taxon_vision/inference/quantizer.py
  - id: openwiki-source-e6a31393a2c56444d9f61713
    resource: repo://src/taxon_vision/models/training/embeddings.py
  - id: openwiki-source-2adf091ac1972d98fc4ca4fb
    resource: repo://src/taxon_vision/models/training/runner.py
  - id: openwiki-source-db9076027e694f431684d0c0
    resource: repo://src/taxon_vision/pipelines/extract_features.py
  - id: openwiki-source-d03d55dcc4b284c8e24e6028
    resource: repo://src/taxon_vision/pipelines/train.py
  - id: openwiki-source-d195e7451779d6a531c995ff
    resource: repo://src/taxon_vision/service/api.py
  - id: openwiki-source-644dda467779a35ba1ffdc4c
    resource: repo://src/taxon_vision/service/inference.py
generated: { by: "openwiki/0.7.2", at: "2026-10-15T19:46:36.219Z" }
---

# System Architecture & Design

TaxonVision is an automated machine learning operations (MLOps) platform engineered for high-throughput biological species classification from photographic occurrences. Operating under strict reliability and biodiversity safety constraints, the system augments deep vision foundation representations with statistical conformal prediction sets and Helmholtz free-energy Out-of-Distribution (OOD) detection.

This document details the high-level system topology, the single source of truth (SSOT) configuration hierarchy, the decoupled dual-pipeline training pattern, and the containerized deployment architecture. For related domain entities, operational runbooks, and workflow procedures, see [Domain Entities & Taxonomy Perimeter](domain-and-taxa.md), [Deployment & Observability](../operations/deployment-and-observability.md), [Service & API](../operations/service-and-api.md), [Quickstart](../quickstart.md), [Inference & Optimization](../workflows/inference-and-optimization.md), and [Model Training & Tracking](../workflows/model-training-and-tracking.md).

---

## Architectural Principles & Component Topology

TaxonVision's architecture is structured around five foundational engineering principles:

1. **Decoupled Representation and Head Learning (Dual-Pipeline Pattern)**: Heavy foundation model forward passes are executed offline to cache dense representation matrices, isolating computationally expensive vision backbones from rapid iterative classifier head tuning.
2. **Single Source of Truth (SSOT) Configuration**: System defaults are declared strictly in declarative YAML files (`config/default_config.yaml` and `config/taxa_catalog.yaml`), layered with subsystem parameters and environment variables, and strictly validated at runtime via Pydantic Settings models.
3. **Rigorous Uncertainty Guarantees**: Predictions are bounded by finite-sample distribution-free conformal coverage guarantees ($1 - \alpha$), flagging ambiguous classifications for human expert triage.
4. **Hardware-Aware Cloud-Native Deployment**: Kubernetes pod topologies leverage virtualized GPU time-slicing and INT8 post-training quantization (PTQ) to maximize hardware resource utilization on single-GPU nodes.
5. **End-to-End Data and Model Lineage**: Data versioning (DVC), experiment tracking (MLflow / DagsHub), and container artifact distribution (GHCR) maintain reproducible audit trails across every model release.

### High-Level System Topology

<!-- openwiki: mermaid parse failed and this diagram was converted to a text fence so it does not break rendering. Fix the diagram source and restore the mermaid fence. Parser error: Heuristic: an unescaped angle bracket inside a label breaks rendering; rephrase the label. -->
```text
flowchart TD
    subgraph S_CFG["Configuration Layer (SSOT)"]
        YML["config/default_config.yaml"]
        CAT["config/taxa_catalog.yaml"]
        PAR["config/params.yaml"]
        CAL["config/conformal_config.yaml"]
        ENV["Environment Overrides (TAXON_*)"]
        PYD["taxon_vision.config.get_settings<br/>(Pydantic BaseSettings Models)"]
        
        YML --> PYD
        CAT --> PYD
        PAR --> PYD
        CAL --> PYD
        ENV --> PYD
    end

    subgraph S_PIPE["Dual-Pipeline Architecture"]
        subgraph S_P1["Pipeline 1: Offline Feature Extraction"]
            INAT["iNaturalist API / S3 Open Data"]
            ING["scripts/ingest_data.py"]
            MAN["data/manifests (Parquet)"]
            EXT["taxon_vision.pipelines.extract_features"]
            BB["Frozen Vision Backbone<br/>(DINOv3 / BioCLIP / MobileNetV4)"]
            EMB["data/embeddings/cached_features.pt<br/>(EmbeddingSplit: Train/Val)"]
            
            INAT --> ING
            ING --> MAN
            MAN --> EXT
            BB --> EXT
            EXT --> EMB
        end

        subgraph S_P2["Pipeline 2: Linear Head Training & Optimization"]
            TRN["taxon_vision.pipelines.train<br/>(taxon_vision.models.trainer)"]
            OPT["Hyperparameter Optimization<br/>(Optuna)"]
            CBL["Class-Balanced Loss<br/>(Effective Number of Samples)"]
            MLF["MLflow / DagsHub Tracking<br/>(Metrics, Artifacts, Registry)"]
            HCK["models/checkpoints/head.pt<br/>(Trained Head Weights)"]
            
            EMB --> TRN
            OPT --> TRN
            CBL --> TRN
            TRN --> MLF
            TRN --> HCK
        end
        
        subgraph S_EXP["Export & Quantization"]
            ONX["taxon_vision.inference.onnx_exporter"]
            QNT["taxon_vision.inference.quantizer<br/>(Static INT8 PTQ Calibration)"]
            QMOD["models/checkpoints/taxon_vision_quantized.onnx<br/>(Dual Output: Logits + Embeddings)"]
            
            BB --> ONX
            HCK --> ONX
            ONX --> QNT
            QNT --> QMOD
        end
    end

    subgraph S_DEPLOY["Deployment & Runtime Topology"]
        subgraph S_K8S["Kubernetes Cluster (taxon-vision Namespace)"]
            SVC["Kubernetes Service<br/>(NodePort 30080:8000)"]
            DEP["Deployment: taxon-vision-api<br/>(2 Replicas, Non-Root UID 10001)"]
            VOL["Checkpoints Volume<br/>(hostPath /app/models/checkpoints)"]
            GPU["NVIDIA Device Plugin<br/>(Time-Slicing: 4 Virtual GPU Slices)"]
            
            SVC --> DEP
            VOL --> DEP
            GPU -.-> DEP
        end

        subgraph S_FASTAPI["FastAPI ASGI Application Root"]
            API_PRED["/predict & /api/v1/predict<br/>(Split-Conformal Prediction & OOD Energy)"]
            API_EXPL["/explain<br/>(Grad-CAM Visual Attribution)"]
            API_TRN["/train<br/>(Async Head Retraining)"]
            API_HLTH["/health & /health/ready<br/>(Probes & Prometheus Metrics)"]
            API_UI["/views<br/>(Interactive HTMX/Jinja2 Dashboard)"]
            
            DEP --> API_PRED
            DEP --> API_EXPL
            DEP --> API_TRN
            DEP --> API_HLTH
            DEP --> API_UI
        end
        
        QMOD -.-> DEP
    end

    PYD -.-> S_PIPE
    PYD -.-> S_FASTAPI
```
High-level architecture showing SSOT configuration resolution, dual-pipeline representation caching and head tuning, model quantization, and Kubernetes serving topology.

---

## Single Source of Truth (SSOT) Configuration Hierarchy

Configuration drift between YAML files, runtime scripts, and deployment manifests is prevented by TaxonVision's centralized configuration engine in `src/taxon_vision/config.py`. To eliminate duplicate declarations, declarative YAML files serve as the authoritative single source of truth, while typed Pydantic models validate and expose settings at runtime.

### Configuration Layering & Precedence

Application settings are loaded and merged by `get_settings()` (`src/taxon_vision/config.py`) using a strict precedence order:

1. **Base Declarative Configuration**: Loaded from `config/default_config.yaml`. Defines base default values across all subsystems (`service`, `model`, `training`, `ingestion`, `mlflow`, `conformal`, `ood`).
2. **DVC Pipeline Parameters Layer**: If `config/params.yaml` exists in the configuration root, its values are layered over the base configuration using recursive dictionary merging (`_deep_merge`).
3. **Statistical Conformal Calibration Layer**: If `config/conformal_config.yaml` exists, its `calibration` block is mapped and layered onto the `conformal` configuration block (`_load_conformal_override`).
4. **Custom YAML Overrides**: If a custom YAML file path is passed explicitly to `get_settings(config_path=...)` or specified via the `TAXON_CONFIG_PATH` environment variable, its contents are overlaid on the merged dictionary (`_resolve_custom_override`).
5. **Targeted Direct Environment Overrides**: Specific environment flags used for test isolation (`TAXON_ALLOW_UNTRAINED_FALLBACK` and `TAXON_FLAGGED_STORAGE_DIR`) are applied directly into the `model` sub-dictionary (`_apply_direct_env_overrides`).
6. **Dynamic Environment Variable Overrides**: Environment variables prefixed with `TAXON_` and separated by double underscores (`__`) take final precedence during Pydantic Settings parsing (for example, `TAXON_SERVICE__PORT=8000` overrides `service.port`, and `TAXON_MODEL__IMAGE_SIZE=336` overrides `model.image_size`).

```
Base YAML (default_config.yaml)
  └──> params.yaml (DVC params)
        └──> conformal_config.yaml (Calibrated quantiles)
              └──> Custom YAML (TAXON_CONFIG_PATH)
                    └──> Direct Env Overrides (TAXON_ALLOW_UNTRAINED_FALLBACK)
                          └──> Dynamic Env Overrides (TAXON_<SECTION>__<KEY>)
```

To prevent redundant file I/O and repetitive schema validation during high-throughput inference requests, `get_settings()` is wrapped with `@lru_cache(maxsize=1)`, caching the initialized `Settings` instance in process memory.

### Pydantic Validation Schemas

All configuration parameters are governed by strongly-typed Pydantic v2 schemas declared in `src/taxon_vision/config.py`:

| Schema Model | Primary Attributes & Type Constraints | Functional Domain |
|---|---|---|
| **`ServiceConfig`** | `name: str`, `version: str`, `port: int` ($0 < \text{port} < 65536$), `host: str`, `max_image_upload_bytes: int` ($> 0$), `allowed_mime_types: list[str]` | FastAPI HTTP server bindings, payload limits, and MIME content validation. |
| **`ModelConfig`** | `default_extractor: str`, `fallback_extractor: str`, `available_extractors: list[str]`, `image_size: int` ($> 0$), `channels: int` (3), `mean: list[float]`, `std: list[float]`, `onnx_model_path: str`, `head_checkpoint_path: str`, `taxa_catalog_path: str`, `eval_resize_ratio: float` ($> 1.0$), `inference_backend: Literal["auto", "onnx", "torch"]`, `quantization_mode: Literal["auto", "static", "dynamic"]`, `calibration_dir: str`, `calibration_samples: int`, `allow_untrained_fallback: bool`, `prefer_onnx_runtime: bool`, `onnx_opset_version: int`, `temperature: float` ($> 0.0$), `ood_temperature: float`, `ood_energy_threshold: float`, `flagged_storage_dir: str`, `backbone_registry: dict[str, str]` | Vision backbone architectures, image dimensions, normalization constants, ONNX export configurations, and OOD detection thresholds. |
| **`TrainingConfig`** | `epochs: int` ($> 0$), `batch_size: int` ($> 0$), `learning_rate: float` ($> 0.0$), `val_split: float` ($0.0 \le v < 1.0$), `random_seed: int`, `extractor: str` | Mini-batch sizing, optimizer hyperparameters, and PRNG seeds for classification head training. |
| **`IngestionConfig`** | `taxa_to_fetch: list[int]`, `per_page: int` ($> 0$) | iNaturalist API query parameters and taxon identifier target list. |
| **`MLflowConfig`** | `tracking_uri: str`, `experiment_name: str` | DagsHub remote tracking URI and MLflow experiment grouping name. |
| **`ConformalConfig`** | `alpha: float` ($0 < \alpha < 1$), `q_hat: float` ($0 < q < 1$), `k_max: int` ($> 0$), `quantile_threshold: float`, `nominal_coverage: float`, `method: str`, `empty_set_action: str`, `min_confidence_threshold: float` ($0.0 \le c \le 1.0$) | Conformal uncertainty calibration, prediction set size bounds ($k_{\max}$), and human triage thresholds. |
| **`OODConfig`** | `energy_threshold: float`, `entropy_threshold: float` | Free energy and predictive entropy thresholds for out-of-distribution observation routing. |
| **`Settings`** | Aggregates all above sub-models; configured with `SettingsConfigDict(env_prefix="TAXON_", env_nested_delimiter="__", extra="ignore")`. | Root application settings singleton. |

### Taxa Catalog Single Source of Truth

The catalog file `config/taxa_catalog.yaml` defines the authoritative taxonomic perimeter for the system (scientific names, vernacular common names, higher taxonomic ranks, primary iNaturalist taxon IDs, and taxonomic aliases).

To prevent configuration drift between catalog definitions and pipeline code, modules never hardcode integer label mappings. Instead, dynamic mapping generators such as `get_catalog_inat_taxon_map()` in `src/taxon_vision/pipelines/extract_features.py` inspect `config/taxa_catalog.yaml` at runtime:

```python
# Dynamically constructed mapping from config/taxa_catalog.yaml (SSOT)
DEFAULT_INAT_TAXON_MAP: dict[int, int] = get_catalog_inat_taxon_map()
```

Each primary taxon identifier, internal catalog ID, and taxonomic alias (such as ancestral genera or related species) is mapped to a dense ordinal class index in $[0, K-1]$, where $K$ is the number of catalogued taxa.

---

## Dual-Pipeline Architecture

Modern deep vision foundation backbones (such as DINOv3 ViT-Small, BioCLIP-2, and MobileNetV4) contain tens to hundreds of millions of parameters. Performing full end-to-end backpropagation through these architectures during rapid iteration or active learning loops is computationally prohibitive and prone to catastrophic forgetting on small datasets.

TaxonVision resolves this bottleneck through the **Dual-Pipeline Pattern**, decoupling heavy feature extraction from rapid linear classifier head training.

```
+-------------------------------------------------------------------------------+
| Pipeline 1: Offline Feature Extraction (taxa_catalog + dataset_manifest)     |
| Raw Images -> Preprocessing -> Frozen Backbone -> Stratified EmbeddingSplit   |
+-------------------------------------------------------------------------------+
                                      │
                                      ▼ (data/embeddings/cached_features.pt)
+-------------------------------------------------------------------------------+
| Pipeline 2: Rapid Head Training & Tuning (Optuna + ClassBalancedLoss + MLflow)|
| Cached Embeddings -> Linear Head -> Evaluation -> Model Checkpoint (head.pt)  |
+-------------------------------------------------------------------------------+
                                      │
                                      ▼
+-------------------------------------------------------------------------------+
| Downstream Serialization: ONNX Dual-Output Graph + Static INT8 Quantization   |
| Full Model (Backbone + Head) -> ONNX Export -> Calibration -> quantized.onnx  |
+-------------------------------------------------------------------------------+
```

### Pipeline 1: Offline Representation Extraction & Caching

The feature extraction pipeline (`taxon_vision.pipelines.extract_features`) processes raw observation images and converts them into dense representation vectors:

1. **Manifest Parsing**: Reads the observation manifest (`data/manifests/dataset_manifest.parquet`) containing observation identifiers, S3/HTTP image URLs, and taxon IDs.
2. **Deterministic Evaluation Transform**: Raw images are preprocessed via `get_evaluation_transform()`. Stochastic augmentations are omitted during caching so that cached representations capture the canonical visual signal. The transform performs:
   - Bilinear resizing of the shorter edge to `resize_size = round(image_size * eval_resize_ratio)` (e.g., $256 \times 256$ for $224 \times 224$ crops).
   - Exact square center cropping to `(image_size, image_size)`.
   - Conversion to float32 tensors scaled to $[0.0, 1.0]$.
   - Channel-wise normalization using ImageNet statistics (`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`).
3. **Forward Representation Pass**: Tensors are streamed in mini-batches through a frozen backbone (`create_feature_extractor`) instantiated without classification heads. The backbone operates with disabled gradient computation (`torch.no_grad()`) on CPU or CUDA hardware.
4. **Stratified Partitioning & Storage**: Extracted representations are mapped to dense class indices, partitioned into stratified train and validation splits (`EmbeddingSplit`) preserving class frequencies, and persisted to disk at `data/embeddings/cached_features.pt`.

The resulting `EmbeddingSplit` container holds `train_embeddings`, `train_labels`, `val_embeddings`, `val_labels`, `feature_dim`, and `samples_per_class`.

### Pipeline 2: Rapid Linear Head Training & Optimization

The classification head training pipeline (`taxon_vision.pipelines.train`, backed by `taxon_vision.models.trainer` and `taxon_vision.models.training.runner`) loads the serialized embeddings and trains a linear classification head (`TaxonClassifierHead`):

1. **In-Memory Training Loop**: Because the backbone is frozen and representations are pre-computed, mini-batch forward and backward passes execute directly on memory-resident embedding tensors ($N \times D$, where $D=1280$ for MobileNetV4 or $D=768$ for ViT-S/BioCLIP). A full 5-epoch training run completes in seconds on commodity hardware.
2. **Class-Balanced Loss**: Biological observation distributions suffer from severe power-law class imbalance. Training utilizes `ClassBalancedLoss` (`src/taxon_vision/models/loss.py`), which weights cross-entropy or focal loss dynamically by the effective number of samples $E_n = (1 - \beta^n) / (1 - \beta)$, preventing majority species from dominating gradient updates.
3. **Automated Hyperparameter Tuning**: An integrated Optuna engine (`tune_hyperparameters`) searches learning rate, weight decay, label smoothing, and dropout schedules over objective validation metrics.
4. **Experiment Tracking**: Training metrics (Macro F1, Macro PR-AUC, Top-1 Accuracy, training loss) and confusion matrices are logged to remote DagsHub MLflow tracking servers (`mlflow_run_scope`).
5. **Checkpoint Promotion**: If validation macro F1 exceeds the incumbent baseline, the pipeline writes the trained head weights to `models/checkpoints/head.pt` and invalidates the runtime memory cache.

### Serialization, Dual-Output Graph, & INT8 Quantization

Once head weights are established, the full model graph is serialized for low-latency serving:

1. **Unified Classifier Composition**: `TaxonClassifier` (`src/taxon_vision/models/head.py`) pairs the pre-trained vision backbone with the trained linear head into a single PyTorch `nn.Module`.
2. **Dual-Output Graph Wrapper**: During ONNX export (`src/taxon_vision/inference/onnx_exporter.py`), the model is wrapped in `_ONNXDualOutputModule`. When executed, the graph produces two output tensors:
   - `logits`: Dimension $(B, K)$ for species classification and conformal calibration.
   - `embeddings`: Dimension $(B, D)$ for Helmholtz free-energy OOD detection, data drift analysis, and active learning query sampling.
3. **Static INT8 Post-Training Quantization (PTQ)**: `src/taxon_vision/inference/quantizer.py` passes representative calibration images (`data/sample`) through the ONNX graph using ONNX Runtime's static calibration tooling. Convolution and matrix multiplication weights and activations are quantized to signed 8-bit integers (`QInt8` / `QUInt8`), reducing memory footprint by $\sim 75\%$ and yielding sub-20ms CPU inference latencies. The quantized artifact is saved to `models/checkpoints/taxon_vision_quantized.onnx`.

---

## Service Architecture & Serving Engine

The TaxonVision runtime microservice is composed as a unified FastAPI ASGI application in `src/taxon_vision/service/api.py`.

```
Client Requests (REST API / Browser / Prometheus Scraper)
                        │
                        ▼
           FastAPI ASGI Application Root
  ┌─────────────────────┼──────────────────────┬─────────────────┐
  │                     │                      │                 │
  ▼                     ▼                      ▼                 ▼
/health, /health/ready  /predict               /explain          /views
(Probes & Prometheus)   (Classification & OOD) (Grad-CAM Maps)   (HTMX Dashboard)
                        │
                        ▼
             Inference Engine Selector
       ┌────────────────┴────────────────┐
       ▼                                 ▼
ONNX Runtime Engine              PyTorch Fallback Classifier
(Quantized INT8 Graph)           (TaxonClassifier: Backbone + Head)
       │                                 │
       └────────────────┬────────────────┘
                        │ (Logits + Embeddings)
                        ▼
           Uncertainty & Safety Pipeline
       ┌────────────────┴────────────────┐
       ▼                                 ▼
Conformal Prediction Engine      Energy OOD Detector
(Split-Conformal Set: C(x))      (Helmholtz Free Energy E(x; T))
       │                                 │
       └────────────────┬────────────────┘
                        ▼
Response Serialization / Active Learning Flagging (data/flagged_observations)
```

### Route Handlers & Operational Surfaces

The application mounts modular route handlers:

- **`/health` & `/health/ready`** (`src/taxon_vision/service/routes/health.py`): Kubernetes liveness and readiness probes. Readiness strictly verifies that model artifacts (`taxon_vision_quantized.onnx` or `head.pt`) exist, load successfully, and match the class perimeter of `config/taxa_catalog.yaml`. Exposes Prometheus telemetry metrics (`/metrics`).
- **`/predict` & `/api/v1/predict`** (`src/taxon_vision/service/routes/predict.py`): Accepts multipart image uploads, validates payload size against `max_image_upload_bytes` and MIME type against `allowed_mime_types`, executes forward inference, computes conformal prediction sets, evaluates OOD energy, and returns JSON predictions.
- **`/explain`** (`src/taxon_vision/service/routes/explain.py`): Generates Class Activation Maps (CAM) highlighting the visual evidence and image regions that guided classification.
- **`/train`** (`src/taxon_vision/service/routes/train.py`): Triggers asynchronous background head retraining and reports execution status.
- **`/views`** (`src/taxon_vision/service/routes/views.py`): Server-rendered Jinja2/HTMX dashboard views for field biologists to inspect predictions, triage flagged observations, and review uncertain sets.

### Inference Backend Resolution

Inference execution is coordinated by `src/taxon_vision/service/inference.py`:

- **ONNX Runtime Engine (`ONNXInferenceEngine`)**: Statically loaded and cached via `get_onnx_engine()`. Activated when `inference_backend` is `"onnx"` or `"auto"` (and the quantized ONNX artifact exists). Avoids PyTorch runtime overhead on the hot serving path.
- **PyTorch Fallback Classifier (`TaxonClassifier`)**: Cached via `get_classifier()`. Activated when `inference_backend` is `"torch"` or when running in test environments where `allow_untrained_fallback` is enabled.
- **Artifact Validation**: Both engines validate that model output dimensions strictly equal the number of species defined in `load_taxa_catalog()`. If an artifact is missing or mismatched, `ModelNotReadyError` is raised, causing `/health/ready` to return HTTP 503.

### Uncertainty & OOD Safety Pipeline

Every prediction request passes through two statistical reliability gates before results are returned:

1. **Helmholtz Free-Energy OOD Detection**:
   Using logits $f_k(x)$ and temperature $T_{\text{ood}}$, free energy is calculated:
   $$E(x; T) = -T \cdot \log \sum_{k=1}^K \exp\left(\frac{f_k(x)}{T}\right)$$
   If $E(x; T) > \tau_{\text{energy}}$ (where $\tau_{\text{energy}}$ defaults to $-2.40$), the observation is flagged as out-of-distribution (e.g., an uncatalogued organism or non-biological noise) and persisted to `data/flagged_observations` for active learning triage.
2. **Split-Conformal Prediction Sets**:
   Given non-conformity calibration score quantile $\hat{q}$, the conformal engine constructs prediction set:
   $$C(x) = \left\{ k \in \{1, \dots, K\} \mid 1 - \text{Softmax}(f(x) / T)_k \le \hat{q} \right\}$$
   Guaranteeing marginal coverage $P(Y \in C(X)) \ge 1 - \alpha$. If the set size $|C(x)| > k_{\max}$ (default 3) or if the set is empty, the observation is automatically routed to human expert triage.

---

## Cloud-Native Deployment Topology & Infrastructure

TaxonVision is packaged as a cloud-native container image (`ghcr.io/foersben/taxon-vision-mlops:latest`) and deployed to a bare-metal or managed Kubernetes cluster.

### Kubernetes Workload Specification

The primary serving workload is configured in `deploy/k8s/api-deployment.yaml`:

- **Namespace**: `taxon-vision`.
- **Replicas & Rollout Strategy**: 2 replicas governed by a `RollingUpdate` strategy (`maxSurge: 1`, `maxUnavailable: 0`), guaranteeing zero-downtime rolling deployments during model updates.
- **Security Context**: Enforces least-privilege security by executing as a non-root user (`runAsNonRoot: true`, `runAsUser: 10001`, `runAsGroup: 10001`).
- **Resource Allocations**:
  - CPU requests: `200m`; limits: `2`.
  - Memory requests: `512Mi`; limits: `2Gi`.
- **Storage Mount**: A read-only `hostPath` volume mounts host checkpoints (`/home/benni/taxon-vision-mlops/models/checkpoints`) into the container at `/app/models/checkpoints`.
- **Health Probes**:
  - Liveness probe: HTTP GET `/health` on port 8000 (initial delay 15s, period 20s).
  - Readiness probe: HTTP GET `/health/ready` on port 8000 (initial delay 5s, period 5s, failure threshold 2).
- **Service Exposure**: A Kubernetes `NodePort` service routes traffic from node port 30080 to container port 8000.

### Hardware-Aware GPU Time-Slicing

On edge nodes and bare-metal environments equipped with consumer or prosumer GPUs (such as an NVIDIA RTX 5070 Ti with 16 GB VRAM), hardware Multi-Instance GPU (MIG) partitioning is unsupported. Under standard Kubernetes scheduling, a single pod requesting `nvidia.com/gpu: 1` places an exclusive lock on the entire card, causing all other workloads (such as CI runners and background batch training jobs) to stall in a `Pending` state.

To overcome this constraint, TaxonVision deploys an NVIDIA Device Plugin time-slicing configuration (`deploy/k8s/nvidia-time-slicing-config.yaml`):

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: nvidia-device-plugin-config
  namespace: kube-system
data:
  config.yaml: |
    version: v1
    sharing:
      timeSlicing:
        resources:
          - name: nvidia.com/gpu
            replicas: 4
```

This configuration multiplexes the single physical GPU into **4 virtual GPU slices**. The NVIDIA driver multiplexes GPU compute cycles via fast hardware context switching, while process memory isolation is enforced at the application layer:
- The Web API serving pods run lightweight INT8 ONNX Runtime models consuming $< 500\text{ MB}$ of memory.
- Batch training and feature extraction processes restrict allocation via PyTorch memory capping (`torch.cuda.set_per_process_memory_fraction`).

### CI/CD Pipeline & RBAC Isolation

Continuous integration and automated retraining runs are orchestrated via Jenkins and DVC:

- **Jenkins CI Agent RBAC**: Defined in `deploy/k8s/jenkins-agent-rbac.yaml`, granting the Jenkins agent service account scoped permissions in the `jenkins-ci` namespace to create, monitor, and delete ephemeral test runner pods without cluster-admin privileges.
- **Pipeline Reproducibility (DVC)**: DVC pipeline stages (`dvc.yaml`) track artifact and dependency hashing across three formal stages:
  1. `ingestion`: Downloads raw observations using `scripts/ingest_data.py` and produces `data/manifests`.
  2. `train`: Executes head training via `pixi run --frozen -e dev python -m taxon_vision.models.trainer` and produces `models/checkpoints/head.pt`.
  3. `export_onnx`: Exports and quantizes the model via `pixi run --frozen -e dev python -m taxon_vision.inference.onnx_exporter` and produces `models/checkpoints/model.onnx`.

---

## Related Documentation & Cross-References

- **[Domain Entities & Taxonomy Perimeter](domain-and-taxa.md)**: DarwinCore taxonomic entities, 10-taxa perimeter definition, alias resolution rules, and open licensing compliance.
- **[Deployment & Observability](../operations/deployment-and-observability.md)**: Production Kubernetes deployment, Prometheus metrics, and operational monitoring.
- **[Service & API](../operations/service-and-api.md)**: OpenAPI endpoints, request and response schemas, and interactive HTMX dashboards.
- **[Quickstart](../quickstart.md)**: Developer setup, Pixi environment configuration, and local execution commands.
- **[Inference & Optimization](../workflows/inference-and-optimization.md)**: ONNX graph export, dual-output embeddings, and static INT8 post-training quantization.
- **[Model Training & Tracking](../workflows/model-training-and-tracking.md)**: Linear head tuning, Optuna hyperparameter optimization, and MLflow experiment tracking.
