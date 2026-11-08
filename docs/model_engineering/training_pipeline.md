---
type: System Pattern
title: Classification Head Training Pipeline
status: stable
stale_after: "2027-01-01T00:00:00Z"
version: 1.1
description: End-to-end training pipeline orchestration covering embedding caching, Optuna hyperparameter tuning, MLflow 3.x tracking, DVC dataset versioning, and Four-Gate model promotion.
tags: [training, mlflow, optuna, dvc, embeddings, promotion, pr-auc]
generated: {by: "process:docs-librarian", at: "2026-11-08T17:25:00Z"}
verified: {by: "process:docs-librarian", at: "2026-11-08T17:25:00Z"}
sources:
  - resource: "src/taxon_vision/models/training/runner.py"
  - resource: "src/taxon_vision/models/training/tracking.py"
  - resource: "src/taxon_vision/models/training/promotion.py"
  - resource: "src/taxon_vision/models/training/checkpoint.py"
  - resource: "src/taxon_vision/models/training/embeddings.py"
  - resource: "src/taxon_vision/models/training/tuning.py"
  - resource: "src/taxon_vision/models/training/types.py"
  - resource: "src/taxon_vision/pipelines/extract_features.py"
  - resource: "src/taxon_vision/pipelines/train.py"
---

# Classification Head Training Pipeline

The training module ([`src/taxon_vision/models/training/`](file:///home/benni/Documents/antigravity_workspace/taxon-vision-mlops/src/taxon_vision/models/training)) orchestrates linear and cosine classification head optimisation over **pre-cached foundation model embeddings**, records 5-dataset lineage and epoch trajectories in MLflow 3.16.0, and propagates its parent run context (`reports/train_metrics.json`) across downstream DVC stages (`export`, `calibrate`, `eval`) for the [Four-Gate Model Promotion](model_promotion_gates.md) audit.

---

## 1. Design Principles

* **Remote Cluster Execution Paradigm:** The local developer workstation serves strictly as an authoring, unit-testing, and orchestration plane. Heavy PyTorch feature extraction, Optuna Bayesian hyperparameter sweeps, and ONNX graph compilation execute on the dedicated bare-metal compute node (`hive-mind`) inside Kubernetes pods governed by GPU time-slicing slots and explicit VRAM fractions.
* **Zero Backbone Backpropagation:** All gradient descent iterates exclusively over the lightweight classification head (`Dropout` + `Linear` / Cosine projection), operating on embeddings pre-extracted by the frozen vision backbone (`dinov3`, `bioclip-2`, `dinov2_vits14`, `mobilenetv4_conv_small`). This reduces a full training epoch over 1,182 observations to milliseconds.
* **Class-Balanced Loss for Long-Tail Distributions:** Biodiversity observation streams follow a steep Pareto power law. The loss formulation re-weights each species $y$ by the effective number of samples $E_{n_y}$:

    $$\mathcal{L}_{\text{CB}}(\mathbf{p}, y) = -\frac{1 - \beta}{1 - \beta^{n_y}} \log(p_y)$$

    Where $\beta \to 1$ approximates inverse class-frequency weighting and $\beta \in [0.99, 0.9999]$ prevents gradient explosion on ultra-rare taxa.

* **Multi-Head Hierarchical Taxonomic Loss:** Rather than relying solely on a flat classification head over terminal leaf species, the architecture supports projecting embeddings across taxonomic ranks (Class, Order, Family, Genus, Species) so that coarse-grained taxonomic misclassifications are penalised more heavily than sibling-species confusion.
* **Macro PR-AUC Optimization & Four-Gate Promotion:** Within Stage 4 (`train`), early stopping and Optuna TPE/ASHA pruning optimise **Validation Macro PR-AUC**, which is invariant to decision thresholds and sensitive to minority-class precision. Authoritative registry promotion (`@Production` vs `@Challenger` vs `@Archived`) is subsequently decided in Stage 7 (`eval`) across all four governance gates (predictive viability + INT8 fidelity, conformal coverage, p95 latency, and live MMD² calibration exchangeability).
* **Self-Hosted MLflow 3.16.0 & DagsHub Interlock:** Primary experiment tracking runs against the self-hosted `mlflow-server` container backed by PostgreSQL 16 (`mlflow` database) and SeaweedFS S3 (`s3://mlflow-artifacts`), with automatic 3-tier fallback to local SQLite (`sqlite:///data/mlflow/mlflow.db`) during offline development and offsite DVC checkpoint synchronisation to DagsHub S3 (`s3://dvc`).
* **Git-Free DVC Versioning:** Dataset content hashes (`dvc_dataset_hash`) are extracted by parsing `dvc.lock` directly in [`dataset.py`](file:///home/benni/Documents/antigravity_workspace/taxon-vision-mlops/src/taxon_vision/models/training/dataset.py), ensuring deterministic lineage even inside minimal container build contexts.

---

## 2. Module Map (Sibling Topology)

Following Rule 08 (Structural Sibling Topology), the training subsystem in `src/taxon_vision/models/training/` is decomposed into focused, single-responsibility sibling modules (each maintaining cognitive complexity $< 15$):

| Module | Key Symbols | Architectural Responsibility |
| :--- | :--- | :--- |
| [`runner.py`](file:///home/benni/Documents/antigravity_workspace/taxon-vision-mlops/src/taxon_vision/models/training/runner.py) | `run_training_pipeline()`, `build_training_components()`, `configure_cuda_memory_budget()` | Top-level CLI training coordinator supporting standard runs, `--tune` Optuna sweeps, and `reports/train_metrics.json` persistence. |
| [`dataset.py`](file:///home/benni/Documents/antigravity_workspace/taxon-vision-mlops/src/taxon_vision/models/training/dataset.py) | `load_or_create_training_split()`, `get_dvc_dataset_hash()`, `resolve_feature_dim()` | Tensor hydration, `dvc.lock` MD5 digest verification, and synthetic Gaussian fallbacks for offline tests. |
| [`loops.py`](file:///home/benni/Documents/antigravity_workspace/taxon-vision-mlops/src/taxon_vision/models/training/loops.py) | `train_head_epoch()`, `evaluate_head()`, `HeadEvaluationResult` | Isolated single-epoch mini-batch training and validation passes. |
| [`embeddings.py`](file:///home/benni/Documents/antigravity_workspace/taxon-vision-mlops/src/taxon_vision/models/training/embeddings.py) | `train_head_on_cached_embeddings()` | Multi-epoch training loop coordinating best-weight restoration, early stopping, and learning-rate plateau scheduling. |
| [`promotion.py`](file:///home/benni/Documents/antigravity_workspace/taxon-vision-mlops/src/taxon_vision/models/training/promotion.py) | `evaluate_promotion_gates()`, `promote_candidate_model()`, `PromotionCriteria`, `PromotionVerdict` | Executes the Four-Gate Model Promotion Architecture in Stage 7 and manages `@Production`, `@Challenger`, and `@Archived` aliases. |
| [`tracking.py`](file:///home/benni/Documents/antigravity_workspace/taxon-vision-mlops/src/taxon_vision/models/training/tracking.py) | `mlflow_run_scope()`, `resume_parent_mlflow_run()`, `_resolve_reachable_tracking_uri()`, `MLflowTracker` | RAII context managers for run lifecycle, cross-stage DVC run resumption, 5-dataset lineage, and `LoggedModel` registration. |
| [`tuning.py`](file:///home/benni/Documents/antigravity_workspace/taxon-vision-mlops/src/taxon_vision/models/training/tuning.py) | `tune_hyperparameters()`, `HyperparameterObjective`, `TuningResult` | Bayesian hyperparameter optimisation study orchestration with TPE sampling and ASHA pruning. |
| [`visualizations.py`](file:///home/benni/Documents/antigravity_workspace/taxon-vision-mlops/src/taxon_vision/models/training/visualizations.py) | `log_training_visualizations()`, `log_holdout_visualizations()`, `HoldoutVisualizationPayload`, `ParetoProfileLike` | Publication-grade figure rendering for training validation and 6-plot Stage 7 holdout diagnostics. |
| [`checkpoint.py`](file:///home/benni/Documents/antigravity_workspace/taxon-vision-mlops/src/taxon_vision/models/training/checkpoint.py) | `save_checkpoint()` | Deterministic PyTorch state dictionary serialisation (`models/checkpoints/head.pt`). |
| [`types.py`](file:///home/benni/Documents/antigravity_workspace/taxon-vision-mlops/src/taxon_vision/models/training/types.py) | `ResolvedTrainingParameters`, `EmbeddingSplit`, `HeadTrainingConfig`, `TrainingPipelineResult`, `PromotionAlias` | Validated Pydantic models, enums, and frozen dataclasses enforcing zero tuple blindness. |

---

## 3. Pipeline Lifecycle & MLflow RAII Strategy

The MLflow run lifecycle is strictly managed via two `@contextmanager` primitives in [`tracking.py`](file:///home/benni/Documents/antigravity_workspace/taxon-vision-mlops/src/taxon_vision/models/training/tracking.py) (detailed in [MLflow 3.x Experiment Tracking, Lineage & PyFunc Registry](mlflow_tracking_and_registry.md)):

1. **`mlflow_run_scope()` (Stage 4 Creation):** Resolves the tracking URI via 3-tier health probing, opens the parent run, logs all five dataset splits (`inaturalist_open_manifest` + `train`, `val`, `cal`, `test` tensors), registers the PyTorch `LoggedModel` (`m-...`), and guarantees that any unhandled exception marks the run as `FAILED` before re-raising.
2. **`resume_parent_mlflow_run()` (Stages 5, 6, 7 Continuity):** Reads `mlflow_run_id` and `mlflow_model_id` from `reports/train_metrics.json` so that standalone DVC subprocesses (`export_onnx`, `calibrate`, `evaluate`) re-enter the exact same parent MLflow run rather than scattering fragmented runs across the tracker.

```mermaid
flowchart TD
    subgraph Coordination["Top-Level Orchestration (Stage 4)"]
        R["models.training.runner<br/>run_training_pipeline()<br/>build_training_components()<br/>configure_cuda_memory_budget()"]
    end

    subgraph DataAndState["State & Tensor Hydration"]
        DS["models.training.dataset<br/>load_or_create_training_split()<br/>get_dvc_dataset_hash()<br/>resolve_feature_dim()"]
        CK["models.training.checkpoint<br/>save_checkpoint()"]
    end

    subgraph Optimization["Epoch Training & Hyperparameter Tuning"]
        LP["models.training.loops<br/>train_head_epoch()<br/>evaluate_head()"]
        EM["models.training.embeddings<br/>train_head_on_cached_embeddings()"]
        TN["models.training.tuning<br/>tune_hyperparameters()<br/>Optuna TPE & ASHA Pruner"]
    end

    subgraph GovernanceAndTracking["Lineage, Promotion & Observability (Stages 4-7)"]
        TR["models.training.tracking<br/>mlflow_run_scope() & resume_parent_mlflow_run()<br/>_log_dataset_lineage()"]
        PR["models.training.promotion<br/>evaluate_promotion_gates()<br/>promote_candidate_model() (Stage 7)"]
        VZ["models.training.visualizations<br/>log_training_visualizations()<br/>log_holdout_visualizations() (6 Plots)"]
    end

    R -->|Hydrates Tensors| DS
    R -->|Scopes Session RAII| TR
    R -->|Dispatches Optuna Sweeps| TN
    R -->|Executes Training Loop| EM
    EM -->|Delegates Mini-Batches| LP
    R -->|Persists head.pt & train_metrics.json| CK
    TR -->|Resumes Run in Stage 7| PR
    TR -->|Resumes Run in Stage 7| VZ
```

---

## 4. Inner Training Loop & Convergence Callbacks

[`train_head_on_cached_embeddings()`](file:///home/benni/Documents/antigravity_workspace/taxon-vision-mlops/src/taxon_vision/models/training/embeddings.py) executes the mini-batch optimisation loop over `EmbeddingSplit`. After each epoch, it computes three validation metrics:

* **Top-1 Accuracy:** Logged to MLflow for diagnostic visibility, but never used as the sole promotion criterion due to class-imbalance sensitivity.
* **Macro F1-Score:** Unweighted harmonic mean of precision and recall across all modelled taxa.
* **Macro PR-AUC:** Primary optimisation objective for early stopping, learning-rate scheduling, Optuna pruning, and Gate 1 incumbent comparison.

### Convergence Callbacks

* **Best Model Weights Restoration (`restore_best_weights=True`):** Tracks validation Macro PR-AUC at every epoch. Whenever a new maximum is achieved, a deep copy of the `state_dict` is cached in memory and restored automatically upon loop termination.
* **Early Stopping (`early_stopping_patience=5`):** Terminates training if validation Macro PR-AUC fails to improve for `early_stopping_patience` consecutive epochs, preventing memorisation of rare long-tail exemplars.
* **Learning Rate Reduction on Plateau (`reduce_lr_patience=2`):** Wraps the `AdamW` optimiser in `ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)`, halving the learning rate when validation PR-AUC plateaus.
* **Optuna Pruner Hook (`pruner_callback`):** During `--tune` sweeps, reports intermediate epoch PR-AUC scores to the ASHA pruner (`SuccessiveHalvingPruner`), aborting underperforming trials after `min_resource` epochs.

---

## 5. Optimal Training Dynamics & Hardware Budgeting

* **Epoch Budget:** Linear and cosine heads over frozen foundation embeddings typically converge within 5 to 15 epochs.
* **Batch Size & VRAM Partitioning:**
    * **Batch Size 32-64 for Cached Embeddings:** Maximises SIMD vector throughput with minimal gradient variance.
    * **70% CUDA Memory Cap on RTX 5070 Ti:** When extracting features or training on the bare-metal host's NVIDIA GeForce RTX 5070 Ti (16 GB GDDR7 VRAM), `configure_cuda_memory_budget()` enforces `torch.cuda.set_per_process_memory_fraction(0.7)` ($11.2\,\text{GB}$ ceiling), guaranteeing an isolated $4.8\,\text{GB}$ VRAM reserve for the active INT8 ONNX serving pod.
* **Biological Data Augmentation Invariants:**
    * **Horizontal Reflection Permitted:** Organisms exhibit bilateral or radial symmetry across the horizontal axis; horizontal flips generate biologically valid training exemplars.
    * **Vertical Reflection Strictly Prohibited:** Gravity, geotropism, and phototropism establish fixed vertical orientations in natural habitats. Vertical flips produce unphysical specimens (such as inverted tree canopies or upside-down perched birds) that corrupt spatial attention maps.

---

## 6. Optuna Bayesian Hyperparameter Optimisation

Optuna explores the hyperparameter space using the Tree-structured Parzen Estimator (`TPESampler`) paired with Asynchronous Successive Halving (`SuccessiveHalvingPruner`):

* **Search Space Formulation:**
    * **Learning Rate ($\eta$):** Log-uniform over $[10^{-4}, 10^{-1}]$.
    * **Weight Decay ($\lambda$):** Log-uniform over $[10^{-6}, 10^{-2}]$.
    * **Dropout Rate ($p$):** Uniform over $[0.0, 0.5]$.
    * **Class-Balanced Loss $\beta$:** Uniform over $[0.99, 0.9999]$.
* **Relational Persistence:** In Docker Compose and K3s deployments, `TAXON_OPTUNA_STORAGE_URI` points to the consolidated PostgreSQL 16 `optuna` database (`postgresql+psycopg2://optuna:...@postgres:5432/optuna`), visualised live via the Optuna Dashboard on port `8080` (NodePort `30081`). In offline local runs, storage falls back to `sqlite:///data/optuna/optuna_studies.db`.
* **CLI Trigger:**

```bash
pixi run -e dev python -m taxon_vision.pipelines.train --tune --n-trials 15
```

```mermaid
sequenceDiagram
    autonumber
    participant Objective as HyperparameterObjective.__call__
    participant Emb as train_head_on_cached_embeddings
    participant Metrics as evaluate_head
    participant Optuna as optuna.Trial (ASHA Pruner)
    participant MLflow as MLflow Nested Run

    Objective->>Objective: _sample_hyperparameters(trial)
    Objective->>MLflow: start_run(nested=True, run_name="trial_N")
    Objective->>Emb: train_head_on_cached_embeddings(head, data, optimizer, criterion, config)

    loop epoch in range(epochs_per_trial)
        Emb->>Emb: mini-batch forward + backward
        Emb->>Metrics: evaluate_head(val_embeddings, val_labels)
        Metrics-->>Emb: HeadEvaluationResult (val_f1, val_pr_auc)
        Emb->>Optuna: pruner_callback(epoch, val_pr_auc)
        Optuna->>Optuna: trial.report(val_pr_auc, step=epoch)
        Optuna-->>Emb: should_prune? True / False
        alt should_prune == True
            Emb-->>Objective: break (early trial termination)
        end
    end

    Emb-->>Objective: history dict
    Objective->>MLflow: log_metrics(val_pr_auc, val_f1, ...)
    Objective->>MLflow: end_run()
    Objective-->>Optuna: return max(history["val_pr_auc"])
```

---

## 7. Remote CI/CD Execution on Bare-Metal K3s (`hive-mind`)

The continuous integration and delivery pipeline defined in [`Jenkinsfile`](file:///home/benni/Documents/antigravity_workspace/taxon-vision-mlops/Jenkinsfile) dispatches the `Model Training & Checkpoint (GPU)` stage whenever commits modify training/pipeline modules or when triggered via the Airflow `model_retraining` DAG (`RETRAIN_MODEL=true`):

```text
1. Prepare Toolchain    -> Ephemeral K3s pod mounts host NVMe Rattler/Pixi cache
2. Static Quality       -> Parallel Ruff lint, MyPy strict type check, Licence audit, OKF validation
3. Pytest Suite         -> Unit & integration tests enforcing >= 80% line coverage
4. Conformal Invariants -> Split conformal coverage mathematical verification
5. Strict Docs Build    -> Knowledge graph validation & strict Zensical compilation
6. GPU Model Training   -> Ephemeral Blackwell GPU pod executes 8-stage DVC pipeline (dvc repro)
7. Rolling Deployment   -> Zero-downtime rollout of deployment/taxon-vision-api + smoke test
```

### Hardware & Storage Integration on `hive-mind`

* **Blackwell GPU & CUDA 12.8 / Driver 13.4 Boundary:** The bare-metal node `hive-mind` is equipped with an Intel Core i7-14700K (8 P-cores, 12 E-cores), 128 GB DDR5 RAM, and an NVIDIA GeForce RTX 5070 Ti (16 GB GDDR7 VRAM, Blackwell compute capability 12.0) running NVIDIA Driver 615.71.09 (CUDA UMD 13.4). Ephemeral GPU training agent pods (`taxon-vision-jenkins-gpu-agent`) request one time-sliced GPU slot (`nvidia.com/gpu: 1`) and execute PyTorch `cu128` kernels pinned to the 16 P-core threads (`taskset -c 0-15`).
* **In-Cluster SeaweedFS Feature Caching:** Intermediate feature tensors (`data/embeddings/*.pt`) are pulled from and pushed to the in-cluster SeaweedFS S3 service (`local-seaweedfs` at `http://seaweedfs.taxon-vision.svc.cluster.local:8333/taxon-dvc-cache`) over NVMe loopback.
* **Four-Gate Promotion & Zero-Downtime Rollout:** Stage 7 (`eval`) evaluates the exported INT8 ONNX model across all four promotion gates ([`promotion.py`](file:///home/benni/Documents/antigravity_workspace/taxon-vision-mlops/src/taxon_vision/models/training/promotion.py)). Upon passing, Jenkins triggers `kubectl rollout restart deployment/taxon-vision-api -n taxon-vision` and runs [`scripts/verify_deployment.sh`](file:///home/benni/Documents/antigravity_workspace/taxon-vision-mlops/scripts/verify_deployment.sh) to verify `/health`, `/health/ready`, and live Prometheus counter increments (`taxon_predictions_total`).

---

## 8. API Reference

::: taxon_vision.models.training.types
::: taxon_vision.models.training.embeddings
::: taxon_vision.models.training.runner
::: taxon_vision.models.training.tuning
::: taxon_vision.pipelines.extract_features
::: taxon_vision.pipelines.train
