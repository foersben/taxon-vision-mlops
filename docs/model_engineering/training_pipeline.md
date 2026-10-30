---
type: System Pattern
title: Classification Head Training Pipeline
status: stable
stale_after: "2027-01-01T00:00:00Z"
version: 1.0
description: End-to-end training pipeline orchestration covering embedding caching, Optuna hyperparameter tuning, MLflow tracking, DVC dataset versioning, and automated PR-AUC model promotion.
tags: [training, mlflow, optuna, dvc, embeddings, promotion, pr-auc]
generated: {by: process:docs-librarian, at: "2026-10-06T20:00:00Z"}
verified: {by: process:docs-librarian, at: "2026-10-06T20:00:00Z"}
sources:
  - resource: "src/taxon_vision/models/training/runner.py"
  - resource: "src/taxon_vision/models/training/tracking.py"
  - resource: "src/taxon_vision/models/training/checkpoint.py"
  - resource: "src/taxon_vision/models/training/embeddings.py"
  - resource: "src/taxon_vision/models/training/tuning.py"
  - resource: "src/taxon_vision/models/training/types.py"
  - resource: "src/taxon_vision/pipelines/extract_features.py"
  - resource: "src/taxon_vision/pipelines/train.py"
---

# Classification Head Training Pipeline

The training module (`taxon_vision.models.training`) executes a four-stage pipeline
that trains a lightweight linear classification head on **pre-cached feature embeddings**,
tracks all experiments with MLflow, and automatically promotes the best model using
**Macro PR-AUC** as the sole promotion metric.

## Design Principles

* **Pure remote CI/CD execution paradigm.** The local developer workstation has zero dedicated GPU resources for model training and serves strictly as an authoring, testing, and orchestration plane. All PyTorch model training, Optuna hyperparameter sweeps, and ONNX graph serialization execute remotely on bare-metal CI/CD agents (`hive-mind`) inside isolated Kubernetes pods equipped with GPU time-slicing slots and VRAM limits.
* **Zero backbone backpropagation.** All training iterates only over the linear head (`Dropout` + `Linear`), operating on embeddings pre-extracted by the frozen backbone (`vit_base_patch14_reg4_dinov2`, `bioclip-2`, `mobilenetv4_conv_small`). This reduces a full training epoch to milliseconds.
* **Class-Balanced Loss for Long-Tail Distributions.** The loss formulation is defined as:

    $$\mathcal{L}_{\text{CB}}(p, y) = -\frac{1 - \beta}{1 - \beta^{n_y}} \log(p_y)$$

    Where $\beta \to 1$ approximates standard class-frequency inverse weighting and intermediate $\beta \in [0.99, 0.9999]$ is optimal for long-tail species classification.

* **Multi-Head Hierarchical Taxonomic Loss.** Rather than relying solely on a single-head flat classification over terminal leaf nodes, the system can optionally project embeddings into a multi-head taxonomy (Class, Order, Family, Genus, Species) ensuring coarse-grained taxonomic errors are penalized far more heavily than fine-grained errors.
* **Single promotion metric.** Macro PR-AUC is the authoritative comparison signal. It is robust to severe class imbalance (characteristic of species observation data) and does not require threshold selection.
* **DagsHub Integrated Ecosystem.** DVC remote storage (`s3://dvc`), centralized MLflow experiment tracking (`https://dagshub.com/foersben/taxon-vision-mlops.mlflow`), and hosted Label Studio are unified under DagsHub. Credentials are authenticated securely in memory via Linux Secret Service without persistent plaintext tokens on disk.
* **Git-free DVC versioning.** Dataset hashes are sourced by parsing `dvc.lock` directly.
  No `git` commands or `.git` state are required, satisfying the zero-allocation
  inference constraint in sandboxed environments.

## Module Map

Following the Rule 08 single-responsibility sibling topology refactoring, the training subsystem in `src/taxon_vision/models/training/` is partitioned into focused, non-monolithic modules (all maintaining cognitive complexity $< 15$):

| Module | Key Symbols | Architectural Responsibility |
| --- | --- | --- |
| `runner.py` | `run_training_pipeline()`, `build_training_components()`, `configure_cuda_memory_budget()` | Top-level CLI training coordinator supporting standard runs and `--tune` Optuna sweeps. |
| `dataset.py` | `load_or_create_training_split()`, `get_dvc_dataset_hash()`, `resolve_feature_dim()` | Tensors hydration, DVC lock hash verification, and synthetic tensor fallbacks. |
| `loops.py` | `train_head_epoch()`, `evaluate_head()`, `HeadEvaluationResult` | Isolated single-epoch mini-batch training and validation passes. |
| `embeddings.py` | `train_head_on_cached_embeddings()` | Multi-epoch training loop coordinating early stopping and learning rate plateau scheduling. |
| `promotion.py` | `_promote_model_if_better()`, `_determine_promotion_alias()` | Evaluates candidate Macro PR-AUC against incumbent and tags MLflow aliases (`Production`, `Challenger`, `Archived`). |
| `tracking.py` | `mlflow_run_scope()`, `MLflowTracker`, `log_stage_metrics_to_parent_run()` | RAII context manager managing MLflow run lifecycle, Git lineage tags, and parameter logging. |
| `tuning.py` | `tune_hyperparameters()`, `HyperparameterObjective`, `TuningResult` | Bayesian hyperparameter optimization study orchestration with TPE sampling and ASHA pruning. |
| `visualizations.py` | `log_training_visualizations()`, `log_optuna_diagnostic_plots()` | Diagnostic figure rendering for confusion matrices, PR curves, and energy distributions. |
| `checkpoint.py` | `save_checkpoint()` | PyTorch model serialization and local checkpoint file export. |
| `types.py` | `ResolvedTrainingParameters`, `EmbeddingSplit`, `HeadTrainingConfig`, `TrainingPipelineResult` | Validated Pydantic models and frozen dataclasses enforcing zero tuple blindness. |

## Pipeline Lifecycle & MLflow RAII Strategy

**What we chose:**
The MLflow run lifecycle is strictly managed via a custom `@contextmanager` named `mlflow_run_scope()`. This native Python RAII (Resource Acquisition Is Initialization) pattern guarantees that if the training pipeline crashes (e.g., CUDA OOM), the exception is caught, the run is explicitly marked as `FAILED` in the MLflow tracking registry, and the run is cleanly closed before bubbling the exception up to the FastAPI route.

**Why we chose it (Semantic Validation):**
A common anti-pattern is tracking state via detached boolean flags (e.g., `is_active: bool`) across multiple methods. If an unhandled exception occurs in a deep submodule, the top-level script crashes without closing the MLflow run. This leaves orphaned, zombie runs stuck in a perpetual `RUNNING` state in the MLflow UI, requiring manual DB cleanup and breaking downstream automation that polls for run completion. The RAII context manager semantically links the scope of the Python execution block directly to the MLflow lifecycle, ensuring perfect state consistency.

## Full Pipeline Architecture & Sibling Collaboration

The diagram below illustrates how `runner.py` coordinates tensor hydration, epoch optimization, Optuna sweeps, and MLflow model registry promotion across sibling modules:

```mermaid
flowchart TD
    subgraph Coordination["Top-Level Orchestration"]
        R["models.training.runner<br/><b>run_training_pipeline()</b><br/>• build_training_components()<br/>• configure_cuda_memory_budget()"]
    end

    subgraph DataAndState["State & Tensor Hydration"]
        DS["models.training.dataset<br/><b>load_or_create_training_split()</b><br/>• get_dvc_dataset_hash()<br/>• resolve_feature_dim()"]
        CK["models.training.checkpoint<br/><b>save_checkpoint()</b>"]
    end

    subgraph Optimization["Epoch Training & Hyperparameter Tuning"]
        LP["models.training.loops<br/><b>train_head_epoch()</b><br/><b>evaluate_head()</b><br/>• returns HeadEvaluationResult"]
        EM["models.training.embeddings<br/><b>train_head_on_cached_embeddings()</b><br/>• best model tracking & early stopping"]
        TN["models.training.tuning<br/><b>tune_hyperparameters()</b><br/>• Optuna TPE & ASHA pruner<br/>• HyperparameterObjective"]
    end

    subgraph GovernanceAndTracking["Lineage, Promotion & Observability"]
        TR["models.training.tracking<br/><b>mlflow_run_scope()</b> (RAII Context)<br/>• MLflowTracker<br/>• log_dataset_lineage()"]
        PR["models.training.promotion<br/><b>_promote_model_if_better()</b><br/>• evaluate against baseline PR-AUC<br/>• assigns Production/Challenger alias"]
        VZ["models.training.visualizations<br/><b>log_training_visualizations()</b><br/>• PR curves, confusion matrix, energy dist"]
    end

    subgraph TypeContracts["Typed Contracts (types.py)"]
        TC["• ResolvedTrainingParameters<br/>• EmbeddingSplit<br/>• HeadTrainingConfig<br/>• TrainingPipelineResult"]
    end

    R -->|Hydrates Tensors| DS
    R -->|Scopes Session RAII| TR
    R -->|Dispatches Optuna Sweeps| TN
    R -->|Executes Training Loop| EM
    EM -->|Delegates Mini-Batches| LP
    R -->|Persists Best State| CK
    R -->|Gates Production Release| PR
    R -->|Renders Diagnostics| VZ
    TypeContracts -.->|Type Safety Protocol| Coordination
    TypeContracts -.->|Type Safety Protocol| Optimization
    TypeContracts -.->|Type Safety Protocol| GovernanceAndTracking
```

## Inner Training Loop & Execution Callbacks

`train_head_on_cached_embeddings()` in `embeddings.py` is the hot-path execution function.
It receives pre-built tensors from `EmbeddingSplit` and runs a mini-batch SGD loop with adaptive metric evaluation and convergence callbacks. After each epoch, it computes three validation metrics:

* **Top-1 accuracy:** Diagnostic check, logged to MLflow but not used for promotion decisions.
* **Macro F1-Score:** Harmonic mean of precision and recall across all modelled taxa.
* **Macro PR-AUC:** Primary optimization and automated model promotion metric.

### Training Callbacks

The inner loop incorporates three convergence callbacks and one tuning hook:

* **Best Model Weights Restoration (`restore_best_weights=True`):** Tracks validation Macro PR-AUC at each epoch. If an epoch yields a new historical high, a deep copy of the model state dict is cached. When training completes (either by reaching the epoch limit or via early stopping), the model weights are automatically restored to the optimal checkpoint.
* **Early Stopping Callback (`early_stopping_patience=4`):** Monitors validation Macro PR-AUC across epochs. If the metric fails to improve for 4 consecutive epochs, training terminates early, preventing overfitting on rare long-tail classes and saving compute cycles.
* **Learning Rate Reduction on Plateau (`reduce_lr_patience=2`):** Wraps the optimizer in `torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)`. When validation PR-AUC plateaus for 2 epochs, the learning rate is halved, allowing fine-grained convergence into sharper minima.
* **Optuna Pruner Callback (`pruner_callback`):** When executing hyperparameter sweeps, Optuna injects a callback that reports intermediate PR-AUC to the ASHA pruner, terminating underperforming trials before full epoch completion.

## Optimal Training Dynamics & Hyperparameters

The following defaults are design guidelines for linear probing on frozen foundation embeddings. The training pipeline operates on pre-extracted feature representations partitioned at `data/embeddings/` by `taxon_vision.pipelines.extract_features` (`train_embeddings.pt`, `val_embeddings.pt`). When no cached embeddings files exist and feature paths are omitted, the runner falls back gracefully to a deterministic synthetic Gaussian split (seed 42) for offline integration testing:

* **Epoch Budget (10-15 Epochs):** Linear probes on frozen embeddings typically converge within 10-15 epochs. Early stopping (patience 4) bounds the effective budget and limits memorisation of long-tail noise.
* **Batch Size Budgeting:**
    * **Batch Size 64 for Cached Embeddings:** Minimizes gradient variance while maximizing SIMD throughput when operating purely on cached feature vectors.
    * **Batch Size 32 for ViT Feature Extraction:** When extracting embeddings from raw images using the RTX 5070 Ti (16 GB GDDR7 VRAM), batch size 32 operates well within the 70% VRAM cap (`TAXON_CUDA_MEMORY_FRACTION=0.7`), reserving headroom for the host system and live inference.
* **Biological Data Augmentation Invariants:**
    * **Horizontal Flips Permitted:** Biological organisms exhibit bilateral or radial symmetry across the horizontal plane; horizontal reflection generates physically valid training exemplars.
    * **Vertical Flips Strictly Prohibited:** Gravity, geotropism, and phototropism define strict environmental orientations in biological imagery. Flipping specimens vertically produces unphysical specimens (e.g. inverted tree canopies or upside-down insects) that distort spatial priors.

## Optuna Bayesian Hyperparameter Optimization

Optuna explores hyperparameter search spaces using the Tree-structured Parzen Estimator (`TPESampler`) coupled with Asynchronous Successive Halving (`SuccessiveHalvingPruner`/ASHA):

* **Search Space Formulation:**
    * **Learning Rate ($\eta$):** Log-uniform distribution over $[10^{-4}, 10^{-1}]$.
    * **Weight Decay ($\lambda$):** Log-uniform distribution over $[10^{-6}, 10^{-2}]$.
    * **Dropout Rate ($p$):** Uniform distribution over $[0.0, 0.5]$.
    * **Class-Balanced Loss $\beta$:** Uniform distribution over $[0.99, 0.9999]$, controlling effective sample re-weighting for long-tail classes.
* **Optimiser:** `AdamW` with the sampled learning rate and weight decay.
* **ASHA Pruning Invariants:** Initialized with `min_resource=2` and `reduction_factor=2`. Trials are first evaluated after 2 epochs; at each subsequent rung only the better half of the trials (a fraction of `1 / reduction_factor`) is promoted.
* **CLI Trigger:**

```bash
pixi run -e dev python -m taxon_vision.pipelines.train --tune --n-trials 15
```

```mermaid
sequenceDiagram
    autonumber
    participant Objective as HyperparameterObjective.__call__
    participant Emb as train_head_on_cached_embeddings
    participant Metrics as _compute_additional_metrics
    participant Optuna as optuna.Trial (ASHA pruner)
    participant MLflow as MLflow nested run

    Objective->>Objective: _sample_hyperparameters(trial)
    Objective->>Objective: _construct_components(params)
    Objective->>Objective: _start_mlflow_trial(trial, params)
    Objective->>MLflow: start_run(nested=True, run_name="trial_N")
    MLflow-->>Objective: active nested run

    Objective->>Emb: train_head_on_cached_embeddings(head, data, optimizer, criterion, config)

    loop epoch in range(epochs_per_trial)
        Emb->>Emb: mini-batch forward + backward
        Emb->>Metrics: _compute_additional_metrics(val_logits, val_lbl)
        Metrics->>Metrics: label_binarize(y_true)
        Metrics->>Metrics: f1_score(macro)
        Metrics->>Metrics: average_precision_score(macro)
        Metrics-->>Emb: (val_f1, val_pr_auc)
        Emb->>Optuna: pruner_callback(epoch, val_pr_auc)
        Optuna->>Optuna: trial.report(val_pr_auc, step=epoch)
        Optuna-->>Emb: should_prune? True / False
        alt should_prune
            Emb-->>Objective: break (early exit)
        end
    end

    Emb-->>Objective: history dict
    Objective->>Objective: _log_trial_metrics(run, history)
    Objective->>MLflow: log_metric(val_pr_auc, val_f1, ...) per epoch
    Objective->>MLflow: end_run()
    Objective-->>Optuna: return max(history["val_pr_auc"])
```

## EmbeddingSplit Data Container

`EmbeddingSplit` is a frozen dataclass that validates tensor shapes at construction time.
It is the single source of truth for both training tensors and class frequency information,
removing parameter clutter from function signatures.

```mermaid
classDiagram
    class EmbeddingSplit {
        +train_embeddings: Tensor  shape(N_train, D)
        +train_labels: Tensor  shape(N_train,)
        +val_embeddings: Tensor  shape(N_val, D)
        +val_labels: Tensor  shape(N_val,)
        +samples_per_class: Sequence[int] or None
        +feature_dim: int  property
        +__post_init__(): validates shapes
        +get_samples_per_class(num_classes): list[int]
    }
    class HeadTrainingConfig {
        +epochs: int = 15
        +batch_size: int = 64
        +early_stopping_patience: int or None = 4
        +reduce_lr_patience: int or None = 2
        +restore_best_weights: bool = True
        +pruner_callback: Callable[[int, float], bool] or None
    }
    class TuningConfig {
        +n_trials: int = 15
        +epochs_per_trial: int = 10
        +batch_size: int = 64
        +seed: int = 42
        +tracking_uri: str or None
        +experiment_name: str
    }
    EmbeddingSplit --> HeadTrainingConfig : consumed by
    EmbeddingSplit --> TuningConfig : consumed by
```

## Model Promotion Strategy

Promotion is fully automated and uses a single metric to guarantee consistent comparison:

| Condition | Assigned Alias |
| --- | --- |
| No model with `Production` alias exists | `Production` |
| `new_pr_auc > production_pr_auc` | `Challenger` |
| `new_pr_auc <= production_pr_auc` | `Archived` |

**Why Macro PR-AUC?** Accuracy and F1 both require threshold decisions and are sensitive
to class imbalance. PR-AUC integrates precision-recall trade-offs across all thresholds
and all classes without normalization bias, making it the most reliable single-number
summary for the long-tailed taxonomic classification task.

## Remote CI/CD Execution & Automated Training on Cluster

To prevent local workstation compute saturation and guarantee strict mathematical reproducibility, all model training, hyperparameter sweeps, and ONNX compilation tasks execute remotely on the bare-metal Kubernetes cluster (`hive-mind`). The local workstation operates purely as an authoring, linting, and orchestration plane.

### The Automated CI/CD Lifecycle

The continuous integration and delivery pipeline defined in [`Jenkinsfile`](file:///home/benni/Documents/antigravity_workspace/taxon-vision-mlops/Jenkinsfile) automatically executes the `Model Training & Checkpoint (GPU)` stage when commits alter relevant modules (`data/**`, `src/taxon_vision/models/**`, `src/taxon_vision/pipelines/**`) or when manually dispatched with `RETRAIN_MODEL = true`:

```text
1. Prepare Toolchain    -> Ephemeral k3s pod mounts host NVMe Rattler cache
2. Static Quality       -> Parallel Ruff lint, MyPy strict type check, License audit, OKF validation
3. Pytest Suite         -> Unit tests enforcing >= 80% line coverage
4. Conformal Invariants -> Split conformal coverage empirical validation (1 - alpha >= 0.90)
5. Strict Docs Build    -> Knowledge graph validation & strict Zensical MkDocs compilation
6. GPU Model Training   -> Ephemeral Blackwell GPU runner executes unified 7-stage DVC pipeline
7. Rolling Deployment   -> Zero-downtime rollout restart of deployment/taxon-vision-api
```

### GPU Time-Slicing, Blackwell Toolchain & In-Cluster Storage

During Stage 6, the runner interacts directly with the cluster's physical topology:

* **Blackwell Toolchain & CUDA 13.4 Forward Compatibility:** The bare-metal host `hive-mind` operates an NVIDIA GeForce RTX 5080 (Blackwell architecture, SM 120, 16 GB GDDR7) powered by NVIDIA Driver 615.71.09 establishing a CUDA 13.4 User-Mode Driver (UMD) boundary. Container runtimes and Pixi environments compile kernels using PyTorch 2.7.0 targeting CUDA 12.8 (`cu128`), executing natively via driver forward-compatibility.
* **GPU Time-Slicing Slot Allocation:** The physical GPU is partitioned into 4 virtual GPU slots via the NVIDIA Kubernetes Device Plugin. The ephemeral GPU training agent pod (`taxon-vision-jenkins-gpu-agent`) requests `nvidia.com/gpu: 1` in both resource limits and requests, guaranteeing non-preemptive access to hardware tensor cores.
* **In-Cluster SeaweedFS Feature Caching:** The runner injects credentials for the in-cluster SeaweedFS remote (`local-seaweedfs` at `http://seaweedfs.taxon-vision.svc.cluster.local:8333`). Feature tensors (`data/embeddings/*.pt`) are retrieved or populated at NVMe wire speed, eliminating redundant Vision Transformer forward passes during hyperparameter tuning.
* **Persistent Optuna Bayesian Telemetry:** When hyperparameter tuning is activated (`TUNE_HYPERPARAMETERS=true`), trials record their trajectory into an ACID-compliant SQLite database configured in Write-Ahead Logging (WAL) mode (`/workspace/optuna/optuna_studies.db`). The study is rendered in real time on the cluster Optuna Dashboard (`http://<server-ip>:30081`), tracking Tree-structured Parzen Estimator (TPE) parameter importances and ASHA pruning.
* **Cryptographic Release Remote (`origin` DagsHub S3):** Upon convergence, the pipeline pushes the trained head (`models/checkpoints/head.pt`) and quantized runtime (`models/checkpoints/model_int8.onnx`) to DagsHub S3, extracting true dataset MD5 digests directly from `dvc.lock` for MLflow lineage.
* **Zero-Downtime Rolling Update & Health Verification:** Jenkins applies the updated deployment and executes `kubectl rollout restart deployment/taxon-vision-api -n taxon-vision`. Once the deployment achieves ready status, the automated smoke test script ([`scripts/verify_deployment.sh`](file:///home/benni/Documents/antigravity_workspace/taxon-vision-mlops/scripts/verify_deployment.sh)) probes `/health`, `/health/ready`, and dispatches a test prediction to verify that Prometheus counter `taxon_predictions_total` increments properly.

## API Reference

::: taxon_vision.models.training.types
::: taxon_vision.models.training.embeddings
::: taxon_vision.models.training.runner
::: taxon_vision.models.training.tuning
::: taxon_vision.pipelines.extract_features
::: taxon_vision.pipelines.train
