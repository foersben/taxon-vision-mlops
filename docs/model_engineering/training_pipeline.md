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
---

# Classification Head Training Pipeline

The training module (`taxon_vision.models.training`) executes a four-stage pipeline
that trains a lightweight linear classification head on **pre-cached feature embeddings**,
tracks all experiments with MLflow, and automatically promotes the best model using
**Macro PR-AUC** as the sole promotion metric.

## Design Principles

* **Zero backbone backpropagation.** All training iterates only over the
  linear head (`Dropout` + `Linear`), operating on embeddings pre-extracted by the
  frozen backbone. This reduces a full training epoch to milliseconds.
* **Single promotion metric.** Macro PR-AUC is the authoritative comparison signal.
  It is robust to severe class imbalance (characteristic of species observation data)
  and does not require threshold selection.
* **Git-free DVC versioning.** Dataset hashes are sourced by parsing `dvc.lock` directly.
  No `git` commands or `.git` state are required, satisfying the zero-allocation
  inference constraint in sandboxed environments.

## Module Map

| Module | Key Symbol | Responsibility |
| --- | --- | --- |
| `types.py` | `EmbeddingSplit` | Validated data container: train/val tensors + class counts |
| `types.py` | `HeadTrainingConfig` | Epoch budget, batch size, and Optuna pruner callback |
| `types.py` | `TuningConfig` | Optuna trial budget, seed, and MLflow coordinates |
| `embeddings.py` | `train_head_on_cached_embeddings()` | Inner training loop with per-epoch metric history |
| `embeddings.py` | `_compute_additional_metrics()` | Macro F1 and Macro PR-AUC from logits |
| `tuning.py` | `HyperparameterObjective` | Optuna callable wrapping the inner training loop |
| `tuning.py` | `tune_hyperparameters()` | Orchestrates Optuna study with ASHA pruning |
| `runner.py` | `run_training_pipeline()` | Top-level orchestrator called by the FastAPI endpoint |
| `tracking.py` | `_get_dvc_dataset_hash()` | Parses `dvc.lock` for reproducible dataset provenance - **planned: delegate to Jenkins** |
| `tracking.py` | `mlflow_run_scope()` | Context manager that opens parent MLflow run and logs training parameters |
| `tracking.py` | `_promote_model_if_better()` | Compares PR-AUC and assigns Production / Challenger alias - **planned: delegate to Jenkins** |
| `runner.py` | `_dispatch_remote_training()` | HTTP dispatch to FastAPI to trigger training remotely - **planned: delete, replaced by Jenkins trigger** |

## Pipeline Lifecycle & MLflow RAII Strategy

**What we chose:**
The MLflow run lifecycle is strictly managed via a custom `@contextmanager` named `mlflow_run_scope()`. This native Python RAII (Resource Acquisition Is Initialization) pattern guarantees that if the training pipeline crashes (e.g., CUDA OOM), the exception is caught, the run is explicitly marked as `FAILED` in the MLflow tracking registry, and the run is cleanly closed before bubbling the exception up to the FastAPI route.

**Why we chose it (Semantic Validation):**
A common anti-pattern is tracking state via detached boolean flags (e.g., `is_active: bool`) across multiple methods. If an unhandled exception occurs in a deep submodule, the top-level script crashes without closing the MLflow run. This leaves orphaned, zombie runs stuck in a perpetual `RUNNING` state in the MLflow UI, requiring manual DB cleanup and breaking downstream automation that polls for run completion. The RAII context manager semantically links the scope of the Python execution block directly to the MLflow lifecycle, ensuring perfect state consistency.

## Full Pipeline Sequence

```mermaid
sequenceDiagram
    autonumber
    participant EP as FastAPI /training
    participant Runner as runner.py
    participant DVC as dvc.lock (filesystem)
    participant MLflow as MLflow Tracking Server
    participant Emb as embeddings.py
    participant Optuna as tuning.py / Optuna

    EP->>Runner: run_training_pipeline(extractor, epochs, ...)
    Runner->>Runner: _resolve_feature_dim(extractor)
    Runner->>Runner: _prepare_embedding_split(dim, num_classes)
    Runner->>Runner: _init_mlflow_run(...)
    Runner->>DVC: open dvc.lock
    DVC-->>Runner: YAML with stages.ingestion.outs[*].md5
    Runner->>MLflow: log_param("dvc_dataset_hash", md5)
    MLflow-->>Runner: run_id

    Runner->>Emb: train_head_on_cached_embeddings(head, data, optimizer, criterion, config)
    loop epoch in range(epochs)
        Emb->>Emb: mini-batch SGD on train_emb
        Emb->>Emb: _compute_additional_metrics(val_logits, val_lbl)
        Emb-->>Runner: history[val_pr_auc, val_f1, val_accuracy, ...]
    end

    Runner->>Runner: _finalize_mlflow_run(history, duration)
    Runner->>Runner: _promote_model_if_better(head, val_pr_auc)
    Runner->>MLflow: pytorch.log_model(head, registered_model_name)
    MLflow-->>Runner: model_version

    alt val_pr_auc > production_score
        Runner->>MLflow: set_alias("Challenger", version)
    else no production exists
        Runner->>MLflow: set_alias("Production", version)
    else val_pr_auc <= production_score
        Runner->>MLflow: set_alias("Archived", version)
    end

    Runner-->>EP: dict{status, metrics, checkpoint_path}
```

## Inner Training Loop

`train_head_on_cached_embeddings()` in `embeddings.py` is the hot-path function.
It receives pre-built tensors from `EmbeddingSplit` and runs a standard mini-batch
SGD loop. After each epoch, it computes three validation metrics:

1. **Top-1 accuracy** - quick sanity check, logged but not used for promotion.
2. **Macro F1-Score** - average F1 across all classes.
3. **Macro PR-AUC** - primary optimization and promotion metric.

The optional `HeadTrainingConfig.pruner_callback(epoch, val_pr_auc) -> bool` is invoked
at the end of every epoch. During hyperparameter tuning, Optuna injects a callback that
reports the current PR-AUC to the ASHA pruner and signals early stopping if the trial
is statistically unlikely to beat the current best.

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
    Objective-->>Optuna: return history["val_pr_auc"][-1]
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
        +epochs: int = 10
        +batch_size: int = 64
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

## Planned Orchestration Transition

Prior to modularization, `runner.py` acted as a God script. We have already initiated the separation of concerns by moving MLflow and DVC tracking logic into `tracking.py`.
However, the pipeline logic still relies heavily on Python orchestration and does not scale perfectly when compute
jobs need to run in dedicated Kubernetes pods.

The planned transition introduces a **two-layer CI architecture**:

### Layer 1: GitHub Actions (lightweight gates - current)

All CPU-bound quality gates continue to run on GitHub Actions via ARC runner pods:

* Lint (`ruff`), type checking (`mypy`), pre-commit hooks.
* Unit and integration tests on the `ci-dev` (CPU) Pixi environment.
* License compliance audit and OKF frontmatter validation.
* Conformal coverage audit.
* Docker image build and push.
* DVC data push / pull.

On merge to `main`, GitHub Actions fires a webhook to trigger Jenkins:

```yaml
# End of GitHub Actions CI workflow
- name: Trigger Jenkins training pipeline
  if: github.ref == 'refs/heads/main'
  run: |
    curl -X POST "${{ secrets.JENKINS_URL }}/job/taxon-vision-train/build" \
      --user "${{ secrets.JENKINS_USER }}:${{ secrets.JENKINS_TOKEN }}"
```

### Layer 2: Jenkins (heavy compute - planned)

Jenkins runs on the bare-metal cluster with direct access to the GPU pod pool.
Each stage runs in an isolated Kubernetes pod, requesting exact resources:

```text
stage('Ingest')    -> CPU pod  - DVC pull, extract dataset hash from dvc.lock
stage('Train')     -> GPU pod  - run_training_pipeline() (pure ML, no promotion logic)
stage('Evaluate')  -> CPU pod  - query MLflow, compare PR-AUC against Production alias
stage('Promote')   -> CPU pod  - set_registered_model_alias() Challenger / Production
stage('Deploy')    -> CPU pod  - rolling restart of FastAPI Kubernetes deployment
```

### Impact on Python Orchestration

Once Jenkins is in place, the following functions become dead code and will be deleted
from `tracking.py` and `runner.py`, reducing the orchestration footprint further:

* `_get_dvc_dataset_hash()` (in `tracking.py`) - moved to the Jenkins `Ingest` stage as a bash step.
* `_promote_model_if_better()` (in `tracking.py`) - moved to the Jenkins `Evaluate` and `Promote` stages.
* `_dispatch_remote_training()` (in `runner.py`) - deleted entirely; Jenkins triggers jobs directly.
* `train_cli()` argument complexity - simplified to just `extractor`, `epochs`, `batch_size`, `lr`.

The result is a Python training package that only a Data Scientist needs to read: PyTorch modules,
loss functions, and metric evaluation. All pipeline wiring lives in the `Jenkinsfile`.

## API Reference

::: taxon_vision.models.training.types
::: taxon_vision.models.training.embeddings
::: taxon_vision.models.training.runner
::: taxon_vision.models.training.tuning
