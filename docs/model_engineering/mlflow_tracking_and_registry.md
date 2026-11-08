---
type: System Pattern
title: MLflow 3.x Experiment Tracking, Lineage & PyFunc Registry
status: stable
stale_after: "2027-01-01T00:00:00Z"
version: 1.0
description: Self-hosted MLflow 3.16.0 tracking server, 3-tier URI resolution, cross-stage DVC run resumption, 5-dataset lineage, holdout visualisations, and TaxonVisionPyFuncModel packaging.
tags: [mlflow, tracking, registry, lineage, pyfunc, visualizations, dvc, evaluation]
generated: {by: "process:docs-librarian", at: "2026-11-08T17:20:00Z"}
verified: {by: "process:docs-librarian", at: "2026-11-08T17:20:00Z"}
sources:
  - resource: "src/taxon_vision/models/training/tracking.py"
  - resource: "src/taxon_vision/models/training/visualizations.py"
  - resource: "src/taxon_vision/inference/mlflow_model.py"
  - resource: "src/taxon_vision/pipelines/evaluate.py"
  - resource: "deploy/k8s/mlops-storage-and-optuna.yaml"
---

# MLflow 3.x Experiment Tracking, Lineage & PyFunc Registry

## 1. Conceptual Motivation: Why Multi-Process Pipelines Break Naive Experiment Tracking

For a newcomer to MLOps, an experiment tracker such as MLflow is often introduced as a simple logging dictionary inside a single Python training script: open a run (`with mlflow.start_run():`), log the learning rate and validation accuracy, save the PyTorch weights, and close the run. While adequate for a standalone notebook, that naive pattern breaks down in production MLOps pipelines for three structural reasons:

1. **Multi-Process DVC Fragmentation:** Data Version Control (DVC) executes each stage of `dvc.yaml` (`train`, `export`, `calibrate`, `eval`) as a completely independent operating system subprocess. When Stage 4 (`train`) finishes training the classification head, its Python interpreter exits and closes the MLflow run. If Stage 5 (`export_onnx`), Stage 6 (`calibrate`), and Stage 7 (`evaluate`) each opened their own new runs, a single model's lifecycle would be scattered across four disconnected runs in the MLflow UI - making it impossible to inspect training loss curves, INT8 quantisation fidelity, conformal quantile $\hat{q}_\alpha$, and holdout reliability diagrams on a single unified model card.
2. **Missing Cryptographic Dataset Lineage:** Logging a scalar metric (`val_pr_auc = 0.996`) is meaningless if an auditor cannot determine *which exact rows* of `dataset_manifest.parquet` and *which exact split tensors* produced that score. Without binding `mlflow.data` dataset digests directly to the run and the `LoggedModel` entity (`m-...`), active learning feedback merges cannot be traced to downstream model versions.
3. **Un-Calibrated Raw Checkpoint Packaging:** Registering a raw PyTorch `nn.Module` or bare `.onnx` file in the Model Registry omits the epistemic safety layer. A raw ONNX graph outputs uncalibrated float32 logits (`[batch, 10]`); it knows nothing of binomial scientific names (`Danaus plexippus`), Split Conformal prediction sets $C(X)$, or negative Helmholtz free energy Out-of-Distribution (OOD) rejection thresholds ($\tau_{\text{OOD}}$).

To resolve all three challenges, TaxonVision-MLOps implements an end-to-end **MLflow 3.16.0 Tracking & Registry Architecture** featuring cross-stage run resumption (`resume_parent_mlflow_run`), five-dataset cryptographic lineage binding, automated holdout visualisations (`visualizations.py`), and a self-contained `TaxonVisionPyFuncModel` wrapper (`mlflow_model.py`).

```mermaid
flowchart TD
    subgraph Infra["Self-Hosted MLflow 3.16.0 Infrastructure"]
        direction LR
        MLServer["mlflow-server (:5000 / NodePort :30500)<br/>(--serve-artifacts proxy)"]
        PG[("PostgreSQL 16 (mlflow DB)<br/>Runs, LoggedModels & Aliases")]
        SW[("SeaweedFS S3 (:8333)<br/>s3://mlflow-artifacts")]
        SQLite[("Local SQLite Fallback<br/>sqlite:///data/mlflow/mlflow.db")]
        MLServer -->|Backend Store| PG
        MLServer -->|Artefact Proxy| SW
    end

    subgraph DVCContinuity["Cross-Stage DVC Run Continuity (Single Unified Run)"]
        direction TB
        S4["Stage 4: train.py<br/>mlflow_run_scope() creates Run & LoggedModel (m-...)<br/>Logs 5-Dataset Lineage & Epoch Curves"]
        JSON["reports/train_metrics.json<br/>Persists mlflow_run_id & mlflow_model_id"]
        S5["Stage 5: export_onnx.py<br/>resume_parent_mlflow_run()<br/>Logs model_int8.onnx & Quantisation Metrics"]
        S6["Stage 6: calibrate.py<br/>resume_parent_mlflow_run()<br/>Logs q_hat, tau_ood & Calibration Coverage"]
        S7["Stage 7: evaluate.py<br/>resume_parent_mlflow_run()<br/>Logs 6 Holdout Plots, Audit Table, PyFunc & 4-Gate Verdict"]

        S4 -->|Writes| JSON
        JSON -->|Re-opens Run| S5
        JSON -->|Re-opens Run| S6
        JSON -->|Re-opens Run| S7
    end

    DVCContinuity -->|3-Tier URI Resolution| MLServer
    DVCContinuity -.->|Tier 3 Offline Fallback| SQLite
```

---

## 2. Self-Hosted MLflow 3.16.0 Server & Three-Tier URI Resolution

### 2.1 Containerised Server & SeaweedFS Artefact Proxy

In both local Docker Compose ([`docker-compose.yml`](file:///home/benni/Documents/antigravity_workspace/taxon-vision-mlops/docker-compose.yml)) and bare-metal Kubernetes ([`deploy/k8s/mlops-storage-and-optuna.yaml`](file:///home/benni/Documents/antigravity_workspace/taxon-vision-mlops/deploy/k8s/mlops-storage-and-optuna.yaml)), TaxonVision-MLOps runs a dedicated **`mlflow-server` (`ghcr.io/mlflow/mlflow:v3.16.0`)** container:

* **Relational Backend Store (`--backend-store-uri`):** Connects to the `mlflow` database on the consolidated PostgreSQL 16 instance (`postgresql+psycopg2://mlflow:...@postgres:5432/mlflow`), persisting runs, metrics, parameters, `LoggedModel` entities, and Model Registry aliases with full ACID concurrency.
* **Proxied S3 Artefact Destination (`--serve-artifacts` & `--artifacts-destination s3://mlflow-artifacts`):** Rather than requiring every client pod or developer script to hold S3 credentials, the `mlflow-server` daemon acts as an authenticated artefact proxy in front of the in-cluster SeaweedFS S3 bucket (`http://seaweedfs:8333`). Clients upload figures, HTML reports, and model bundles directly to the MLflow HTTP server on port `5000` (exposed on K3s NodePort `30500`), and the server streams the binary payloads into SeaweedFS over the internal cluster network.

### 2.2 Three-Tier Reachable URI Resolution (`_resolve_reachable_tracking_uri`)

While CI pipelines and cluster pods communicate with `http://mlflow-server:5000`, offline unit tests and air-gapped developer workstations must execute pipeline stages without crashing or hanging on network timeouts. The resolver [`_resolve_reachable_tracking_uri()`](file:///home/benni/Documents/antigravity_workspace/taxon-vision-mlops/src/taxon_vision/models/training/tracking.py#L179-L230) in [`src/taxon_vision/models/training/tracking.py`](file:///home/benni/Documents/antigravity_workspace/taxon-vision-mlops/src/taxon_vision/models/training/tracking.py) enforces a deterministic three-tier cascade:

1. **Tier 1 (Explicit Environment Override):** If `MLFLOW_TRACKING_URI` is set in the process environment, it is returned immediately.
2. **Tier 2 (Bounded HTTP Health Probe):** If `settings.mlflow.tracking_uri` (`http://127.0.0.1:5000` by default) starts with `http://` or `https://`, the resolver dispatches an HTTP `GET` request to `<tracking_uri>/health` bounded by `health_check_timeout_seconds = 1.0`. If the server responds with HTTP `2xx` within 1 second, the remote tracking URI is selected.
3. **Tier 3 (Zero-Drop SQLite Offline Fallback):** If the health probe times out or refuses connection, the resolver automatically provisions `data/mlflow/artifacts/`, logs a structured warning, and falls back to `settings.mlflow.fallback_sqlite_uri` (`sqlite:///data/mlflow/mlflow.db`). No experiment run is ever lost due to a stopped Docker container.

---

## 3. RAII Run Lifecycle & Cross-Stage DVC Continuity

### 3.1 Deterministic Context Management (`mlflow_run_scope`)

In compliance with Rule 08 (RAII Lifecycle Management), manual boolean state flags (`is_active: bool`) are prohibited. Every experiment run is scoped via the `@contextmanager` [`mlflow_run_scope()`](file:///home/benni/Documents/antigravity_workspace/taxon-vision-mlops/src/taxon_vision/models/training/tracking.py#L232-L303):

* On entry, it resolves the reachable tracking URI, sets the target experiment (`taxon-vision-head-training` or `taxon-vision-drift`), starts the run, logs the DVC dataset lock hash (`dvc_dataset_hash`), and calls [`_inject_lineage_tags()`](file:///home/benni/Documents/antigravity_workspace/taxon-vision-mlops/src/taxon_vision/models/training/tracking.py#L153-L177) to attach `git.commit_sha`, `git.branch`, `ci.jenkins_build_tag`, and `ci.github_run_id`.
* If an unhandled exception occurs inside the `with` block, the context manager marks the run as `FAILED` via `mlflow.end_run(status="FAILED")` before re-raising the exception, preventing orphaned `RUNNING` zombie runs in the database.
* On normal exit, it attaches any execution log file artefact and closes the run as `FINISHED`.

### 3.2 Cross-Stage Run Resumption (`resume_parent_mlflow_run`)

To unify the multi-process DVC pipeline into a single cohesive MLflow run:

1. During Stage 4 (`train`), `runner.py` records the active `mlflow_run_id` and the MLflow 3 `mlflow_model_id` (`m-...` returned by `MLflowTracker.log_model`) inside `reports/train_metrics.json`.
2. When Stage 5 (`export_onnx.py`), Stage 6 (`calibrate.py`), and Stage 7 (`evaluate.py`) execute as separate DVC subprocesses, each enters the `@contextmanager` [`resume_parent_mlflow_run("reports/train_metrics.json")`](file:///home/benni/Documents/antigravity_workspace/taxon-vision-mlops/src/taxon_vision/models/training/tracking.py#L305-L373).
3. `resume_parent_mlflow_run` deserialises `reports/train_metrics.json`, constructs an immutable [`ParentRunContext(run_id, model_id, extractor, status)`](file:///home/benni/Documents/antigravity_workspace/taxon-vision-mlops/src/taxon_vision/models/training/tracking.py#L62-L83), re-opens the existing run via `mlflow.start_run(run_id=ctx.run_id)`, yields the context to the stage, and cleanly closes the run as `FINISHED` upon completion.

---

## 4. Five-Dataset Cryptographic Lineage & `LoggedModel` Binding

During Stage 4 (`train`), [`_log_dataset_lineage()`](file:///home/benni/Documents/antigravity_workspace/taxon-vision-mlops/src/taxon_vision/models/training/tracking.py#L463-L559) registers all five canonical data inputs into the active MLflow run and returns a frozen [`LoggedDatasets`](file:///home/benni/Documents/antigravity_workspace/taxon-vision-mlops/src/taxon_vision/models/training/tracking.py#L85-L102) container:

| Dataset Identifier | MLflow Context Tag | Constructor & Source | Attached Provenance Metadata |
| :--- | :--- | :--- | :--- |
| `inaturalist_open_manifest` | `context="manifest"` | `mlflow.data.from_pandas(df, source="data/manifests/dataset_manifest.parquet", targets="taxon_id")` | `sha256` hex digest, `license.policy="CC0,CC-BY,CC-BY-NC"`, `records_count="1646"` |
| `train_embeddings` | `context="training"` | `mlflow.data.from_numpy(features=train_embeddings, targets=train_labels)` | 70% stratified training partition ($1{,}153 \times 384$) |
| `val_embeddings` | `context="validation"` | `mlflow.data.from_numpy(features=val_embeddings, targets=val_labels)` | 10% stratified validation partition ($164 \times 384$) |
| `cal_embeddings` | `context="calibration"` | `mlflow.data.from_numpy(features=cal_embeddings, targets=cal_labels)` | 10% conformal calibration partition ($164 \times 384$; $155$ inliers) |
| `test_embeddings` | `context="test"` | `mlflow.data.from_numpy(features=test_embeddings, targets=test_labels)` | 10% unbiased holdout evaluation partition ($164 \times 384$) |

Furthermore, [`_log_bound_evaluation_metrics()`](file:///home/benni/Documents/antigravity_workspace/taxon-vision-mlops/src/taxon_vision/models/training/tracking.py#L391-L415) passes `model_id=model_id` and `dataset=val_ds` directly to `mlflow.log_metrics(...)`, linking validation curves and holdout scores to both the `LoggedModel` entity and the specific dataset split in the MLflow 3.x UI.

---

## 5. Holdout Evaluation, Diagnostic Plots & Per-Taxon Audit Table

When Stage 7 ([`src/taxon_vision/pipelines/evaluate.py`](file:///home/benni/Documents/antigravity_workspace/taxon-vision-mlops/src/taxon_vision/pipelines/evaluate.py)) resumes the parent MLflow run, it executes a comprehensive holdout verification suite over `test_embeddings.pt`:

### 5.1 Standardized `mlflow.models.evaluate()` Integration

Stage 7 invokes `mlflow.models.evaluate()` against a custom callable wrapping the quantised ONNX inference engine with `evaluator_config={"default": {"log_explainer": False, "metric_prefix": "test_"}}`.

> [!TIP]
> **Why `log_explainer=False` Is Enforced:** By default, MLflow's built-in evaluator attempts to run a KernelSHAP permutation explainer across every input feature column. Over a 384-dimensional ViT-S embedding vector, KernelSHAP triggers $\mathcal{O}(2^{384})$ coalition sampling, stalling CPU evaluation for several minutes while providing zero biological interpretability on abstract latent dimensions (visual explainability is instead provided by our sub-5ms `ForwardHookCAM` over spatial image patches). Disabling the SHAP kernel explainer keeps Stage 7 execution under 3 seconds.

### 5.2 Per-Taxon Holdout Audit Table (`mlflow.log_table`)

Aggregate metrics such as Macro PR-AUC can mask degraded performance on an individual species. Stage 7 computes per-species precision, recall, F1-score, average precision (AP), empirical conformal coverage, and mean prediction set size across all 10 target taxa, persists the structured report to `reports/per_taxon_holdout_audit.json`, and logs it to the MLflow run via `mlflow.log_table(..., artifact_file="reports/per_taxon_holdout_audit.json")` for interactive sorting in the MLflow UI.

### 5.3 Six Publication-Grade Diagnostic Figures (`visualizations.py`)

Using the frozen [`HoldoutVisualizationPayload`](file:///home/benni/Documents/antigravity_workspace/taxon-vision-mlops/src/taxon_vision/models/training/visualizations.py#L49-L86) dataclass and the structural [`ParetoProfileLike`](file:///home/benni/Documents/antigravity_workspace/taxon-vision-mlops/src/taxon_vision/models/training/visualizations.py#L27-L47) protocol (which eliminates `getattr`/`hasattr` reflection per Rule 00), [`log_holdout_visualizations()`](file:///home/benni/Documents/antigravity_workspace/taxon-vision-mlops/src/taxon_vision/models/training/visualizations.py#L457-L531) renders and logs six diagnostic figures under the `visualizations/` artefact directory:

1. **`visualizations/test_conformal_set_sizes.png` (`_plot_conformal_set_sizes`):** Histogram of holdout conformal prediction set cardinalities $|C(X)|$ annotated with the human referral boundary ($k_{\text{max}} = 3$) and empirical vs nominal ($95\%$) coverage.
2. **`visualizations/test_reliability_diagram.png` (`_plot_reliability_diagram`):** 10-bin confidence calibration curve plotted against the ideal diagonal alongside the red calibration gap bars and Expected Calibration Error ($\text{ECE}$).
3. **`visualizations/test_energy_distribution.png` (`_plot_energy_distribution`):** Histogram of negative Helmholtz free energy scores $E(\mathbf{x}; T)$ annotated with the calibrated 95% TPR epistemic cutoff $\tau_{\text{OOD}}$.
4. **`visualizations/test_confusion_matrix.png` (`_plot_confusion_matrix`):** Row-normalised $10 \times 10$ taxonomic confusion matrix across the holdout split.
5. **`visualizations/test_precision_recall_curves.png` (`_plot_precision_recall_curves`):** Per-species Precision-Recall trajectories with individual Average Precision ($\text{AP}$) scores in the legend.
6. **`visualizations/pareto_frontier.png` (`_plot_pareto_frontier`):** Scatter plot of p95 ONNX inference latency (ms) vs throughput (FPS) across candidate backbones (`dinov3`, `bioclip-2`, `mobilenetv4_conv_small`), with marker area proportional to parameter count and a vertical red line marking the latency SLO ceiling.

---

## 6. Self-Contained `TaxonVisionPyFuncModel` Wrapper

To ensure that any model version downloaded from the MLflow Registry contains the complete epistemic safety pipeline - not merely raw matrix weights - [`src/taxon_vision/inference/mlflow_model.py`](file:///home/benni/Documents/antigravity_workspace/taxon-vision-mlops/src/taxon_vision/inference/mlflow_model.py) implements [`TaxonVisionPyFuncModel(mlflow.pyfunc.PythonModel)`](file:///home/benni/Documents/antigravity_workspace/taxon-vision-mlops/src/taxon_vision/inference/mlflow_model.py#L286-L439):

* **Bundled Artefact Context (`load_context`):** Packages three physical artefacts inside the MLflow `pyfunc_model` directory:
    * `onnx_model`: `models/checkpoints/model_int8.onnx` (loaded into `ONNXInferenceEngine`).
    * `conformal_config`: `config/conformal_config.yaml` (deserialising $\hat{q}_\alpha$, $\alpha$, $\tau_{\text{OOD}}$, $\tau_{\text{conf}}$, and $k_{\text{max}}$ into `ConformalPredictionEngine` and `EnergyOODDetector`).
    * `taxa_catalog`: `config/taxa_catalog.yaml` (mapping integer class indices $[0, K-1]$ to scientific and vernacular names).
* **Polymorphic Input Ingestion (`_preprocess_input_batch`):** Accepts 4D normalised image tensors (`[B, 3, 224, 224]`), 2D pre-extracted embedding matrices (`[B, 384]`), lists of raw JPEG bytes or `PIL.Image` objects, or pandas DataFrames containing base64-encoded image strings.
* **Epistemic Output Contract (`predict`):** Executes the zero-allocation ONNX forward pass, applies temperature-scaled softmax, evaluates Helmholtz free energy OOD gating, constructs the Split Conformal prediction set $C(X) = \{k : 1 - \hat{\pi}_k(\mathbf{x}) \le \hat{q}_\alpha\}$, and returns a structured `pandas.DataFrame` with columns:
    * `top1_taxon`, `top1_common_name`, `top1_confidence`
    * `conformal_set` (`list[str]`), `set_size` (`int`)
    * `ood_energy` (`float`), `is_ood` (`bool`), `requires_review` (`bool`)
