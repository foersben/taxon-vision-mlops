/**
 * TaxonVision-MLOps Interactive Subpackage & Module Codebase Explorer
 */
(function () {
/**
 * Interactive Subpackage & Module Codebase Explorer (Part 1)
 */
function initCodebaseExplorer() {
    const mountPoint = document.getElementById("taxon-codebase-explorer");
    if (!mountPoint) return;
    if (mountPoint.dataset.initialized === "true") return;
    mountPoint.dataset.initialized = "true";

    const modules = [
        {
            subpkg: "taxon_vision.pipelines",
            file: "ingest.py",
            tier: 1,
            tierName: "Tier 1: Orchestration",
            badgeClass: "tier-badge-1",
            complexity: 6,
            dvcStage: "ingest",
            responsibilities: "Queries iNaturalist REST API for target taxa, formats records into DarwinCore observations, and writes raw manifest.",
            classes: "run_ingest(config_path: Path | None = None) -> None",
            methods: "run_ingest(): Entrypoint for DVC stage 'ingest'. Calls INatAPIClient and exports data/manifests/raw_manifest.parquet.",
            github: "https://github.com/foersben/taxon-vision-mlops/blob/develop/src/taxon_vision/pipelines/ingest.py"
        },
        {
            subpkg: "taxon_vision.pipelines",
            file: "audit_license.py",
            tier: 1,
            tierName: "Tier 1: Orchestration",
            badgeClass: "tier-badge-1",
            complexity: 8,
            dvcStage: "audit_license",
            responsibilities: "Audits Creative Commons licenses, ensures photographer attribution retention, and generates validated dataset manifest.",
            classes: "run_audit_license(config_path: Path | None = None) -> None",
            methods: "run_audit_license(): Entrypoint for DVC stage 'audit_license'. Uses LicenseFilter and exports data/manifests/dataset_manifest.parquet.",
            github: "https://github.com/foersben/taxon-vision-mlops/blob/develop/src/taxon_vision/pipelines/audit_license.py"
        },
        {
            subpkg: "taxon_vision.pipelines",
            file: "extract_features.py",
            tier: 1,
            tierName: "Tier 1: Orchestration",
            badgeClass: "tier-badge-1",
            complexity: 11,
            dvcStage: "extract_features",
            responsibilities: "Extracts frozen trunk embeddings via DINOv3, BioCLIP-2, or MobileNetV4 and splits tensors into train/val/cal/test.",
            classes: "run_feature_extraction(config_path: Path | None = None) -> None",
            methods: "run_feature_extraction(): Extracts representations and writes data/embeddings/{train,val,cal,test}_embeddings.pt.",
            github: "https://github.com/foersben/taxon-vision-mlops/blob/develop/src/taxon_vision/pipelines/extract_features.py"
        },
        {
            subpkg: "taxon_vision.pipelines",
            file: "data_drift.py",
            tier: 1,
            tierName: "Tier 1: Orchestration",
            badgeClass: "tier-badge-1",
            complexity: 7,
            dvcStage: "data_drift",
            responsibilities: "Executes Stage 3.5 training representation drift verification (1D Wasserstein-1 on L2 norms + multivariate RKHS MMD^2) before head training.",
            classes: "run_data_drift_check(...) -> DataDriftReport",
            methods: "run_data_drift_check(): Compares reference_train_embeddings.pt against train_embeddings.pt and writes reports/data_drift.json.",
            github: "https://github.com/foersben/taxon-vision-mlops/blob/develop/src/taxon_vision/pipelines/data_drift.py"
        },
        {
            subpkg: "taxon_vision.pipelines",
            file: "calibrate_drift.py",
            tier: 1,
            tierName: "Tier 1: Orchestration",
            badgeClass: "tier-badge-1",
            complexity: 6,
            dvcStage: "calibrate_drift",
            responsibilities: "Calibrates baseline 1D Wasserstein-1 drift thresholds (tau_w1) via Monte Carlo split-subsampling on reference embeddings.",
            classes: "run_drift_calibration(...) -> DriftCalibrationConfig",
            methods: "run_drift_calibration(): Computes empirical null distribution quantile and writes config/drift_config.yaml.",
            github: "https://github.com/foersben/taxon-vision-mlops/blob/develop/src/taxon_vision/pipelines/calibrate_drift.py"
        },
        {
            subpkg: "taxon_vision.pipelines",
            file: "train.py",
            tier: 1,
            tierName: "Tier 1: Orchestration",
            badgeClass: "tier-badge-1",
            complexity: 9,
            dvcStage: "train",
            responsibilities: "Optimises cosine classification head on pre-extracted embeddings using Class-Balanced Loss.",
            classes: "run_training(config_path: Path | None = None) -> None",
            methods: "run_training(): Evaluates loss, executes early stopping, and exports models/checkpoints/head.pt.",
            github: "https://github.com/foersben/taxon-vision-mlops/blob/develop/src/taxon_vision/pipelines/train.py"
        },
        {
            subpkg: "taxon_vision.pipelines",
            file: "export_onnx.py",
            tier: 1,
            tierName: "Tier 1: Orchestration",
            badgeClass: "tier-badge-1",
            complexity: 7,
            dvcStage: "export",
            responsibilities: "Exports PyTorch backbone + head to dual-output ONNX graph with static INT8 QDQ quantization.",
            classes: "run_export(config_path: Path | None = None) -> None",
            methods: "run_export(): Verifies parity and writes models/checkpoints/model_int8.onnx and standalone head_int8.onnx.",
            github: "https://github.com/foersben/taxon-vision-mlops/blob/develop/src/taxon_vision/pipelines/export_onnx.py"
        },
        {
            subpkg: "taxon_vision.pipelines",
            file: "calibrate.py",
            tier: 1,
            tierName: "Tier 1: Orchestration",
            badgeClass: "tier-badge-1",
            complexity: 8,
            dvcStage: "calibrate",
            responsibilities: "Calibrates Split Conformal Prediction non-conformity threshold (q_hat) and Helmholtz Free Energy OOD threshold (tau_ood).",
            classes: "run_calibration(config_path: Path | None = None) -> None",
            methods: "run_calibration(): Calibrates marginal coverage guarantee (1 - alpha) and writes config/conformal_config.yaml.",
            github: "https://github.com/foersben/taxon-vision-mlops/blob/develop/src/taxon_vision/pipelines/calibrate.py"
        },
        {
            subpkg: "taxon_vision.pipelines",
            file: "evaluate.py",
            tier: 1,
            tierName: "Tier 1: Orchestration",
            badgeClass: "tier-badge-1",
            complexity: 12,
            dvcStage: "evaluate",
            responsibilities: "Runs holdout test split evaluation, verifies the four-gate model promotion criteria, and registers models.",
            classes: "run_evaluation(config_path: Path | None = None) -> None",
            methods: "run_evaluation(): Evaluates holdout split, promotes @Production alias in MLflow, and writes reports/pareto_evaluation.json.",
            github: "https://github.com/foersben/taxon-vision-mlops/blob/develop/src/taxon_vision/pipelines/evaluate.py"
        },
        {
            subpkg: "taxon_vision.service",
            file: "api.py",
            tier: 1,
            tierName: "Tier 1: Orchestration",
            badgeClass: "tier-badge-1",
            complexity: 4,
            dvcStage: "service",
            responsibilities: "FastAPI ASGI application root, lifespan pre-warming, exception handling, and APIRouter aggregation.",
            classes: "app: FastAPI
lifespan(app: FastAPI)",
            methods: "create_app() -> FastAPI
lifespan(app) -> AsyncIterator[None]",
            github: "https://github.com/foersben/taxon-vision-mlops/blob/develop/src/taxon_vision/service/api.py"
        },
        {
            subpkg: "taxon_vision.service",
            file: "inference.py",
            tier: 1,
            tierName: "Tier 1: Orchestration",
            badgeClass: "tier-badge-1",
            complexity: 9,
            dvcStage: "service",
            responsibilities: "Zero-allocation inference pipeline orchestrating image decoding, ONNX session, conformal sets, and CAM.",
            classes: "TaxonInferencePipeline
PredictionResult",
            methods: "predict_species(image_bytes: bytes, explain: bool) -> PredictionResult
get_inference_pipeline() -> TaxonInferencePipeline",
            github: "https://github.com/foersben/taxon-vision-mlops/blob/develop/src/taxon_vision/service/inference.py"
        },
        {
            subpkg: "taxon_vision.service",
            file: "storage.py",
            tier: 1,
            tierName: "Tier 1: Orchestration",
            badgeClass: "tier-badge-1",
            complexity: 5,
            dvcStage: "service",
            responsibilities: "Distributed Redis observation image cache (with bounded LRU in-memory fallback) and triage disk storage.",
            classes: "ImageCacheProtocol, RedisImageCache, MemoryImageCache",
            methods: "get_image_cache() -> ImageCacheProtocol
persist_flagged_observation(data: bytes, metadata: dict) -> Path",
            github: "https://github.com/foersben/taxon-vision-mlops/blob/develop/src/taxon_vision/service/storage.py"
        },
        {
            subpkg: "taxon_vision.service",
            file: "telemetry_store.py",
            tier: 1,
            tierName: "Tier 1: Orchestration",
            badgeClass: "tier-badge-1",
            complexity: 7,
            dvcStage: "service, monitoring",
            responsibilities: "Relational PostgreSQL store (with bounded in-memory fallback) persisting inference embeddings, conformal sets, and drift runs.",
            classes: "TelemetryStore, PostgresTelemetryStore, BoundedInMemoryTelemetryStore",
            methods: "record(record: PredictionRecord) -> None
recent_embeddings(limit: int) -> np.ndarray
latest_drift_run() -> DriftRunRecord | None",
            github: "https://github.com/foersben/taxon-vision-mlops/blob/develop/src/taxon_vision/service/telemetry_store.py"
        },
        {
            subpkg: "taxon_vision.service",
            file: "passport.py",
            tier: 1,
            tierName: "Tier 1: Orchestration",
            badgeClass: "tier-badge-1",
            complexity: 14,
            dvcStage: "service",
            responsibilities: "Aggregates MLflow @Production run metadata, conformal calibration parameters, and sliding drift for the HTMX UI.",
            classes: "ModelPassport
load_model_passport() -> ModelPassport",
            methods: "load_model_passport(force_refresh: bool = False) -> ModelPassport
clear_passport_cache() -> None",
            github: "https://github.com/foersben/taxon-vision-mlops/blob/develop/src/taxon_vision/service/passport.py"
        },
        {
            subpkg: "taxon_vision.service",
            file: "presentation.py",
            tier: 1,
            tierName: "Tier 1: Orchestration",
            badgeClass: "tier-badge-1",
            complexity: 1,
            dvcStage: "service",
            responsibilities: "Formats prediction outputs, probability distribution bars, and conformal set indicators into view-models for Jinja2/HTMX templates.",
            classes: "build_prediction_view_model, build_top_candidates",
            methods: "build_prediction_view_model(prediction: PredictionResult, image_hash: str) -> dict[str, Any]",
            github: "https://github.com/foersben/taxon-vision-mlops/blob/develop/src/taxon_vision/service/presentation.py"
        },
        {
            subpkg: "taxon_vision.service",
            file: "structured_logging.py",
            tier: 1,
            tierName: "Tier 1: Orchestration",
            badgeClass: "tier-badge-1",
            complexity: 5,
            dvcStage: "service, telemetry",
            responsibilities: "Configures Python logging with structured JSON formatting, correlation IDs, and millisecond timestamps for Promtail/Loki aggregation.",
            classes: "StructuredJsonFormatter
configure_structured_logging() -> None",
            methods: "format(record: logging.LogRecord) -> str
configure_structured_logging(level: str) -> None",
            github: "https://github.com/foersben/taxon-vision-mlops/blob/develop/src/taxon_vision/service/structured_logging.py"
        },
        {
            subpkg: "taxon_vision.service",
            file: "schemas.py",
            tier: 1,
            tierName: "Tier 1: Orchestration",
            badgeClass: "tier-badge-1",
            complexity: 0,
            dvcStage: "service",
            responsibilities: "Strict Pydantic request and response models enforcing parameter contracts across REST API routes and HTMX form uploads.",
            classes: "PredictionResponse, ExplainResponse, TrainingRequest, DriftEvaluationRequest",
            methods: "Pydantic BaseModel schemas with typed validation",
            github: "https://github.com/foersben/taxon-vision-mlops/blob/develop/src/taxon_vision/service/schemas.py"
        },
        {
            subpkg: "taxon_vision.service",
            file: "routes/views.py",
            tier: 1,
            tierName: "Tier 1: Orchestration",
            badgeClass: "tier-badge-1",
            complexity: 4,
            dvcStage: "service",
            responsibilities: "Server-rendered HTMX visual operator dashboard (GET /) rendering ModelPassport and handling observation uploads via partial HTML swaps (/ui/predict-htmx).",
            classes: "router: APIRouter",
            methods: "index(request: Request) -> HTMLResponse
predict_htmx(request: Request, file: UploadFile) -> HTMLResponse",
            github: "https://github.com/foersben/taxon-vision-mlops/blob/develop/src/taxon_vision/service/routes/views.py"
        },
        {
            subpkg: "taxon_vision.service",
            file: "routes/predict.py",
            tier: 1,
            tierName: "Tier 1: Orchestration",
            badgeClass: "tier-badge-1",
            complexity: 10,
            dvcStage: "service",
            responsibilities: "High-throughput POST /api/v1/predict REST endpoint executing zero-allocation inference, conformal coverage bounding, and background telemetry recording.",
            classes: "router: APIRouter",
            methods: "predict_species(file: UploadFile, explain: bool, background_tasks: BackgroundTasks) -> JSONResponse",
            github: "https://github.com/foersben/taxon-vision-mlops/blob/develop/src/taxon_vision/service/routes/predict.py"
        },
        {
            subpkg: "taxon_vision.service",
            file: "routes/explain.py",
            tier: 1,
            tierName: "Tier 1: Orchestration",
            badgeClass: "tier-badge-1",
            complexity: 5,
            dvcStage: "service",
            responsibilities: "On-demand POST /api/v1/explain endpoint retrieving cached observation images from Redis and generating forward CAM heatmaps across any candidate taxon.",
            classes: "router: APIRouter",
            methods: "explain_prediction(image_hash: str, target_class: str, file: UploadFile) -> ExplainResponse | HTMLResponse",
            github: "https://github.com/foersben/taxon-vision-mlops/blob/develop/src/taxon_vision/service/routes/explain.py"
        },
        {
            subpkg: "taxon_vision.service",
            file: "routes/train.py",
            tier: 1,
            tierName: "Tier 1: Orchestration",
            badgeClass: "tier-badge-1",
            complexity: 6,
            dvcStage: "service, train",
            responsibilities: "POST /api/v1/train endpoint triggering asynchronous classification head retraining in background worker threads without blocking the event loop.",
            classes: "router: APIRouter",
            methods: "trigger_training(request: TrainingRequest, background_tasks: BackgroundTasks) -> JSONResponse",
            github: "https://github.com/foersben/taxon-vision-mlops/blob/develop/src/taxon_vision/service/routes/train.py"
        },
        {
            subpkg: "taxon_vision.service",
            file: "routes/retrain.py",
            tier: 1,
            tierName: "Tier 1: Orchestration",
            badgeClass: "tier-badge-1",
            complexity: 6,
            dvcStage: "service, operations",
            responsibilities: "Secret-authenticated POST /api/v1/retrain/webhook endpoint receiving Grafana Alertmanager alerts and dispatching the Airflow model_retraining DAG.",
            classes: "router: APIRouter",
            methods: "retrain_webhook_endpoint(request: Request, authorization: str) -> JSONResponse",
            github: "https://github.com/foersben/taxon-vision-mlops/blob/develop/src/taxon_vision/service/routes/retrain.py"
        },
        {
            subpkg: "taxon_vision.service",
            file: "routes/health.py",
            tier: 1,
            tierName: "Tier 1: Orchestration",
            badgeClass: "tier-badge-1",
            complexity: 4,
            dvcStage: "service, monitoring",
            responsibilities: "Endpoints for Kubernetes liveness (/health), model readiness (/health/ready), and OpenMetrics Prometheus scraping (/metrics).",
            classes: "router: APIRouter",
            methods: "health() -> dict
health_ready() -> dict
metrics() -> Response",
            github: "https://github.com/foersben/taxon-vision-mlops/blob/develop/src/taxon_vision/service/routes/health.py"
        },
        {
            subpkg: "taxon_vision.service",
            file: "routes/drift.py",
            tier: 1,
            tierName: "Tier 1: Orchestration",
            badgeClass: "tier-badge-1",
            complexity: 3,
            dvcStage: "service, monitoring",
            responsibilities: "Sliding-window representation drift evaluation endpoint (POST /api/v1/drift/evaluate) and latest HTML report retrieval (GET /api/v1/drift/latest).",
            classes: "router: APIRouter",
            methods: "evaluate_drift_endpoint(window_size: int) -> JSONResponse
get_latest_drift_endpoint() -> HTMLResponse",
            github: "https://github.com/foersben/taxon-vision-mlops/blob/develop/src/taxon_vision/service/routes/drift.py"
        },
        {
            subpkg: "taxon_vision.models",
            file: "factory.py",
            tier: 2,
            tierName: "Tier 2: ML Engine",
            badgeClass: "tier-badge-2",
            complexity: 4,
            dvcStage: "extract_features",
            responsibilities: "Vision backbone factory instantiating and caching DINOv3, BioCLIP-2, MobileNetV4, and DINOv2 with Hugging Face integration.",
            classes: "create_backbone(model_name: str, pretrained: bool = True) -> tuple[nn.Module, int]",
            methods: "create_backbone(): Loads model trunk and returns frozen nn.Module and feature dimension.",
            github: "https://github.com/foersben/taxon-vision-mlops/blob/develop/src/taxon_vision/models/factory.py"
        },
        {
            subpkg: "taxon_vision.models",
            file: "head.py",
            tier: 2,
            tierName: "Tier 2: ML Engine",
            badgeClass: "tier-badge-2",
            complexity: 1,
            dvcStage: "train, export",
            responsibilities: "Lightweight linear probe classification head mapping pre-extracted representations to taxonomic class logits.",
            classes: "TaxonClassifier(nn.Module)",
            methods: "forward(x: torch.Tensor) -> torch.Tensor: Linear projection with dropout regularization.",
            github: "https://github.com/foersben/taxon-vision-mlops/blob/develop/src/taxon_vision/models/head.py"
        },
        {
            subpkg: "taxon_vision.models",
            file: "loss.py",
            tier: 2,
            tierName: "Tier 2: ML Engine",
            badgeClass: "tier-badge-2",
            complexity: 2,
            dvcStage: "train",
            responsibilities: "Class-Balanced Loss using effective sample numbers E_n = (1 - beta^n) / (1 - beta) to handle extreme long-tail class imbalance.",
            classes: "ClassBalancedLoss(nn.Module)",
            methods: "forward(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor",
            github: "https://github.com/foersben/taxon-vision-mlops/blob/develop/src/taxon_vision/models/loss.py"
        },
        {
            subpkg: "taxon_vision.models",
            file: "training/runner.py",
            tier: 2,
            tierName: "Tier 2: ML Engine",
            badgeClass: "tier-badge-2",
            complexity: 8,
            dvcStage: "train",
            responsibilities: "Top-level training orchestrator coordinating data hydration, Optuna sweeps, training loops, and model promotion.",
            classes: "run_training_pipeline(...) -> dict[str, Any]",
            methods: "run_training_pipeline(): Dispatches training execution across decomposed sibling modules.",
            github: "https://github.com/foersben/taxon-vision-mlops/blob/develop/src/taxon_vision/models/training/runner.py"
        },
        {
            subpkg: "taxon_vision.models",
            file: "training/dataset.py",
            tier: 2,
            tierName: "Tier 2: ML Engine",
            badgeClass: "tier-badge-2",
            complexity: 4,
            dvcStage: "train",
            responsibilities: "Loads stratified embedding tensors and extracts DVC lock hashes without git dependencies.",
            classes: "load_or_create_training_split(...) -> EmbeddingSplit
get_dvc_dataset_hash() -> str",
            methods: "load_or_create_training_split(): Returns typed frozen EmbeddingSplit dataclass.",
            github: "https://github.com/foersben/taxon-vision-mlops/blob/develop/src/taxon_vision/models/training/dataset.py"
        },
        {
            subpkg: "taxon_vision.models",
            file: "training/loops.py",
            tier: 2,
            tierName: "Tier 2: ML Engine",
            badgeClass: "tier-badge-2",
            complexity: 7,
            dvcStage: "train",
            responsibilities: "Isolated single-epoch mini-batch SGD passes returning typed HeadEvaluationResult.",
            classes: "train_head_epoch(...) -> float
evaluate_head(...) -> HeadEvaluationResult",
            methods: "train_head_epoch(): Computes gradients with Class-Balanced Loss. evaluate_head(): Calculates PR-AUC and accuracy.",
            github: "https://github.com/foersben/taxon-vision-mlops/blob/develop/src/taxon_vision/models/training/loops.py"
        },
        {
            subpkg: "taxon_vision.models",
            file: "training/embeddings.py",
            tier: 2,
            tierName: "Tier 2: ML Engine",
            badgeClass: "tier-badge-2",
            complexity: 6,
            dvcStage: "train",
            responsibilities: "Multi-epoch SGD training loop over pre-extracted in-memory embedding tensors, coordinating early stopping and learning rate scheduling.",
            classes: "train_head_on_cached_embeddings(...) -> tuple[TaxonClassifier, TrainingMetrics]",
            methods: "train_head_on_cached_embeddings(): Drives full multi-epoch classification head training on cached tensors.",
            github: "https://github.com/foersben/taxon-vision-mlops/blob/develop/src/taxon_vision/models/training/embeddings.py"
        },
        {
            subpkg: "taxon_vision.models",
            file: "training/checkpoint.py",
            tier: 2,
            tierName: "Tier 2: ML Engine",
            badgeClass: "tier-badge-2",
            complexity: 2,
            dvcStage: "train",
            responsibilities: "Deterministic state dictionary serialization and atomic file swapping for PyTorch classification heads.",
            classes: "save_checkpoint(model: nn.Module, path: Path) -> Path
load_checkpoint(model: nn.Module, path: Path) -> nn.Module",
            methods: "save_checkpoint(): Atomically stages and writes head weights to disk.",
            github: "https://github.com/foersben/taxon-vision-mlops/blob/develop/src/taxon_vision/models/training/checkpoint.py"
        },
        {
            subpkg: "taxon_vision.models",
            file: "training/promotion.py",
            tier: 2,
            tierName: "Tier 2: ML Engine",
            badgeClass: "tier-badge-2",
            complexity: 4,
            dvcStage: "train, eval",
            responsibilities: "Evaluates Macro PR-AUC against incumbent models and tags MLflow aliases (@Production / @Challenger).",
            classes: "_promote_model_if_better(...) -> None
_determine_promotion_alias(...) -> str",
            methods: "_promote_model_if_better(): Conditionally assigns @Production alias upon surpassing holdout benchmarks.",
            github: "https://github.com/foersben/taxon-vision-mlops/blob/develop/src/taxon_vision/models/training/promotion.py"
        },
        {
            subpkg: "taxon_vision.models",
            file: "training/tracking.py",
            tier: 2,
            tierName: "Tier 2: ML Engine",
            badgeClass: "tier-badge-2",
            complexity: 5,
            dvcStage: "train, eval",
            responsibilities: "RAII context manager guaranteeing deterministic MLflow session lifecycle, fail-fast logging, and artifact persistence.",
            classes: "mlflow_run_scope(run_name: str, params: dict) -> Generator
MLflowTracker",
            methods: "mlflow_run_scope(): Guarantees proper teardown on unhandled exceptions. log_metrics(metrics: dict) -> None",
            github: "https://github.com/foersben/taxon-vision-mlops/blob/develop/src/taxon_vision/models/training/tracking.py"
        },
        {
            subpkg: "taxon_vision.models",
            file: "training/tuning.py",
            tier: 2,
            tierName: "Tier 2: ML Engine",
            badgeClass: "tier-badge-2",
            complexity: 8,
            dvcStage: "train",
            responsibilities: "Bayesian hyperparameter optimization study orchestration with TPE sampling and ASHA pruning over cached embeddings.",
            classes: "tune_hyperparameters(data: EmbeddingSplit, ...) -> TuningResult
HyperparameterObjective",
            methods: "tune_hyperparameters(): Executes Optuna study and returns best hyperparameters and trials.",
            github: "https://github.com/foersben/taxon-vision-mlops/blob/develop/src/taxon_vision/models/training/tuning.py"
        },
        {
            subpkg: "taxon_vision.models",
            file: "training/visualizations.py",
            tier: 2,
            tierName: "Tier 2: ML Engine",
            badgeClass: "tier-badge-2",
            complexity: 3,
            dvcStage: "train, eval",
            responsibilities: "Renders diagnostic training plots (confusion matrices, precision-recall curves, and Helmholtz energy distributions) for MLflow logging.",
            classes: "log_training_visualizations(tracker, history, val_labels, val_probs) -> None",
            methods: "log_training_visualizations(): Renders and saves PNG figures to the active MLflow artifact store.",
            github: "https://github.com/foersben/taxon-vision-mlops/blob/develop/src/taxon_vision/models/training/visualizations.py"
        },
        {
            subpkg: "taxon_vision.models",
            file: "training/types.py",
            tier: 2,
            tierName: "Tier 2: ML Engine",
            badgeClass: "tier-badge-2",
            complexity: 0,
            dvcStage: "train",
            responsibilities: "Parameter containers enforcing parameter encapsulation and zero tuple blindness (typed frozen dataclasses and models).",
            classes: "ResolvedTrainingParameters, EmbeddingSplit, HeadEvaluationResult, TuningResult, TrainingMetrics",
            methods: "Immutable dataclasses with typed fields",
            github: "https://github.com/foersben/taxon-vision-mlops/blob/develop/src/taxon_vision/models/training/types.py"
        },
        {
            subpkg: "taxon_vision.inference",
            file: "onnx_engine.py",
            tier: 2,
            tierName: "Tier 2: ML Engine",
            badgeClass: "tier-badge-2",
            complexity: 6,
            dvcStage: "service, eval",
            responsibilities: "Zero-allocation ONNX Runtime inference session with locked thread affinity and pre-allocated contiguous C-buffers.",
            classes: "ONNXInferenceEngine
EngineOutputs",
            methods: "predict(tensor: np.ndarray) -> np.ndarray
predict_outputs(tensor: np.ndarray) -> EngineOutputs",
            github: "https://github.com/foersben/taxon-vision-mlops/blob/develop/src/taxon_vision/inference/onnx_engine.py"
        },
        {
            subpkg: "taxon_vision.inference",
            file: "onnx_exporter.py",
            tier: 2,
            tierName: "Tier 2: ML Engine",
            badgeClass: "tier-badge-2",
            complexity: 10,
            dvcStage: "export",
            responsibilities: "Exports dual-output PyTorch models (logits + embeddings) to ONNX Opset 17 with dynamic batch axes, verifying numerical parity against PyTorch FP32.",
            classes: "export_to_onnx, export_head_to_onnx",
            methods: "export_to_onnx(model: nn.Module, dummy_input: torch.Tensor, path: Path) -> Path",
            github: "https://github.com/foersben/taxon-vision-mlops/blob/develop/src/taxon_vision/inference/onnx_exporter.py"
        },
        {
            subpkg: "taxon_vision.inference",
            file: "head_projector.py",
            tier: 2,
            tierName: "Tier 2: ML Engine",
            badgeClass: "tier-badge-2",
            complexity: 9,
            dvcStage: "export, service",
            responsibilities: "Extracts ONNX linear probe classification head weights and biases into standalone projector session for sub-0.2ms latency scoring.",
            classes: "extract_onnx_head_weights, compute_onnx_logits",
            methods: "extract_onnx_head_weights(onnx_path: Path) -> tuple[np.ndarray, np.ndarray]",
            github: "https://github.com/foersben/taxon-vision-mlops/blob/develop/src/taxon_vision/inference/head_projector.py"
        },
        {
            subpkg: "taxon_vision.inference",
            file: "quantizer.py",
            tier: 2,
            tierName: "Tier 2: ML Engine",
            badgeClass: "tier-badge-2",
            complexity: 7,
            dvcStage: "export",
            responsibilities: "Static and dynamic INT8 post-training quantization (PTQ) protecting final logit nodes via find_logit_nodes().",
            classes: "quantize_onnx_model(...) -> Path
quantize_onnx_static(...) -> Path",
            methods: "quantize_onnx_model(): Dynamic PTQ. quantize_onnx_static(): Calibration-driven static PTQ.",
            github: "https://github.com/foersben/taxon-vision-mlops/blob/develop/src/taxon_vision/inference/quantizer.py"
        },
        {
            subpkg: "taxon_vision.inference",
            file: "calibration.py",
            tier: 2,
            tierName: "Tier 2: ML Engine",
            badgeClass: "tier-badge-2",
            complexity: 10,
            dvcStage: "export, calibrate",
            responsibilities: "Streams representative calibration batches for ONNX Runtime static Post-Training Quantization (PTQ) and non-conformity threshold calibration.",
            classes: "TaxonCalibrationDataReader, load_calibration_tensors",
            methods: "get_next() -> dict[str, np.ndarray] | None",
            github: "https://github.com/foersben/taxon-vision-mlops/blob/develop/src/taxon_vision/inference/calibration.py"
        },
        {
            subpkg: "taxon_vision.inference",
            file: "cam.py",
            tier: 2,
            tierName: "Tier 2: ML Engine",
            badgeClass: "tier-badge-2",
            complexity: 11,
            dvcStage: "service",
            responsibilities: "Forward-only Class Activation Mapping projecting ViT patch tokens and CNN activations without backward-pass autograd.",
            classes: "generate_heatmap(...) -> np.ndarray
generate_cam_overlay(...) -> np.ndarray",
            methods: "generate_heatmap(): Extracts spatial activations and computes class-specific weighted projections.",
            github: "https://github.com/foersben/taxon-vision-mlops/blob/develop/src/taxon_vision/inference/cam.py"
        },
        {
            subpkg: "taxon_vision.inference",
            file: "mlflow_model.py",
            tier: 2,
            tierName: "Tier 2: ML Engine",
            badgeClass: "tier-badge-2",
            complexity: 12,
            dvcStage: "export, eval",
            responsibilities: "Custom mlflow.pyfunc.PythonModel implementation wrapping model_int8.onnx and conformal calibration config for MLflow Model Registry serving.",
            classes: "TaxonVisionPyFuncModel",
            methods: "load_context(context), predict(context, model_input) -> pd.DataFrame",
            github: "https://github.com/foersben/taxon-vision-mlops/blob/develop/src/taxon_vision/inference/mlflow_model.py"
        },
        {
            subpkg: "taxon_vision.inference",
            file: "cli.py",
            tier: 2,
            tierName: "Tier 2: ML Engine",
            badgeClass: "tier-badge-2",
            complexity: 3,
            dvcStage: "inference",
            responsibilities: "Command-line interface entrypoint for standalone image classification, latency microbenchmarking, and conformal prediction inspection.",
            classes: "main() -> None",
            methods: "main(): Parses image path and flags, invokes ONNXInferenceEngine, and prints formatted rich terminal reports.",
            github: "https://github.com/foersben/taxon-vision-mlops/blob/develop/src/taxon_vision/inference/cli.py"
        },
        {
            subpkg: "taxon_vision.uncertainty",
            file: "conformal.py",
            tier: 2,
            tierName: "Tier 2: ML Engine",
            badgeClass: "tier-badge-2",
            complexity: 4,
            dvcStage: "calibrate, service",
            responsibilities: "Split Conformal Prediction engine guaranteeing marginal coverage 1 - alpha with finite-sample quantile correction.",
            classes: "SplitConformalPredictor
ConformalPredictionEngine",
            methods: "compute_quantile(scores: np.ndarray, alpha: float) -> float
predict_set(probs: np.ndarray) -> list[int]",
            github: "https://github.com/foersben/taxon-vision-mlops/blob/develop/src/taxon_vision/uncertainty/conformal.py"
        },
        {
            subpkg: "taxon_vision.uncertainty",
            file: "ood_detector.py",
            tier: 2,
            tierName: "Tier 2: ML Engine",
            badgeClass: "tier-badge-2",
            complexity: 3,
            dvcStage: "calibrate, service",
            responsibilities: "Thermodynamic Helmholtz Free Energy out-of-distribution detector and normalized Shannon entropy filter.",
            classes: "EnergyOODDetector",
            methods: "compute_energy(logits: np.ndarray, temp: float = 1.0) -> np.ndarray
is_ood(logits: np.ndarray) -> bool",
            github: "https://github.com/foersben/taxon-vision-mlops/blob/develop/src/taxon_vision/uncertainty/ood_detector.py"
        },
        {
            subpkg: "taxon_vision.active_learning",
            file: "query.py",
            tier: 2,
            tierName: "Tier 2: ML Engine",
            badgeClass: "tier-badge-2",
            complexity: 0,
            dvcStage: "active_learning",
            responsibilities: "Active learning query acquisition algorithms including margin sampling and BADGE gradient space embeddings.",
            classes: "margin_sampling(probs: np.ndarray) -> np.ndarray",
            methods: "margin_sampling(): Returns uncertainty margin score 1.0 - (p_top1 - p_top2).",
            github: "https://github.com/foersben/taxon-vision-mlops/blob/develop/src/taxon_vision/active_learning/query.py"
        },
        {
            subpkg: "taxon_vision.active_learning",
            file: "triage_queue.py",
            tier: 2,
            tierName: "Tier 2: ML Engine",
            badgeClass: "tier-badge-2",
            complexity: 1,
            dvcStage: "service, triage",
            responsibilities: "In-memory binary max-heap prioritizing uncertain, ambiguous, or out-of-distribution observations for human expert taxonomist review.",
            classes: "TriageQueue, TriageItem",
            methods: "push(item: TriageItem) -> None
pop() -> TriageItem
queue() -> list[TriageItem]",
            github: "https://github.com/foersben/taxon-vision-mlops/blob/develop/src/taxon_vision/active_learning/triage_queue.py"
        },
        {
            subpkg: "taxon_vision.active_learning",
            file: "feedback.py",
            tier: 2,
            tierName: "Tier 2: ML Engine",
            badgeClass: "tier-badge-2",
            complexity: 12,
            dvcStage: "active_learning",
            responsibilities: "Ingests verified human-in-the-loop annotations and Label Studio exports into dataset manifests with attribution retention.",
            classes: "load_taxa_mappings, merge_feedback_to_manifest",
            methods: "merge_feedback_to_manifest(raw_manifest_path, export_path, taxa_catalog_path) -> int",
            github: "https://github.com/foersben/taxon-vision-mlops/blob/develop/src/taxon_vision/active_learning/feedback.py"
        },
        {
            subpkg: "taxon_vision.monitoring",
            file: "drift.py",
            tier: 2,
            tierName: "Tier 2: ML Engine",
            badgeClass: "tier-badge-2",
            complexity: 11,
            dvcStage: "data_drift, monitoring",
            responsibilities: "Two-tier data drift detection: 1D Wasserstein-1 Earth Mover's Distance on L2 embedding norms + multivariate RKHS MMD^2 permutation tests.",
            classes: "compute_wasserstein_drift(...) -> float
run_mmd_permutation_test(...) -> MMDPermutationResult",
            methods: "compute_wasserstein_drift(): Calculates W1 distance. run_mmd_permutation_test(): Calculates MMD^2 p-value.",
            github: "https://github.com/foersben/taxon-vision-mlops/blob/develop/src/taxon_vision/monitoring/drift.py"
        },
        {
            subpkg: "taxon_vision.monitoring",
            file: "drift_runner.py",
            tier: 2,
            tierName: "Tier 2: ML Engine",
            badgeClass: "tier-badge-2",
            complexity: 5,
            dvcStage: "monitoring",
            responsibilities: "Sliding-window drift runner fetching recent embeddings from TelemetryStore, generating PCA reports, and updating Prometheus gauges.",
            classes: "evaluate_sliding_window_drift(...) -> DriftEvaluationOutcome",
            methods: "evaluate_sliding_window_drift(): Evaluates window, logs Evidently PCA reports, and updates Prometheus drift metrics.",
            github: "https://github.com/foersben/taxon-vision-mlops/blob/develop/src/taxon_vision/monitoring/drift_runner.py"
        },
        {
            subpkg: "taxon_vision.monitoring",
            file: "telemetry.py",
            tier: 2,
            tierName: "Tier 2: ML Engine",
            badgeClass: "tier-badge-2",
            complexity: 1,
            dvcStage: "service, monitoring",
            responsibilities: "OpenMetrics Prometheus collector exposing inference throughput counters, latency histograms, and representation drift gauges.",
            classes: "PrometheusTelemetry
DriftMetricsCollector",
            methods: "record_inference(...) -> None
update_drift_metrics(...) -> None",
            github: "https://github.com/foersben/taxon-vision-mlops/blob/develop/src/taxon_vision/monitoring/telemetry.py"
        },
        {
            subpkg: "taxon_vision.data",
            file: "inat_client.py",
            tier: 3,
            tierName: "Tier 3: Domain & Data",
            badgeClass: "tier-badge-3",
            complexity: 4,
            dvcStage: "ingest",
            responsibilities: "Queries iNaturalist v1 REST API for research-grade biological observation records.",
            classes: "INatAPIClient",
            methods: "fetch_observations(taxon_id: int, per_page: int) -> list[dict]",
            github: "https://github.com/foersben/taxon-vision-mlops/blob/develop/src/taxon_vision/data/inat_client.py"
        },
        {
            subpkg: "taxon_vision.data",
            file: "license_filter.py",
            tier: 3,
            tierName: "Tier 3: Domain & Data",
            badgeClass: "tier-badge-3",
            complexity: 3,
            dvcStage: "audit_license",
            responsibilities: "Enforces strict open-license whitelist (CC0, CC-BY, CC-BY-NC) and validates photographer attribution metadata.",
            classes: "LicenseFilter",
            methods: "filter_compliant_records(records: list[dict]) -> tuple[list[dict], dict]",
            github: "https://github.com/foersben/taxon-vision-mlops/blob/develop/src/taxon_vision/data/license_filter.py"
        },
        {
            subpkg: "taxon_vision.data",
            file: "s3_streamer.py",
            tier: 3,
            tierName: "Tier 3: Domain & Data",
            badgeClass: "tier-badge-3",
            complexity: 13,
            dvcStage: "extract_features",
            responsibilities: "Zero-disk image streaming directly from public AWS S3 bucket (s3://inaturalist-open-data/) with keep-alive connection pooling.",
            classes: "S3ImageStreamer",
            methods: "stream_images(manifest: pd.DataFrame) -> Generator[tuple[int, Image.Image], None, None]",
            github: "https://github.com/foersben/taxon-vision-mlops/blob/develop/src/taxon_vision/data/s3_streamer.py"
        },
        {
            subpkg: "taxon_vision.domain",
            file: "taxonomy.py",
            tier: 3,
            tierName: "Tier 3: Domain & Data",
            badgeClass: "tier-badge-3",
            complexity: 3,
            dvcStage: "ingest, service",
            responsibilities: "DarwinCore taxonomic hierarchy models, single source of truth class index mapping, and scientific nomenclature.",
            classes: "TaxonomyCatalog
TaxonNode",
            methods: "load_taxa_catalog(path: Path) -> TaxonomyCatalog
get_by_id(taxon_id: int) -> TaxonNode",
            github: "https://github.com/foersben/taxon-vision-mlops/blob/develop/src/taxon_vision/domain/taxonomy.py"
        },
        {
            subpkg: "taxon_vision.domain",
            file: "license.py",
            tier: 3,
            tierName: "Tier 3: Domain & Data",
            badgeClass: "tier-badge-3",
            complexity: 1,
            dvcStage: "audit_license",
            responsibilities: "Creative Commons license enumerations, compliance predicates, and legal license classification rules.",
            classes: "LicenseCode(StrEnum)
is_compliant_license(code: str) -> bool",
            methods: "is_compliant_license(code: str) -> bool",
            github: "https://github.com/foersben/taxon-vision-mlops/blob/develop/src/taxon_vision/domain/license.py"
        },
        {
            subpkg: "taxon_vision.domain",
            file: "observation.py",
            tier: 3,
            tierName: "Tier 3: Domain & Data",
            badgeClass: "tier-badge-3",
            complexity: 2,
            dvcStage: "ingest, service",
            responsibilities: "Immutable DarwinCore observation dataclass encapsulating specimen IDs, taxonomic hierarchy, photographic attribution, and licensing codes.",
            classes: "Observation, ObservationPassport",
            methods: "from_dict(data: dict) -> Observation
to_dict() -> dict",
            github: "https://github.com/foersben/taxon-vision-mlops/blob/develop/src/taxon_vision/domain/observation.py"
        },
        {
            subpkg: "taxon_vision.config",
            file: "schemas.py",
            tier: 4,
            tierName: "Tier 4: Configuration",
            badgeClass: "tier-badge-4",
            complexity: 0,
            dvcStage: "all",
            responsibilities: "Typed Pydantic Settings container validating configuration hierarchies and typed sub-models across all stages.",
            classes: "Settings, ModelConfig, IngestionConfig, ConformalConfig, DriftConfig, TelemetryConfig",
            methods: "Pydantic Settings and BaseModel configurations",
            github: "https://github.com/foersben/taxon-vision-mlops/blob/develop/src/taxon_vision/config/schemas.py"
        },
        {
            subpkg: "taxon_vision.config",
            file: "loader.py",
            tier: 4,
            tierName: "Tier 4: Configuration",
            badgeClass: "tier-badge-4",
            complexity: 10,
            dvcStage: "all",
            responsibilities: "3-Tier Configuration Precedence Pyramid loader, deep merge, and LRU cache.",
            classes: "get_settings(config_path: Path | None = None) -> Settings
reset_settings() -> None",
            methods: "_deep_merge(base: dict, override: dict) -> dict
_apply_direct_env_overrides(data: dict) -> dict",
            github: "https://github.com/foersben/taxon-vision-mlops/blob/develop/src/taxon_vision/config/loader.py"
        }
    ];

    let currentTier = "all";
    let searchQuery = "";
    let activeModule = modules[0];

    // Render skeleton
    mountPoint.innerHTML = `
        <div class="codebase-explorer">
            <div class="explorer-toolbar">
                <input type="text" class="explorer-search-input" id="explorer-search" placeholder="🔍 Search modules, classes, or methods..." />
                <div class="explorer-tier-filters">
                    <button class="explorer-tier-btn active" data-tier="all">All Tiers</button>
                    <button class="explorer-tier-btn" data-tier="1">Tier 1: Orchestration</button>
                    <button class="explorer-tier-btn" data-tier="2">Tier 2: ML Engines</button>
                    <button class="explorer-tier-btn" data-tier="3">Tier 3: Domain & Data</button>
                    <button class="explorer-tier-btn" data-tier="4">Tier 4: Config</button>
                </div>
            </div>
            <div class="explorer-layout">
                <div class="explorer-module-list" id="explorer-list"></div>
                <div class="explorer-detail-panel" id="explorer-detail"></div>
            </div>
        </div>
    `;

    const searchInput = document.getElementById("explorer-search");
    const listContainer = document.getElementById("explorer-list");
    const detailContainer = document.getElementById("explorer-detail");
    const filterButtons = mountPoint.querySelectorAll(".explorer-tier-btn");

    function renderList() {
        const filtered = modules.filter(function (m) {
            const matchesTier = currentTier === "all" || m.tier.toString() === currentTier;
            const q = searchQuery.toLowerCase();
            const matchesSearch = !q ||
                m.file.toLowerCase().includes(q) ||
                m.subpkg.toLowerCase().includes(q) ||
                m.classes.toLowerCase().includes(q) ||
                m.responsibilities.toLowerCase().includes(q);
            return matchesTier && matchesSearch;
        });

        if (filtered.length === 0) {
            listContainer.innerHTML = '<div style="padding: 1.5rem; text-align: center; opacity: 0.7;">No matching modules found.</div>';
            detailContainer.innerHTML = '<div style="padding: 1.5rem; text-align: center; opacity: 0.7;">Select a module to view architectural details.</div>';
            return;
        }

        if (!filtered.includes(activeModule)) {
            activeModule = filtered[0];
        }

        listContainer.innerHTML = filtered.map(function (m) {
            const isActive = m === activeModule ? "active" : "";
            return `
                <div class="explorer-module-item ${isActive}" data-file="${m.file}">
                    <div class="mod-name">
                        <span>${m.file}</span>
                        <span class="explorer-tier-badge ${m.badgeClass}">Tier ${m.tier}</span>
                    </div>
                    <div class="mod-subpkg">${m.subpkg}</div>
                </div>
            `;
        }).join("");

        const items = listContainer.querySelectorAll(".explorer-module-item");
        items.forEach(function (el) {
            el.addEventListener("click", function () {
                const targetFile = el.getAttribute("data-file");
                activeModule = modules.find(function (m) { return m.file === targetFile; }) || activeModule;
                renderList();
                renderDetail();
            });
        });

        renderDetail();
    }

    function renderDetail() {
        if (!activeModule) return;
        const m = activeModule;
        detailContainer.innerHTML = `
            <div class="explorer-detail-header">
                <div class="explorer-detail-title">${m.subpkg}.${m.file.replace(".py", "")}</div>
                <div class="explorer-meta-row">
                    <span class="explorer-meta-tag"><strong>Layer:</strong> ${m.tierName}</span>
                    <span class="explorer-meta-tag"><strong>Cognitive Complexity:</strong> Score ${m.complexity} (≤ 15 Bound)</span>
                    <span class="explorer-meta-tag"><strong>DVC Consumer:</strong> ${m.dvcStage}</span>
                </div>
            </div>
            <div>
                <div class="explorer-section-title">Architectural Responsibilities</div>
                <p style="margin: 0.25rem 0 1rem 0; line-height: 1.5;">${m.responsibilities}</p>

                <div class="explorer-section-title">Key Classes & Functions</div>
                <div class="explorer-code-block">${m.classes}</div>

                <div class="explorer-section-title">Primary Method Signatures</div>
                <div class="explorer-code-block">${m.methods}</div>

                <a href="${m.github}" target="_blank" rel="noopener noreferrer" class="explorer-github-btn">
                    <span>↗ View Source on GitHub</span>
                </a>
            </div>
        `;
    }

    filterButtons.forEach(function (btn) {
        btn.addEventListener("click", function () {
            filterButtons.forEach(function (b) { b.classList.remove("active"); });
            btn.classList.add("active");
            currentTier = btn.getAttribute("data-tier") || "all";
            renderList();
        });
    });

    searchInput.addEventListener("input", function (e) {
        searchQuery = e.target.value.trim();
        renderList();
    });

    renderList();
}

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", initCodebaseExplorer);
    } else {
        initCodebaseExplorer();
    }

    if (typeof document$ !== "undefined" && typeof document$.subscribe === "function") {
        document$.subscribe(initCodebaseExplorer);
    }
})();
