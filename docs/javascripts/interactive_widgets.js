/**
 * TaxonVision-MLOps Interactive Widgets & Visualizer Controller
 */

document.addEventListener("DOMContentLoaded", function () {
    initFullscreenHandlers();
    initConformalSimulators();
    initCodebaseExplorer();
    initDVCExplorer();
});

// Support dynamic navigation in Material for MkDocs / Zensical
if (typeof document$ !== "undefined") {
    document$.subscribe(function () {
        initFullscreenHandlers();
        initConformalSimulators();
        initCodebaseExplorer();
        initDVCExplorer();
    });
}

/**
 * Initializes fullscreen toggle behavior for visualizer wrappers
 */
function initFullscreenHandlers() {
    const fullscreenButtons = document.querySelectorAll(".btn-fullscreen");
    fullscreenButtons.forEach(function (btn) {
        if (btn.dataset.initialized) return;
        btn.dataset.initialized = "true";

        btn.addEventListener("click", function () {
            const targetId = btn.getAttribute("data-target");
            const wrapper = targetId ? document.getElementById(targetId) : btn.closest(".visualizer-wrapper");
            if (!wrapper) return;

            wrapper.classList.toggle("is-fullscreen");
            const isFull = wrapper.classList.contains("is-fullscreen");
            btn.innerHTML = isFull ? "✕ Exit Fullscreen" : "⛶ Fullscreen";

            // If entering fullscreen, notify iframe
            const iframe = wrapper.querySelector("iframe");
            if (iframe && iframe.contentWindow) {
                iframe.contentWindow.dispatchEvent(new Event("resize"));
            }
        });
    });

    // Escape key listener to exit fullscreen
    document.addEventListener("keydown", function (e) {
        if (e.key === "Escape") {
            const activeFullscreen = document.querySelectorAll(".visualizer-wrapper.is-fullscreen");
            activeFullscreen.forEach(function (wrapper) {
                wrapper.classList.remove("is-fullscreen");
                const btn = wrapper.querySelector(".btn-fullscreen");
                if (btn) btn.innerHTML = "⛶ Fullscreen";
            });
        }
    });
}

/**
 * Interactive Conformal Coverage & Helmholtz Energy Simulator
 */
function initConformalSimulators() {
    const alphaSlider = document.getElementById("sim-alpha-slider");
    if (!alphaSlider) return;

    const energySlider = document.getElementById("sim-energy-slider");
    const tempSlider = document.getElementById("sim-temp-slider");

    function updateSimulation() {
        const alpha = parseFloat(alphaSlider.value) || 0.05;
        const energyThresh = energySlider ? parseFloat(energySlider.value) : -2.28;
        const temp = tempSlider ? parseFloat(tempSlider.value) : 1.0;

        const coverageTarget = (1.0 - alpha) * 100;
        const qHat = (1.0 - alpha * 0.92).toFixed(4);
        const avgSetSize = (1.0 + Math.pow(alpha, -0.42) * 0.45).toFixed(2);
        const triageRate = (alpha * 100 * 0.85 + (energyThresh < -12.0 ? 5.2 : 1.8)).toFixed(1);

        // Update DOM displays
        const alphaValDisplay = document.getElementById("sim-alpha-val");
        if (alphaValDisplay) alphaValDisplay.textContent = alpha.toFixed(2);

        const energyValDisplay = document.getElementById("sim-energy-val");
        if (energyValDisplay) energyValDisplay.textContent = energyThresh.toFixed(1);

        const tempValDisplay = document.getElementById("sim-temp-val");
        if (tempValDisplay) tempValDisplay.textContent = temp.toFixed(1);

        const covDisplay = document.getElementById("sim-out-coverage");
        if (covDisplay) covDisplay.textContent = coverageTarget.toFixed(1) + "%";

        const qDisplay = document.getElementById("sim-out-qhat");
        if (qDisplay) qDisplay.textContent = qHat;

        const sizeDisplay = document.getElementById("sim-out-setsize");
        if (sizeDisplay) sizeDisplay.textContent = avgSetSize;

        const triageDisplay = document.getElementById("sim-out-triage");
        if (triageDisplay) triageDisplay.textContent = triageRate + "%";
    }

    alphaSlider.addEventListener("input", updateSimulation);
    if (energySlider) energySlider.addEventListener("input", updateSimulation);
    if (tempSlider) tempSlider.addEventListener("input", updateSimulation);

    updateSimulation();
}

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
            methods: "run_ingest(): Entrypoint for DVC stage 'ingest'. Calls INatAPIClient and exports data/raw_manifest.parquet.",
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
            methods: "run_audit_license(): Entrypoint for DVC stage 'audit_license'. Uses LicenseFilter and exports data/dataset_manifest.parquet.",
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
            methods: "run_feature_extraction(): Extracts representations and writes data/{train,val,cal,test}_embeddings.pt.",
            github: "https://github.com/foersben/taxon-vision-mlops/blob/develop/src/taxon_vision/pipelines/extract_features.py"
        },
        {
            subpkg: "taxon_vision.pipelines",
            file: "train.py",
            tier: 1,
            tierName: "Tier 1: Orchestration",
            badgeClass: "tier-badge-1",
            complexity: 9,
            dvcStage: "train",
            responsibilities: "Optimizes linear probe head on pre-extracted embeddings using Class-Balanced Loss.",
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
            dvcStage: "export_onnx",
            responsibilities: "Exports linear probe and backbone to ONNX Opset 17 and performs MinMax INT8 PTQ.",
            classes: "run_export(config_path: Path | None = None) -> None",
            methods: "run_export(): Calls ONNXExporter and ModelQuantizer to generate models/checkpoints/taxon_vision_quantized.onnx.",
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
            responsibilities: "Computes conformal non-conformity scores, empirical quantile q_hat, and energy threshold tau_ood.",
            classes: "run_calibration(config_path: Path | None = None) -> None",
            methods: "run_calibration(): Calibrates SplitConformalPredictor and outputs config/conformal_config.yaml.",
            github: "https://github.com/foersben/taxon-vision-mlops/blob/develop/src/taxon_vision/pipelines/calibrate.py"
        },
        {
            subpkg: "taxon_vision.pipelines",
            file: "evaluate.py",
            tier: 1,
            tierName: "Tier 1: Orchestration",
            badgeClass: "tier-badge-1",
            complexity: 10,
            dvcStage: "evaluate",
            responsibilities: "Validates statistical coverage 1 - alpha, accuracy, and p95 latency SLA on test split.",
            classes: "run_evaluation(config_path: Path | None = None) -> None",
            methods: "run_evaluation(): Asserts coverage bound and writes reports/evaluation_metrics.json.",
            github: "https://github.com/foersben/taxon-vision-mlops/blob/develop/src/taxon_vision/pipelines/evaluate.py"
        },
        {
            subpkg: "taxon_vision.service",
            file: "api.py",
            tier: 1,
            tierName: "Tier 1: Orchestration",
            badgeClass: "tier-badge-1",
            complexity: 4,
            dvcStage: "N/A (Serving Boundary)",
            responsibilities: "Instantiates FastAPI application, registers route handlers, CORS middleware, and exception handlers.",
            classes: "create_app() -> FastAPI",
            methods: "create_app(): Application factory attaching /predict, /dashboard, /feedback, and /health routers.",
            github: "https://github.com/foersben/taxon-vision-mlops/blob/develop/src/taxon_vision/service/api.py"
        },
        {
            subpkg: "taxon_vision.service",
            file: "inference.py",
            tier: 1,
            tierName: "Tier 1: Orchestration",
            badgeClass: "tier-badge-1",
            complexity: 12,
            dvcStage: "N/A (Serving Boundary)",
            responsibilities: "Coordinates zero-allocation ONNX Runtime inference, conformal prediction filtering, and EigenCAM generation.",
            classes: "class InferenceService",
            methods: "predict_image(image_bytes: bytes) -> PredictionResponse\nwarmup(): Pre-allocates ONNX IO buffers",
            github: "https://github.com/foersben/taxon-vision-mlops/blob/develop/src/taxon_vision/service/inference.py"
        },
        {
            subpkg: "taxon_vision.service",
            file: "storage.py",
            tier: 1,
            tierName: "Tier 1: Orchestration",
            badgeClass: "tier-badge-1",
            complexity: 5,
            dvcStage: "N/A (Serving Boundary)",
            responsibilities: "Manages disk persistence of flagged, out-of-distribution, or empty-set observation images.",
            classes: "class ObservationStorage",
            methods: "save_flagged_observation(image_bytes: bytes, metadata: dict) -> str\nlist_flagged(limit: int) -> list[dict]",
            github: "https://github.com/foersben/taxon-vision-mlops/blob/develop/src/taxon_vision/service/storage.py"
        },
        {
            subpkg: "taxon_vision.models",
            file: "factory.py",
            tier: 2,
            tierName: "Tier 2: ML Engines",
            badgeClass: "tier-badge-2",
            complexity: 8,
            dvcStage: "extract_features",
            responsibilities: "Foundation model trunk factory supporting timm and Hugging Face architectures.",
            classes: "class BackboneFactory",
            methods: "create_backbone(model_name: str, pretrained: bool = True) -> tuple[nn.Module, int]",
            github: "https://github.com/foersben/taxon-vision-mlops/blob/develop/src/taxon_vision/models/factory.py"
        },
        {
            subpkg: "taxon_vision.models",
            file: "head.py",
            tier: 2,
            tierName: "Tier 2: ML Engines",
            badgeClass: "tier-badge-2",
            complexity: 5,
            dvcStage: "train, export_onnx",
            responsibilities: "PyTorch classification head with linear projection, LayerNorm, and dropout.",
            classes: "class TaxonClassifier(nn.Module)",
            methods: "forward(x: torch.Tensor) -> torch.Tensor\npredict_proba(x: torch.Tensor) -> torch.Tensor",
            github: "https://github.com/foersben/taxon-vision-mlops/blob/develop/src/taxon_vision/models/head.py"
        },
        {
            subpkg: "taxon_vision.models",
            file: "loss.py",
            tier: 2,
            tierName: "Tier 2: ML Engines",
            badgeClass: "tier-badge-2",
            complexity: 7,
            dvcStage: "train",
            responsibilities: "Effective-number class-balanced cross-entropy and focal loss computation.",
            classes: "class ClassBalancedLoss(nn.Module)",
            methods: "forward(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor",
            github: "https://github.com/foersben/taxon-vision-mlops/blob/develop/src/taxon_vision/models/loss.py"
        },
        {
            subpkg: "taxon_vision.models",
            file: "training/runner.py",
            tier: 2,
            tierName: "Tier 2: ML Engines",
            badgeClass: "tier-badge-2",
            complexity: 13,
            dvcStage: "train",
            responsibilities: "Batch training execution, validation metrics tracking, learning rate decay, and checkpoint saving.",
            classes: "class TrainingRunner",
            methods: "execute(params: TrainingParams) -> TrainingResult\nvalidate(epoch: int) -> float",
            github: "https://github.com/foersben/taxon-vision-mlops/blob/develop/src/taxon_vision/models/training/runner.py"
        },
        {
            subpkg: "taxon_vision.inference",
            file: "onnx_engine.py",
            tier: 2,
            tierName: "Tier 2: ML Engines",
            badgeClass: "tier-badge-2",
            complexity: 9,
            dvcStage: "evaluate, service",
            responsibilities: "Zero-allocation ONNX Runtime inference engine with pinned IO binding.",
            classes: "class ONNXInferenceEngine",
            methods: "predict(image_tensor: np.ndarray) -> np.ndarray\npredict_with_metadata(image_tensor: np.ndarray) -> dict",
            github: "https://github.com/foersben/taxon-vision-mlops/blob/develop/src/taxon_vision/inference/onnx_engine.py"
        },
        {
            subpkg: "taxon_vision.inference",
            file: "cam.py",
            tier: 2,
            tierName: "Tier 2: ML Engines",
            badgeClass: "tier-badge-2",
            complexity: 10,
            dvcStage: "service",
            responsibilities: "Generates EigenCAM visual saliency attribution heatmaps using SVD.",
            classes: "class EigenCAMExplainer",
            methods: "generate_cam(image: torch.Tensor, target_layer: str) -> np.ndarray",
            github: "https://github.com/foersben/taxon-vision-mlops/blob/develop/src/taxon_vision/inference/cam.py"
        },
        {
            subpkg: "taxon_vision.inference",
            file: "quantizer.py",
            tier: 2,
            tierName: "Tier 2: ML Engines",
            badgeClass: "tier-badge-2",
            complexity: 11,
            dvcStage: "export_onnx",
            responsibilities: "Static INT8 Post-Training Quantization (PTQ) via ONNX Runtime quantization API.",
            classes: "class ModelQuantizer",
            methods: "quantize_static(input_path: Path, output_path: Path, reader: Any) -> Path",
            github: "https://github.com/foersben/taxon-vision-mlops/blob/develop/src/taxon_vision/inference/quantizer.py"
        },
        {
            subpkg: "taxon_vision.uncertainty",
            file: "conformal.py",
            tier: 2,
            tierName: "Tier 2: ML Engines",
            badgeClass: "tier-badge-2",
            complexity: 8,
            dvcStage: "calibrate, evaluate, service",
            responsibilities: "Computes empirical quantile q_hat and generates prediction sets satisfying P(Y in C(X)) >= 1 - alpha.",
            classes: "class SplitConformalPredictor",
            methods: "calibrate(probs: np.ndarray, labels: np.ndarray, alpha: float) -> float\npredict_set(probs: np.ndarray) -> list[list[int]]",
            github: "https://github.com/foersben/taxon-vision-mlops/blob/develop/src/taxon_vision/uncertainty/conformal.py"
        },
        {
            subpkg: "taxon_vision.uncertainty",
            file: "ood_detector.py",
            tier: 2,
            tierName: "Tier 2: ML Engines",
            badgeClass: "tier-badge-2",
            complexity: 7,
            dvcStage: "calibrate, evaluate, service",
            responsibilities: "Calculates Helmholtz Free Energy and detects out-of-distribution or corrupted observations.",
            classes: "class EnergyOODDetector",
            methods: "compute_energy(logits: np.ndarray, temp: float = 1.0) -> np.ndarray\ncalibrate_threshold(logits: np.ndarray, tpr: float = 0.95) -> float\nis_ood(logits: np.ndarray) -> np.ndarray",
            github: "https://github.com/foersben/taxon-vision-mlops/blob/develop/src/taxon_vision/uncertainty/ood_detector.py"
        },
        {
            subpkg: "taxon_vision.active_learning",
            file: "query.py",
            tier: 2,
            tierName: "Tier 2: ML Engines",
            badgeClass: "tier-badge-2",
            complexity: 9,
            dvcStage: "active_learning",
            responsibilities: "BADGE and CoreSet active learning query strategies for informative sample acquisition.",
            classes: "class BADGEQueryStrategy\nclass CoreSetQueryStrategy",
            methods: "select_samples(embeddings: np.ndarray, logits: np.ndarray, budget: int) -> list[int]",
            github: "https://github.com/foersben/taxon-vision-mlops/blob/develop/src/taxon_vision/active_learning/query.py"
        },
        {
            subpkg: "taxon_vision.monitoring",
            file: "telemetry.py",
            tier: 2,
            tierName: "Tier 2: ML Engines",
            badgeClass: "tier-badge-2",
            complexity: 6,
            dvcStage: "service",
            responsibilities: "Prometheus metric counters, gauges, and latency histograms for real-time observability.",
            classes: "class PrometheusTelemetry",
            methods: "record_inference(latency_sec: float, set_size: int, is_ood: bool) -> None",
            github: "https://github.com/foersben/taxon-vision-mlops/blob/develop/src/taxon_vision/monitoring/telemetry.py"
        },
        {
            subpkg: "taxon_vision.data",
            file: "inat_client.py",
            tier: 3,
            tierName: "Tier 3: Domain & Data",
            badgeClass: "tier-badge-3",
            complexity: 8,
            dvcStage: "ingest",
            responsibilities: "Paginated HTTP client querying the official iNaturalist v1 REST API.",
            classes: "class INatAPIClient",
            methods: "fetch_observations(taxon_id: int, per_page: int) -> list[dict]",
            github: "https://github.com/foersben/taxon-vision-mlops/blob/develop/src/taxon_vision/data/inat_client.py"
        },
        {
            subpkg: "taxon_vision.data",
            file: "license_filter.py",
            tier: 3,
            tierName: "Tier 3: Domain & Data",
            badgeClass: "tier-badge-3",
            complexity: 7,
            dvcStage: "audit_license",
            responsibilities: "Filters observations by permissible open licenses (CC0, CC-BY, CC-BY-NC) and retains attribution.",
            classes: "class LicenseFilter",
            methods: "filter_compliant_records(records: list[dict]) -> tuple[list[dict], dict]",
            github: "https://github.com/foersben/taxon-vision-mlops/blob/develop/src/taxon_vision/data/license_filter.py"
        },
        {
            subpkg: "taxon_vision.data",
            file: "s3_streamer.py",
            tier: 3,
            tierName: "Tier 3: Domain & Data",
            badgeClass: "tier-badge-3",
            complexity: 12,
            dvcStage: "extract_features",
            responsibilities: "Streams raw observation JPEG bytes directly from AWS S3 Open Data buckets with local caching.",
            classes: "class S3ImageStreamer",
            methods: "fetch_image_bytes(photo_id: int | str) -> bytes\nstream_observation_images(observations: list[dict]) -> Iterator[tuple[dict, bytes]]",
            github: "https://github.com/foersben/taxon-vision-mlops/blob/develop/src/taxon_vision/data/s3_streamer.py"
        },
        {
            subpkg: "taxon_vision.domain",
            file: "taxonomy.py",
            tier: 3,
            tierName: "Tier 3: Domain & Data",
            badgeClass: "tier-badge-3",
            complexity: 4,
            dvcStage: "All (Domain)",
            responsibilities: "DarwinCore taxonomic tree nodes and catalog lookup entity.",
            classes: "class TaxonNode(BaseModel)\nclass TaxonomyCatalog(BaseModel)",
            methods: "get_by_id(taxon_id: int) -> TaxonNode | None",
            github: "https://github.com/foersben/taxon-vision-mlops/blob/develop/src/taxon_vision/domain/taxonomy.py"
        },
        {
            subpkg: "taxon_vision.domain",
            file: "license.py",
            tier: 3,
            tierName: "Tier 3: Domain & Data",
            badgeClass: "tier-badge-3",
            complexity: 3,
            dvcStage: "All (Domain)",
            responsibilities: "StrEnum of Creative Commons open licenses and attribution metadata entity.",
            classes: "class OpenLicense(StrEnum)\nclass AttributionRecord(BaseModel)",
            methods: "OpenLicense.is_commercial_permissible() -> bool",
            github: "https://github.com/foersben/taxon-vision-mlops/blob/develop/src/taxon_vision/domain/license.py"
        },
        {
            subpkg: "taxon_vision.config",
            file: "schemas.py",
            tier: 4,
            tierName: "Tier 4: Configuration",
            badgeClass: "tier-badge-4",
            complexity: 5,
            dvcStage: "All (Config)",
            responsibilities: "Pydantic settings schemas for service, models, training, conformal calibration, and Optuna.",
            classes: "class Settings(BaseModel)\nclass ServiceSettings(BaseModel)\nclass ModelSettings(BaseModel)",
            methods: "Settings.model_validate(obj: Any) -> Settings",
            github: "https://github.com/foersben/taxon-vision-mlops/blob/develop/src/taxon_vision/config/schemas.py"
        },
        {
            subpkg: "taxon_vision.config",
            file: "loader.py",
            tier: 4,
            tierName: "Tier 4: Configuration",
            badgeClass: "tier-badge-4",
            complexity: 10,
            dvcStage: "All (Config)",
            responsibilities: "3-Tier Configuration Precedence Pyramid loader, deep merge, and LRU cache.",
            classes: "get_settings(config_path: Path | None = None) -> Settings\nreset_settings() -> None",
            methods: "_deep_merge(base: dict, override: dict) -> dict\n_apply_direct_env_overrides(data: dict) -> dict",
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

/**
 * Interactive 7-Stage Epistemic Diamond DVC DAG & Live Metrics Inspector (Part 2)
 */
function initDVCExplorer() {
    const mountPoint = document.getElementById("taxon-dvc-explorer");
    if (!mountPoint) return;
    if (mountPoint.dataset.initialized === "true") return;
    mountPoint.dataset.initialized = "true";

    const stages = [
        {
            id: "ingestion",
            number: "1",
            title: "1. Data Ingestion (Citizen-Science)",
            dvcStage: "ingestion",
            cmd: "pixi run -e dev python -m taxon_vision.pipelines.ingest --out data/manifests/raw_manifest.parquet",
            deps: [
                "config/taxa_catalog.yaml",
                "src/taxon_vision/data/inat_client.py",
                "src/taxon_vision/pipelines/ingest.py"
            ],
            params: [
                "ingestion.per_page = 200",
                "ingestion.taxa_to_fetch = [10 Taxa IDs]"
            ],
            outs: [
                "data/manifests/raw_manifest.parquet"
            ],
            metrics: "N/A (Metadata manifest stage)",
            algorithm: [
                "Parses Level 1 perimeter taxa catalog from config/taxa_catalog.yaml.",
                "Executes paginated HTTP queries to iNaturalist v1 REST API (per_page=200).",
                "Filters observation records requiring research-grade verification and coordinate retention.",
                "Serializes metadata into Snappy-compressed columnar Parquet format (< 500 KB on disk)."
            ],
            triggers: "Triggered when config/taxa_catalog.yaml, inat_client.py, ingest.py, or ingestion.* parameters change.",
            downstreamRadius: "Invalidates: Stage 2 (audit_license), Stage 3 (extract_features), Stage 4 (train), Stage 5 (export), Stage 6 (calibrate), and Stage 7 (eval)."
        },
        {
            id: "audit_license",
            number: "2",
            title: "2. Open License Audit (Legal Compliance)",
            dvcStage: "audit_license",
            cmd: "pixi run -e dev python -m taxon_vision.pipelines.audit_license --manifest data/manifests/raw_manifest.parquet --out-manifest data/manifests/dataset_manifest.parquet --out reports/license_audit.json",
            deps: [
                "data/manifests/raw_manifest.parquet",
                "src/taxon_vision/data/license_filter.py",
                "src/taxon_vision/domain/license.py",
                "src/taxon_vision/pipelines/audit_license.py"
            ],
            params: [
                "Strict Open Licensing Mandate (CC0, CC-BY, CC-BY-NC)"
            ],
            outs: [
                "data/manifests/dataset_manifest.parquet"
            ],
            metrics: "reports/license_audit.json: 1,689 / 2,000 Verified (84.5%), 311 Violations Screened, Status: PASSED",
            algorithm: [
                "Reads raw_manifest.parquet into zero-copy Apache Arrow table.",
                "Evaluates each observation's license_code against the OpenLicense StrEnum.",
                "Rejects un-licensed images, proprietary copyrights, and commercial-restricted media.",
                "Retains photographer attribution strings and writes verified dataset_manifest.parquet.",
                "Emits reports/license_audit.json tracking compliance percentage."
            ],
            triggers: "Triggered when raw_manifest.parquet, license_filter.py, domain/license.py, or audit_license.py changes.",
            downstreamRadius: "Invalidates: Stage 3 (extract_features) and all downstream training, calibration, and evaluation stages."
        },
        {
            id: "extract_features",
            number: "3",
            title: "3. Feature Extraction (Foundation Embedding Caching)",
            dvcStage: "extract_features",
            cmd: "pixi run -e dev python -m taxon_vision.pipelines.extract_features --extractor dinov3 --image-size 224 --batch-size 32 --val-split 0.10 --cal-split 0.10 --test-split 0.10 --manifest data/manifests/dataset_manifest.parquet --out-dir data/embeddings",
            deps: [
                "config/taxa_catalog.yaml",
                "data/manifests/dataset_manifest.parquet",
                "src/taxon_vision/data/s3_streamer.py",
                "src/taxon_vision/models/factory.py",
                "src/taxon_vision/pipelines/extract_features.py"
            ],
            params: [
                "model.extractor = dinov3",
                "model.image_size = 224",
                "feature_extraction.batch_size = 32",
                "feature_extraction.val_split = 0.10",
                "feature_extraction.cal_split = 0.10",
                "feature_extraction.test_split = 0.10"
            ],
            outs: [
                "data/embeddings/train_embeddings.pt (pushed to local-seaweedfs)",
                "data/embeddings/val_embeddings.pt (pushed to local-seaweedfs)",
                "data/embeddings/cal_embeddings.pt (pushed to local-seaweedfs)",
                "data/embeddings/test_embeddings.pt (pushed to local-seaweedfs)"
            ],
            metrics: "N/A (Tensors pushed to in-cluster SeaweedFS S3 edge cache)",
            algorithm: [
                "Instantiates frozen foundation backbone (DINOv3, BioCLIP-2, or MobileNetV4) via BackboneFactory.",
                "Streams raw observation JPEG bytes on demand from s3://inaturalist-open-data with local LRU caching.",
                "Normalizes images (ImageNet mean & std) and computes 384-dimensional representation vectors.",
                "Executes deterministic 4-way stratified hash split (70% train, 10% val, 10% cal, 10% test, seed=42).",
                "Pushes memory-mapped PyTorch .pt tensors directly to in-cluster SeaweedFS S3."
            ],
            triggers: "Triggered when model.extractor, model.image_size, split ratios, dataset_manifest.parquet, or backbone factory code changes.",
            downstreamRadius: "Invalidates: Stage 4 (train), Stage 5 (export), Stage 6 (calibrate), and Stage 7 (eval). Heavy compute boundary."
        },
        {
            id: "train",
            number: "4",
            title: "4. Linear Probe Training (Class-Balanced Loss)",
            dvcStage: "train",
            cmd: "pixi run -e dev python -m taxon_vision.pipelines.train --extractor dinov3 --epochs 5 --batch-size 32 --lr 0.001 --train-embeddings data/embeddings/train_embeddings.pt --val-embeddings data/embeddings/val_embeddings.pt --checkpoint-path models/checkpoints/head.pt --metrics-out reports/train_metrics.json",
            deps: [
                "data/embeddings/train_embeddings.pt",
                "data/embeddings/val_embeddings.pt",
                "src/taxon_vision/models/loss.py",
                "src/taxon_vision/models/training/",
                "src/taxon_vision/pipelines/train.py"
            ],
            params: [
                "model.extractor = dinov3",
                "training.batch_size = 32",
                "training.cb_beta = 0.9999",
                "training.epochs = 5",
                "training.learning_rate = 0.001",
                "training.tune = false",
                "training.weight_decay = 0.0001"
            ],
            outs: [
                "models/checkpoints/head.pt (pushed to origin/DagsHub)"
            ],
            metrics: "reports/train_metrics.json: Train Loss: 0.7524, Val Loss: 0.7694, Val Accuracy: 82.84%, Duration: 1.58s",
            algorithm: [
                "Loads pre-extracted train_embeddings.pt and val_embeddings.pt into memory.",
                "Calculates Class-Balanced effective number weights: E_n = (1 - beta) / (1 - beta^n) with beta=0.9999.",
                "Optimizes linear probe head (TaxonClassifier) using AdamW with weight decay 1e-4 for 5 epochs.",
                "Tracks cross-entropy loss and validation accuracy per epoch, logging to MLflow.",
                "Exports champion probe weights to models/checkpoints/head.pt."
            ],
            triggers: "Triggered when training hyperparameters (lr, epochs, cb_beta), training embeddings, or loss code changes.",
            downstreamRadius: "Invalidates: Stage 5 (export), Stage 6 (calibrate), and Stage 7 (eval). Does NOT invalidate Stage 3 feature extraction!"
        },
        {
            id: "export",
            number: "5",
            title: "5. Graph Fusion & INT8 PTQ (ONNX Runtime)",
            dvcStage: "export",
            cmd: "pixi run -e dev python -m taxon_vision.pipelines.export_onnx --extractor dinov3 --image-size 224 --checkpoint models/checkpoints/head.pt --quantize-int8 --quantize-mode dynamic --quantized-output models/checkpoints/model_int8.onnx",
            deps: [
                "models/checkpoints/head.pt",
                "src/taxon_vision/inference/onnx_exporter.py",
                "src/taxon_vision/inference/quantizer.py",
                "src/taxon_vision/pipelines/export_onnx.py"
            ],
            params: [
                "export.quantize_int8 = true",
                "model.extractor = dinov3",
                "model.image_size = 224"
            ],
            outs: [
                "models/checkpoints/model_int8.onnx (pushed to origin/DagsHub)"
            ],
            metrics: "N/A (Optimized INT8 runtime graph pushed to DagsHub remote)",
            algorithm: [
                "Instantiates full computational graph fusing the DINOv3 backbone and trained linear probe head.",
                "Exports unified PyTorch model to ONNX Opset 17 with dynamic batch axes.",
                "Streams representative unlabelled images using TaxonCalibrationDataReader.",
                "Applies MinMax Static Post-Training Quantization (PTQ) to convert FP32 weights and activations to INT8.",
                "Validates numerical stability and exports models/checkpoints/model_int8.onnx."
            ],
            triggers: "Triggered when head.pt, export.quantize_int8, onnx_exporter.py, or quantizer.py changes.",
            downstreamRadius: "Invalidates: Stage 6 (calibrate) and Stage 7 (eval)."
        },
        {
            id: "calibrate",
            number: "6",
            title: "6. Conformal & OOD Calibration (Mathematical Safety)",
            dvcStage: "calibrate",
            cmd: "pixi run -e dev python -m taxon_vision.pipelines.calibrate --onnx-model models/checkpoints/model_int8.onnx --embeddings data/embeddings/cal_embeddings.pt --out config/conformal_config.yaml --metrics-out reports/conformal_metrics.json --alpha 0.05 --ood-tpr 0.95",
            deps: [
                "data/embeddings/cal_embeddings.pt",
                "models/checkpoints/model_int8.onnx",
                "src/taxon_vision/inference/onnx_engine.py",
                "src/taxon_vision/pipelines/calibrate.py",
                "src/taxon_vision/uncertainty/conformal.py",
                "src/taxon_vision/uncertainty/ood_detector.py"
            ],
            params: [
                "conformal.alpha = 0.05 (1 - alpha = 95% Coverage Guarantee)",
                "conformal.min_confidence = 0.60",
                "conformal.ood_tpr = 0.95"
            ],
            outs: [
                "config/conformal_config.yaml (tracked in Git)"
            ],
            metrics: "reports/conformal_metrics.json: Empirical Cal Coverage: 96.45%, q_hat: 0.8405, tau_ood: -2.2792",
            algorithm: [
                "Runs zero-allocation ONNX Runtime inference over unseen cal_embeddings.pt (N=169).",
                "Calculates true-class non-conformity scores: s_i = 1 - f(x_i)_{y_i}.",
                "Computes finite-sample inflated quantile q_hat at level ceil((n+1)(1-alpha))/n.",
                "Computes Helmholtz Free Energy scores E(x) = -T * log sum(exp(logits/T)).",
                "Establishes OOD decision threshold tau_ood anchoring In-Distribution TPR to exactly 95%.",
                "Writes calibrated parameters to config/conformal_config.yaml."
            ],
            triggers: "Triggered when cal_embeddings.pt, model_int8.onnx, conformal.alpha, or conformal.ood_tpr changes.",
            downstreamRadius: "Invalidates: Stage 7 (eval)."
        },
        {
            id: "eval",
            number: "7",
            title: "7. Pareto Evaluation & Latency SLA (Deployment Gate)",
            dvcStage: "eval",
            cmd: "pixi run -e dev python -m taxon_vision.pipelines.evaluate --onnx-model models/checkpoints/model_int8.onnx --embeddings data/embeddings/test_embeddings.pt --conformal-config config/conformal_config.yaml --out docs/assets/pareto_results.json --metrics-out reports/pareto_evaluation.json --latency-sla-ms 25.0",
            deps: [
                "config/conformal_config.yaml",
                "data/embeddings/test_embeddings.pt",
                "models/checkpoints/model_int8.onnx",
                "src/taxon_vision/inference/onnx_engine.py",
                "src/taxon_vision/pipelines/evaluate.py",
                "src/taxon_vision/uncertainty/conformal.py",
                "src/taxon_vision/uncertainty/ood_detector.py"
            ],
            params: [
                "conformal.alpha = 0.05",
                "evaluation.latency_sla_ms = 25.0"
            ],
            outs: [
                "docs/assets/pareto_results.json"
            ],
            metrics: "reports/pareto_evaluation.json: Test Accuracy: 89.35%, Empirical Coverage: 96.45%, Mean Set Size: 1.72, Throughput: 20.7 FPS",
            algorithm: [
                "Executes ONNX Runtime engine over holdout test_embeddings.pt (N=169).",
                "Forms prediction sets C(X) = {y : f(x)_y >= 1 - q_hat}.",
                "Asserts empirical test coverage satisfies 1 - alpha bound (96.45% >= 95.0%).",
                "Executes 100 warm-up cycles and 500 timed inference loops to benchmark p50 and p95 latency.",
                "Asserts p95 latency SLA <= 25.0 ms (flags deployment violation if SLA breached).",
                "Writes reports/pareto_evaluation.json and exports Pareto frontier results for documentation portal."
            ],
            triggers: "Triggered when test_embeddings.pt, model_int8.onnx, conformal_config.yaml, or latency_sla_ms changes.",
            downstreamRadius: "Terminal stage. Exits with code 1 upon statistical coverage failure or SLA breach, preventing release."
        }
    ];

    let currentStage = stages[0];

    // Render skeleton
    mountPoint.innerHTML = `
        <div class="dvc-explorer">
            <div class="dvc-metrics-summary">
                <div class="dvc-metric-pill">
                    <span class="title">Stage 2: License Audit</span>
                    <span class="value">1,689 / 2,000</span>
                    <span class="subtext">84.5% Open Licensed (311 Violations Rejected)</span>
                </div>
                <div class="dvc-metric-pill">
                    <span class="title">Stage 4: Probe Training</span>
                    <span class="value">82.84% Val Acc</span>
                    <span class="subtext">Loss: 0.7524 → 0.7694 (5 Epochs in 1.58s)</span>
                </div>
                <div class="dvc-metric-pill">
                    <span class="title">Stage 6: Conformal Cal</span>
                    <span class="value">96.45% Cov</span>
                    <span class="subtext">q̂: 0.8405 | τ_ood: -2.2792 (α = 0.05)</span>
                </div>
                <div class="dvc-metric-pill">
                    <span class="title">Stage 7: Pareto Eval</span>
                    <span class="value">89.35% Test Acc</span>
                    <span class="subtext">Set Size: 1.72 | p95: 51.99 ms | $2.60/M</span>
                </div>
            </div>
            <div class="dvc-stage-nav" id="dvc-stage-nav"></div>
            <div class="dvc-stage-body" id="dvc-stage-body"></div>
        </div>
    `;

    const navContainer = document.getElementById("dvc-stage-nav");
    const bodyContainer = document.getElementById("dvc-stage-body");

    function renderNav() {
        navContainer.innerHTML = stages.map(function (s) {
            const isActive = s === currentStage ? "active" : "";
            return `
                <button class="dvc-stage-tab ${isActive}" data-id="${s.id}">
                    <span>${s.title}</span>
                </button>
            `;
        }).join("");

        const tabs = navContainer.querySelectorAll(".dvc-stage-tab");
        tabs.forEach(function (tab) {
            tab.addEventListener("click", function () {
                const stageId = tab.getAttribute("data-id");
                currentStage = stages.find(function (s) { return s.id === stageId; }) || currentStage;
                renderNav();
                renderStage();
            });
        });
    }

    function renderStage() {
        const s = currentStage;
        bodyContainer.innerHTML = `
            <div class="dvc-stage-header">
                <div class="dvc-stage-title">${s.title}</div>
                <div style="font-size: 0.85rem; opacity: 0.8;">DVC Stage Identifier: <code>${s.dvcStage}</code></div>
            </div>

            <div class="explorer-section-title">Execution Command (dvc.yaml)</div>
            <div class="explorer-code-block">${s.cmd}</div>

            <div class="dvc-grid-two-col">
                <div>
                    <div class="explorer-section-title">Input Dependencies (deps)</div>
                    <div class="explorer-code-block">${s.deps.join("\n")}</div>

                    <div class="explorer-section-title">Tracked Hyperparameters (params)</div>
                    <div class="explorer-code-block">${s.params.join("\n")}</div>
                </div>
                <div>
                    <div class="explorer-section-title">Produced Outputs (outs)</div>
                    <div class="explorer-code-block">${s.outs.join("\n")}</div>

                    <div class="explorer-section-title">Stage Metrics (metrics)</div>
                    <div class="explorer-code-block">${s.metrics}</div>
                </div>
            </div>

            <div class="explorer-section-title">Step-by-Step Execution Algorithm</div>
            <ol style="margin: 0.5rem 0 1rem 1.25rem; line-height: 1.5; font-size: 0.9rem;">
                ${s.algorithm.map(function (step) { return `<li>${step}</li>`; }).join("")}
            </ol>

            <div class="dvc-blast-radius-box">
                <div class="header">⚡ Invalidation Blast Radius & Dependency Triggers</div>
                <div style="font-size: 0.85rem; line-height: 1.5; margin-bottom: 0.35rem;">
                    <strong>Upstream Triggers:</strong> ${s.triggers}
                </div>
                <div style="font-size: 0.85rem; line-height: 1.5; color: var(--tv-accent);">
                    <strong>Downstream Invalidation:</strong> ${s.downstreamRadius}
                </div>
            </div>
        `;
    }

    renderNav();
    renderStage();
}
