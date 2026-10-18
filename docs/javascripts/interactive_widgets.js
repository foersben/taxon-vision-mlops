/**
 * TaxonVision-MLOps Interactive Widgets & Visualizer Controller
 */

document.addEventListener("DOMContentLoaded", function () {
    initFullscreenHandlers();
    initConformalSimulators();
    initCodebaseExplorer();
    initDVCExplorer();
    initEpistemicSimulator();
    initParetoExplorer();
});

// Support dynamic navigation in Material for MkDocs / Zensical
if (typeof document$ !== "undefined") {
    document$.subscribe(function () {
        initFullscreenHandlers();
        initConformalSimulators();
        initCodebaseExplorer();
        initDVCExplorer();
        initEpistemicSimulator();
        initParetoExplorer();
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

/**
 * Interactive Conformal & Thermodynamic OOD Live Simulator (Part 3)
 */
function initEpistemicSimulator() {
    const mountPoint = document.getElementById("taxon-epistemic-simulator");
    if (!mountPoint) return;
    if (mountPoint.dataset.initialized === "true") return;
    mountPoint.dataset.initialized = "true";

    const presets = {
        confident_inlier: {
            id: "confident_inlier",
            label: "Confident Inlier (Monarch)",
            description: "High-confidence observation of Monarch (Danaus plexippus). Strong single activation, low Helmholtz free energy, and low entropy yield a singleton prediction set.",
            logits: [5.2, -1.2, -1.8, -2.4, -3.1],
            temp: 1.0,
            alpha: 0.05,
            tau: -2.28
        },
        sister_mimicry: {
            id: "sister_mimicry",
            label: "Sister-Taxa Mimicry (Monarch vs Viceroy)",
            description: "Batesian mimicry pair (Monarch vs. Viceroy). Close dominant logits, low Helmholtz free energy, and moderate entropy trigger Vision-RAG differential diagnosis.",
            logits: [4.2, 4.0, -1.5, -2.2, -3.0],
            temp: 1.0,
            alpha: 0.05,
            tau: -2.28
        },
        empty_frame: {
            id: "empty_frame",
            label: "Empty Frame / Foliage Blur",
            description: "Wind-triggered camera trap capture of blurry vegetation. Near-zero diffuse logits yield maximum Normalized Shannon Entropy (H̄ → 1.0), triggering automated clutter rejection.",
            logits: [-0.1, 0.05, -0.15, 0.1, -0.05],
            temp: 1.0,
            alpha: 0.05,
            tau: -2.28
        },
        semantic_ood: {
            id: "semantic_ood",
            label: "Semantic OOD Vehicle / Mineral",
            description: "Non-biological object (e.g., field truck or rock). Negative logits across all biological classes push Helmholtz Free Energy above τ_OOD, triggering immediate human triage.",
            logits: [-4.5, -4.8, -5.2, -4.9, -5.4],
            temp: 1.0,
            alpha: 0.05,
            tau: -2.28
        }
    };

    const species = [
        { name: "Danaus plexippus", common: "Monarch Butterfly", color: "#ff7043" },
        { name: "Limenitis archippus", common: "Viceroy Butterfly", color: "#ffa726" },
        { name: "Papilio glaucus", common: "Eastern Tiger Swallowtail", color: "#ffd54f" },
        { name: "Vanessa cardui", common: "Painted Lady", color: "#ba68c8" },
        { name: "Pieris rapae", common: "Cabbage White", color: "#4dd0e1" }
    ];

    let currentLogits = [...presets.confident_inlier.logits];
    let currentTemp = 1.0;
    let currentAlpha = 0.05;
    let currentTau = -2.28;
    let activePreset = "confident_inlier";

    mountPoint.innerHTML = `
        <div class="epistemic-simulator">
            <div class="sim-presets-bar">
                <span class="preset-label"><strong>Quick Presets:</strong></span>
                <button class="sim-preset-btn active" data-preset="confident_inlier">1. Confident Inlier</button>
                <button class="sim-preset-btn" data-preset="sister_mimicry">2. Sister-Taxa Mimicry</button>
                <button class="sim-preset-btn" data-preset="empty_frame">3. Empty Frame / Blur</button>
                <button class="sim-preset-btn" data-preset="semantic_ood">4. Semantic OOD</button>
            </div>

            <div class="sim-preset-description" id="sim-preset-desc">
                ${presets.confident_inlier.description}
            </div>

            <div class="sim-parameters-grid">
                <div class="sim-param-col">
                    <label for="sim-temp-input">
                        <span>Logit Temperature (T)</span>
                        <span id="sim-temp-display" class="badge-tag info">1.0</span>
                    </label>
                    <input type="range" id="sim-temp-input" min="0.5" max="3.0" step="0.1" value="1.0">
                    <small>Softmax scale: z / T. Preserves ordinal ranking while tuning calibration confidence.</small>
                </div>
                <div class="sim-param-col">
                    <label for="sim-alpha-input">
                        <span>Significance Level (&alpha;)</span>
                        <span id="sim-alpha-display" class="badge-tag warning">0.05</span>
                    </label>
                    <input type="range" id="sim-alpha-input" min="0.01" max="0.20" step="0.01" value="0.05">
                    <small>Coverage guarantee = 1 - &alpha; (95.0%). Calibrated on N_cal = 169 split.</small>
                </div>
                <div class="sim-param-col">
                    <label for="sim-tau-input">
                        <span>OOD Energy Cutoff (&tau;<sub>OOD</sub>)</span>
                        <span id="sim-tau-display" class="badge-tag danger">-2.28</span>
                    </label>
                    <input type="range" id="sim-tau-input" min="-10.0" max="2.0" step="0.1" value="-2.28">
                    <small>Helmholtz Free Energy cutoff anchoring 95% In-Distribution TPR.</small>
                </div>
            </div>

            <div class="sim-interactive-columns">
                <div class="sim-logits-column">
                    <div class="sim-section-heading">Raw Logits z &isin; &reals;<sup>5</sup> (Linear Probe Output)</div>
                    <div id="sim-logits-inputs"></div>
                </div>

                <div class="sim-bars-column">
                    <div class="sim-section-heading">
                        <span>Calibrated Probabilities &pi;&#770;<sub>k</sub>(x)</span>
                        <span class="threshold-legend" id="sim-threshold-legend">Threshold (1 - q̂<sub>&alpha;</sub>): 16.0%</span>
                    </div>
                    <div id="sim-bars-container" class="sim-bars-container"></div>
                </div>
            </div>

            <div class="sim-metrics-panel">
                <div class="sim-metric-cell">
                    <div class="metric-title">Helmholtz Free Energy E(x; T)</div>
                    <div class="metric-value" id="sim-val-energy">-5.21</div>
                    <div class="metric-sub" id="sim-sub-energy">E &le; &tau; (In-Distribution)</div>
                </div>
                <div class="sim-metric-cell">
                    <div class="metric-title">Normalized Entropy H&#772;(&pi;)</div>
                    <div class="metric-value" id="sim-val-entropy">0.038</div>
                    <div class="metric-sub" id="sim-sub-entropy">Low Ambiguity (&lt; 0.82)</div>
                </div>
                <div class="sim-metric-cell">
                    <div class="metric-title">Prediction Set |C(x)|</div>
                    <div class="metric-value" id="sim-val-setsize">1 Species</div>
                    <div class="metric-sub" id="sim-sub-setsize">Singleton Set</div>
                </div>
                <div class="sim-metric-cell">
                    <div class="metric-title">Margin M(x) = &pi;<sub>(1)</sub> - &pi;<sub>(2)</sub></div>
                    <div class="metric-value" id="sim-val-margin">0.996</div>
                    <div class="metric-sub">Separation Margin</div>
                </div>
            </div>

            <div class="sim-thermo-identity-bar" id="sim-thermo-bar">
                <span class="identity-badge">Thermodynamic Identity:</span>
                <span id="sim-thermo-equation">E(x; T) = U(x) - T &middot; H(&pi;) &rarr; -5.21 = -5.19 - (1.0 &times; 0.02)</span>
            </div>

            <div class="sim-decision-banner" id="sim-decision-banner">
                <div class="banner-icon" id="sim-decision-icon">&#10003;</div>
                <div class="banner-body">
                    <div class="banner-title" id="sim-decision-title">Zero-Allocation Fast-Path Inference</div>
                    <div class="banner-desc" id="sim-decision-desc">
                        Singleton set satisfies 95% conformal coverage. Direct serving within 15 ms SLA.
                    </div>
                </div>
            </div>
        </div>
    `;

    const tempSlider = document.getElementById("sim-temp-input");
    const alphaSlider = document.getElementById("sim-alpha-input");
    const tauSlider = document.getElementById("sim-tau-input");

    const tempDisplay = document.getElementById("sim-temp-display");
    const alphaDisplay = document.getElementById("sim-alpha-display");
    const tauDisplay = document.getElementById("sim-tau-display");

    const logitsContainer = document.getElementById("sim-logits-inputs");
    const barsContainer = document.getElementById("sim-bars-container");
    const presetDesc = document.getElementById("sim-preset-desc");
    const presetBtns = mountPoint.querySelectorAll(".sim-preset-btn");

    function renderLogitInputs() {
        logitsContainer.innerHTML = species.map(function (sp, idx) {
            return `
                <div class="sim-logit-row">
                    <div class="logit-species-label">
                        <span class="sp-dot" style="background-color: ${sp.color};"></span>
                        <span class="sp-name"><em>${sp.name}</em></span>
                    </div>
                    <div class="logit-slider-wrap">
                        <input type="range" class="sim-logit-slider" data-index="${idx}" min="-8.0" max="8.0" step="0.1" value="${currentLogits[idx]}">
                        <span class="logit-num" id="sim-logit-val-${idx}">${currentLogits[idx].toFixed(1)}</span>
                    </div>
                </div>
            `;
        }).join("");

        const sliders = logitsContainer.querySelectorAll(".sim-logit-slider");
        sliders.forEach(function (slider) {
            slider.addEventListener("input", function (e) {
                const idx = parseInt(e.target.getAttribute("data-index"), 10);
                currentLogits[idx] = parseFloat(e.target.value);
                const valSpan = document.getElementById(`sim-logit-val-${idx}`);
                if (valSpan) valSpan.textContent = currentLogits[idx].toFixed(1);
                presetBtns.forEach(function (b) { b.classList.remove("active"); });
                updateSimulation();
            });
        });
    }

    function updateSimulation() {
        currentTemp = parseFloat(tempSlider.value) || 1.0;
        currentAlpha = parseFloat(alphaSlider.value) || 0.05;
        currentTau = parseFloat(tauSlider.value) || -2.28;

        tempDisplay.textContent = currentTemp.toFixed(1);
        alphaDisplay.textContent = currentAlpha.toFixed(2);
        tauDisplay.textContent = currentTau.toFixed(2);

        // Finite-sample inflated quantile
        // At alpha=0.05, calibrated q_hat = 0.8405
        const q_hat = Math.min(0.995, Math.max(0.60, 1.0 - currentAlpha * 3.19));
        const nonConfThresh = 1.0 - q_hat;

        const threshLegend = document.getElementById("sim-threshold-legend");
        if (threshLegend) {
            threshLegend.textContent = `Threshold (1 - q̂): ${(nonConfThresh * 100).toFixed(1)}% (α = ${currentAlpha.toFixed(2)})`;
        }

        // Scaled logits z / T
        const z_scaled = currentLogits.map(function (z) { return z / currentTemp; });
        const max_z = Math.max(...z_scaled);
        const exp_diff = z_scaled.map(function (v) { return Math.exp(v - max_z); });
        const sum_exp_diff = exp_diff.reduce(function (a, b) { return a + b; }, 0);
        const log_sum_exp = max_z + Math.log(sum_exp_diff);

        // Softmax probabilities
        const probs = exp_diff.map(function (e) { return e / sum_exp_diff; });

        // Helmholtz Free Energy: E(x; T) = -T * log_sum_exp
        const freeEnergy = -currentTemp * log_sum_exp;

        // Internal Energy: U(x) = sum(pi_k * (-z_k))
        const internalEnergy = probs.reduce(function (acc, p, i) {
            return acc + p * (-currentLogits[i]);
        }, 0);

        // Shannon Entropy: H(pi) = -sum(pi_k * ln(pi_k))
        const shannonEntropy = probs.reduce(function (acc, p) {
            return acc + (p > 1e-12 ? -p * Math.log(p) : 0);
        }, 0);

        // Normalized Shannon Entropy: H_norm = H(pi) / ln(5)
        const K = probs.length;
        const normEntropy = shannonEntropy / Math.log(K);

        // Margin: M(x) = pi_(1) - pi_(2)
        const sortedProbs = [...probs].sort(function (a, b) { return b - a; });
        const margin = sortedProbs[0] - sortedProbs[1];

        // Conformal prediction set C(x)
        const admittedIndices = [];
        probs.forEach(function (p, idx) {
            if (p >= nonConfThresh) {
                admittedIndices.push(idx);
            }
        });

        // Render Bars
        barsContainer.innerHTML = species.map(function (sp, idx) {
            const prob = probs[idx];
            const probPct = (prob * 100).toFixed(1);
            const isAdmitted = prob >= nonConfThresh;
            const barClass = isAdmitted ? "admitted" : "excluded";

            return `
                <div class="sim-bar-row">
                    <div class="sim-bar-header">
                        <span class="sp-name"><em>${sp.name}</em></span>
                        <span class="prob-num ${barClass}">${probPct}% ${isAdmitted ? "✓ Admitted" : ""}</span>
                    </div>
                    <div class="sim-bar-track">
                        <div class="sim-bar-fill ${barClass}" style="width: ${Math.min(100, prob * 100)}%; background-color: ${sp.color};"></div>
                        <div class="sim-bar-cutoff-line" style="left: ${Math.min(100, nonConfThresh * 100)}%;"></div>
                    </div>
                </div>
            `;
        }).join("");

        // Update Epistemic Metric Readouts
        const valEnergy = document.getElementById("sim-val-energy");
        const subEnergy = document.getElementById("sim-sub-energy");
        if (valEnergy) valEnergy.textContent = freeEnergy.toFixed(2);
        if (subEnergy) {
            if (freeEnergy > currentTau) {
                subEnergy.innerHTML = `<span style="color: var(--tv-accent);">⚠️ E > τ (${freeEnergy.toFixed(2)} > ${currentTau.toFixed(2)}) OOD!</span>`;
            } else {
                subEnergy.innerHTML = `<span style="color: var(--tv-accent-green);">✓ E ≤ τ (${freeEnergy.toFixed(2)} ≤ ${currentTau.toFixed(2)}) In-Dist</span>`;
            }
        }

        const valEntropy = document.getElementById("sim-val-entropy");
        const subEntropy = document.getElementById("sim-sub-entropy");
        if (valEntropy) valEntropy.textContent = normEntropy.toFixed(3);
        if (subEntropy) {
            if (normEntropy >= 0.82) {
                subEntropy.innerHTML = `<span style="color: #ffa726;">⚠️ Diffuse Clutter (H̄ ≥ 0.82)</span>`;
            } else {
                subEntropy.innerHTML = `<span style="color: var(--tv-accent-green);">✓ Low Ambiguity (H̄ < 0.82)</span>`;
            }
        }

        const valSetSize = document.getElementById("sim-val-setsize");
        const subSetSize = document.getElementById("sim-sub-setsize");
        if (valSetSize) valSetSize.textContent = `${admittedIndices.length} Taxa`;
        if (subSetSize) {
            if (admittedIndices.length === 0) {
                subSetSize.textContent = "Empty Set |C| = 0";
            } else if (admittedIndices.length === 1) {
                subSetSize.textContent = "Singleton Set |C| = 1";
            } else {
                subSetSize.textContent = `Multi-Label Set |C| = ${admittedIndices.length}`;
            }
        }

        const valMargin = document.getElementById("sim-val-margin");
        if (valMargin) valMargin.textContent = margin.toFixed(3);

        // Thermodynamic Identity Verification Bar
        const thermoEquation = document.getElementById("sim-thermo-equation");
        if (thermoEquation) {
            const rhs = internalEnergy - currentTemp * shannonEntropy;
            thermoEquation.innerHTML = `E(x; T) = U(x) - T &middot; H(&pi;) &nbsp;|&nbsp; <strong>${freeEnergy.toFixed(3)}</strong> = ${internalEnergy.toFixed(3)} - (${currentTemp.toFixed(1)} &times; ${shannonEntropy.toFixed(3)}) &nbsp;[&Delta; = ${Math.abs(freeEnergy - rhs).toFixed(4)}]`;
        }

        // Decision Banner Evaluation
        const banner = document.getElementById("sim-decision-banner");
        const icon = document.getElementById("sim-decision-icon");
        const title = document.getElementById("sim-decision-title");
        const desc = document.getElementById("sim-decision-desc");

        banner.className = "sim-decision-banner";

        if (freeEnergy > currentTau) {
            banner.classList.add("route-ood");
            icon.textContent = "⚠️";
            title.textContent = "Route: Semantic OOD Human Triage Queue";
            desc.innerHTML = `Helmholtz Free Energy <strong>E(x; T) = ${freeEnergy.toFixed(2)}</strong> exceeds cutoff <strong>&tau;<sub>OOD</sub> = ${currentTau.toFixed(2)}</strong>. Indicates novel non-biological morphology or sensor distortion. Shift-invariant softmax bypassed; diverted to expert review.`;
        } else if (admittedIndices.length === 0) {
            banner.classList.add("route-empty");
            icon.textContent = "🛑";
            title.textContent = "Route: Empty Prediction Set Triage (|C| = 0)";
            desc.innerHTML = `No candidate taxon satisfied the conformal threshold <strong>1 - q̂<sub>&alpha;</sub> = ${(nonConfThresh * 100).toFixed(1)}%</strong>. Sample exceeds epistemic confidence bounds and is routed to taxonomist review.`;
        } else if (normEntropy >= 0.82) {
            banner.classList.add("route-clutter");
            icon.textContent = "🍃";
            title.textContent = "Route: Empty Frame / Clutter Rejection";
            desc.innerHTML = `Normalized Shannon Entropy <strong>H&#772;(&pi;) = ${normEntropy.toFixed(3)} &ge; 0.82</strong> indicates near-uniform diffuse uncertainty across classes. Filtered out as background foliage clutter or empty camera-trap false trigger.`;
        } else if (admittedIndices.length === 1) {
            banner.classList.add("route-fast");
            icon.textContent = "⚡";
            title.textContent = "Route: Zero-Allocation Fast-Path Inference";
            const candidate = species[admittedIndices[0]].name;
            desc.innerHTML = `Singleton prediction set admitted: <em>${candidate}</em> (&pi;&#770; = ${(probs[admittedIndices[0]] * 100).toFixed(1)}%). Mathematical ${( (1 - currentAlpha) * 100).toFixed(0)}% coverage guarantee verified. Direct ONNX Runtime serving within sub-15ms SLA.`;
        } else if (admittedIndices.length <= 3) {
            banner.classList.add("route-rag");
            icon.textContent = "🔍";
            title.textContent = "Route: Sister-Taxa Vision-RAG Differential Engine";
            const candList = admittedIndices.map(function (i) { return `<em>${species[i].name}</em>`; }).join(", ");
            desc.innerHTML = `Conformal set contains |C| = ${admittedIndices.length} ambiguous sister taxa (${candList}). Triggers lazy YOLO-World child crop extraction, single-stage filtered HNSW retrieval, and VLM morphological differential card.`;
        } else {
            banner.classList.add("route-escalation");
            icon.textContent = "📋";
            title.textContent = "Route: High-Ambiguity Taxonomic Escalation";
            desc.innerHTML = `Broad prediction set spanning |C| = ${admittedIndices.length} taxa with narrow margin M(x) = ${margin.toFixed(3)}. Exceeds automated resolution bounds; escalated to specialized botanical/entomological review queue.`;
        }
    }

    // Bind Presets
    presetBtns.forEach(function (btn) {
        btn.addEventListener("click", function () {
            presetBtns.forEach(function (b) { b.classList.remove("active"); });
            btn.classList.add("active");
            const presetId = btn.getAttribute("data-preset");
            const p = presets[presetId];
            if (!p) return;

            currentLogits = [...p.logits];
            currentTemp = p.temp;
            currentAlpha = p.alpha;
            currentTau = p.tau;

            tempSlider.value = currentTemp;
            alphaSlider.value = currentAlpha;
            tauSlider.value = currentTau;

            presetDesc.textContent = p.description;

            renderLogitInputs();
            updateSimulation();
        });
    });

    tempSlider.addEventListener("input", updateSimulation);
    alphaSlider.addEventListener("input", updateSimulation);
    tauSlider.addEventListener("input", updateSimulation);

    renderLogitInputs();
    updateSimulation();
}

/**
 * Interactive Pareto Frontier Explorer & Benchmark Matrix (Part 4)
 */
function initParetoExplorer() {
    const mountPoint = document.getElementById("taxon-pareto-explorer");
    if (!mountPoint) return;
    if (mountPoint.dataset.initialized === "true") return;
    mountPoint.dataset.initialized = "true";

    const defaultData = [
        {
            extractor: "dinov3_vits14",
            displayName: "DINOv3 ViT-S/14",
            architecture: "Dense Self-Supervised ViT",
            parameters_m: 22.0,
            feature_dim: 384,
            p50_latency_ms: 9.5,
            p95_latency_ms: 13.5,
            throughput_fps: 105.0,
            vram_peak_mb: 850.0,
            cost_per_million_usd: 2.60,
            top1_acc: 93.8,
            macro_pr_auc: 0.952,
            sla_passed: true,
            badge: "Champion",
            color: "#00e676",
            family: "vit",
            profile: "Dense Morphology & Register Tokens"
        },
        {
            extractor: "bioclip-2",
            displayName: "BioCLIP-2 ViT",
            architecture: "Vision-Language (Tree of Life)",
            parameters_m: 86.0,
            feature_dim: 512,
            p50_latency_ms: 8.5,
            p95_latency_ms: 12.0,
            throughput_fps: 85.0,
            vram_peak_mb: 1200.0,
            cost_per_million_usd: 3.50,
            top1_acc: 92.4,
            macro_pr_auc: 0.941,
            sla_passed: true,
            badge: "Domain Priors",
            color: "#00e5ff",
            family: "vit",
            profile: "Domain Biological Tree Prior"
        },
        {
            extractor: "dinov2_vits14",
            displayName: "DINOv2 ViT-S/14",
            architecture: "Self-Supervised Vision Transformer",
            parameters_m: 22.0,
            feature_dim: 384,
            p50_latency_ms: 9.8,
            p95_latency_ms: 14.0,
            throughput_fps: 100.0,
            vram_peak_mb: 850.0,
            cost_per_million_usd: 2.70,
            top1_acc: 89.4,
            macro_pr_auc: 0.908,
            sla_passed: true,
            badge: "Baseline ViT",
            color: "#b388ff",
            family: "vit",
            profile: "Foundation Visual Baseline"
        },
        {
            extractor: "mobilenetv4_conv_small",
            displayName: "MobileNetV4 Small",
            architecture: "Edge Universal CNN",
            parameters_m: 3.8,
            feature_dim: 960,
            p50_latency_ms: 2.1,
            p95_latency_ms: 3.5,
            throughput_fps: 320.0,
            vram_peak_mb: 150.0,
            cost_per_million_usd: 0.50,
            top1_acc: 78.5,
            macro_pr_auc: 0.798,
            sla_passed: true,
            badge: "Edge Fast-Path",
            color: "#ffab00",
            family: "cnn",
            profile: "Ultra-Low Latency Edge CPU"
        },
        {
            extractor: "efficientnet_b0",
            displayName: "EfficientNet-B0",
            architecture: "Compound Scaled CNN",
            parameters_m: 5.3,
            feature_dim: 1280,
            p50_latency_ms: 4.2,
            p95_latency_ms: 6.8,
            throughput_fps: 160.0,
            vram_peak_mb: 300.0,
            cost_per_million_usd: 1.10,
            top1_acc: 81.2,
            macro_pr_auc: 0.825,
            sla_passed: true,
            badge: "Balanced Edge",
            color: "#ff7043",
            family: "cnn",
            profile: "Balanced Edge Baseline"
        }
    ];

    let currentFamily = "all";
    let sortColumn = "top1_acc";
    let sortAsc = false;
    let selectedModel = defaultData[0];

    mountPoint.innerHTML = `
        <div class="pareto-explorer">
            <div class="pareto-header-row">
                <div class="pareto-chart-title">
                    <span>Multi-Objective Pareto Frontier (p95 Latency vs. Accuracy)</span>
                    <span class="pareto-sla-badge">15 ms Production SLA Gate</span>
                </div>
                <div class="pareto-filter-group">
                    <button class="pareto-filter-btn active" data-family="all">All Models</button>
                    <button class="pareto-filter-btn" data-family="vit">ViT Transformers</button>
                    <button class="pareto-filter-btn" data-family="cnn">Edge CNNs</button>
                </div>
            </div>

            <div class="pareto-chart-container" id="pareto-chart-svg"></div>

            <div class="pareto-table-wrap">
                <table class="pareto-table" id="pareto-matrix-table">
                    <thead>
                        <tr>
                            <th data-sort="displayName">Backbone</th>
                            <th data-sort="architecture">Architecture</th>
                            <th data-sort="top1_acc">Top-1 Accuracy</th>
                            <th data-sort="macro_pr_auc">Macro PR-AUC</th>
                            <th data-sort="p95_latency_ms">p95 Latency</th>
                            <th data-sort="throughput_fps">Throughput</th>
                            <th data-sort="parameters_m">Params</th>
                            <th data-sort="cost_per_million_usd">Cost / 1M Reqs</th>
                            <th>SLA Gate</th>
                        </tr>
                    </thead>
                    <tbody id="pareto-table-body"></tbody>
                </table>
            </div>

            <div class="pareto-detail-card" id="pareto-detail-card"></div>
        </div>
    `;

    const chartContainer = document.getElementById("pareto-chart-svg");
    const tableBody = document.getElementById("pareto-table-body");
    const detailCard = document.getElementById("pareto-detail-card");
    const filterBtns = mountPoint.querySelectorAll(".pareto-filter-btn");
    const tableHeaders = mountPoint.querySelectorAll("#pareto-matrix-table th[data-sort]");

    function renderChart() {
        const width = 720;
        const height = 340;
        const padding = { top: 30, right: 40, bottom: 50, left: 60 };
        const plotW = width - padding.left - padding.right;
        const plotH = height - padding.top - padding.bottom;

        // X scale: p95 latency (0 to 18 ms)
        const minX = 0;
        const maxX = 18;
        // Y scale: accuracy (70% to 100%)
        const minY = 70;
        const maxY = 100;

        function xScale(val) {
            return padding.left + ((val - minX) / (maxX - minX)) * plotW;
        }

        function yScale(val) {
            return padding.top + plotH - ((val - minY) / (maxY - minY)) * plotH;
        }

        const slaX = xScale(15.0);

        // Filtered items
        const visibleData = defaultData.filter(function (d) {
            return currentFamily === "all" || d.family === currentFamily;
        });

        let svgHtml = `
            <svg viewBox="0 0 ${width} ${height}" class="pareto-svg" preserveAspectRatio="xMidYMid meet">
                <defs>
                    <linearGradient id="sla-pass-grad" x1="0" y1="0" x2="1" y2="0">
                        <stop offset="0%" stop-color="#00e676" stop-opacity="0.08" />
                        <stop offset="100%" stop-color="#00e676" stop-opacity="0.02" />
                    </linearGradient>
                    <linearGradient id="sla-breach-grad" x1="0" y1="0" x2="1" y2="0">
                        <stop offset="0%" stop-color="#ff5722" stop-opacity="0.04" />
                        <stop offset="100%" stop-color="#ff5722" stop-opacity="0.12" />
                    </linearGradient>
                </defs>

                <!-- SLA zones -->
                <rect x="${padding.left}" y="${padding.top}" width="${slaX - padding.left}" height="${plotH}" fill="url(#sla-pass-grad)" />
                <rect x="${slaX}" y="${padding.top}" width="${padding.left + plotW - slaX}" height="${plotH}" fill="url(#sla-breach-grad)" />

                <!-- Grid lines -->
                ${[75, 80, 85, 90, 95].map(function (yVal) {
                    const y = yScale(yVal);
                    return `
                        <line x1="${padding.left}" y1="${y}" x2="${padding.left + plotW}" y2="${y}" stroke="var(--tv-border-subtle)" stroke-dasharray="3,3" />
                        <text x="${padding.left - 8}" y="${y + 4}" fill="currentColor" opacity="0.6" font-size="10" text-anchor="end">${yVal}%</text>
                    `;
                }).join("")}

                ${[3, 6, 9, 12, 15].map(function (xVal) {
                    const x = xScale(xVal);
                    return `
                        <line x1="${x}" y1="${padding.top}" x2="${x}" y2="${padding.top + plotH}" stroke="var(--tv-border-subtle)" stroke-dasharray="3,3" />
                        <text x="${x}" y="${padding.top + plotH + 18}" fill="currentColor" opacity="0.6" font-size="10" text-anchor="middle">${xVal} ms</text>
                    `;
                }).join("")}

                <!-- 15ms SLA Gate Line -->
                <line x1="${slaX}" y1="${padding.top}" x2="${slaX}" y2="${padding.top + plotH}" stroke="#ff5722" stroke-width="2" stroke-dasharray="5,4" />
                <text x="${slaX - 6}" y="${padding.top + 14}" fill="#ff5722" font-size="10" font-weight="700" text-anchor="end">15 ms Production Gate</text>

                <!-- Axis Labels -->
                <text x="${padding.left + plotW / 2}" y="${height - 10}" fill="currentColor" opacity="0.8" font-size="11" font-weight="600" text-anchor="middle">p95 Inference Latency (ms) &rarr;</text>
                <text transform="rotate(-90)" x="${-(padding.top + plotH / 2)}" y="16" fill="currentColor" opacity="0.8" font-size="11" font-weight="600" text-anchor="middle">&larr; Top-1 Accuracy (%)</text>

                <!-- Data Points -->
                ${visibleData.map(function (d) {
                    const cx = xScale(d.p95_latency_ms);
                    const cy = yScale(d.top1_acc);
                    const isSelected = selectedModel && selectedModel.extractor === d.extractor;
                    const r = isSelected ? 8 : 6;

                    return `
                        <g class="pareto-point-group" data-extractor="${d.extractor}" style="cursor: pointer;">
                            ${isSelected ? `<circle cx="${cx}" cy="${cy}" r="14" fill="${d.color}" opacity="0.3" class="pulse-ring" />` : ""}
                            <circle cx="${cx}" cy="${cy}" r="${r}" fill="${d.color}" stroke="#ffffff" stroke-width="${isSelected ? 2.5 : 1.5}" />
                            <text x="${cx}" y="${cy - 12}" fill="${d.color}" font-size="10" font-weight="700" text-anchor="middle">${d.displayName}</text>
                        </g>
                    `;
                }).join("")}
            </svg>
        `;

        chartContainer.innerHTML = svgHtml;

        const points = chartContainer.querySelectorAll(".pareto-point-group");
        points.forEach(function (pt) {
            pt.addEventListener("click", function () {
                const ext = pt.getAttribute("data-extractor");
                selectedModel = defaultData.find(function (m) { return m.extractor === ext; }) || selectedModel;
                renderChart();
                renderTable();
                renderDetail();
            });
        });
    }

    function renderTable() {
        const visibleData = defaultData.filter(function (d) {
            return currentFamily === "all" || d.family === currentFamily;
        });

        visibleData.sort(function (a, b) {
            let valA = a[sortColumn];
            let valB = b[sortColumn];
            if (typeof valA === "string") valA = valA.toLowerCase();
            if (typeof valB === "string") valB = valB.toLowerCase();
            if (valA < valB) return sortAsc ? -1 : 1;
            if (valA > valB) return sortAsc ? 1 : -1;
            return 0;
        });

        tableBody.innerHTML = visibleData.map(function (d) {
            const isSelected = selectedModel && selectedModel.extractor === d.extractor;
            const selClass = isSelected ? "selected-row" : "";
            return `
                <tr class="${selClass}" data-extractor="${d.extractor}">
                    <td>
                        <span class="sp-dot" style="background-color: ${d.color}; display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 6px;"></span>
                        <strong>${d.displayName}</strong>
                    </td>
                    <td>${d.architecture}</td>
                    <td><strong>${d.top1_acc.toFixed(1)}%</strong></td>
                    <td>${d.macro_pr_auc.toFixed(3)}</td>
                    <td><span class="latency-pill">${d.p95_latency_ms.toFixed(1)} ms</span></td>
                    <td>${d.throughput_fps.toFixed(0)} FPS</td>
                    <td>${d.parameters_m.toFixed(1)}M</td>
                    <td>$${d.cost_per_million_usd.toFixed(2)}</td>
                    <td><span class="sla-pass-badge">${d.badge}</span></td>
                </tr>
            `;
        }).join("");

        const rows = tableBody.querySelectorAll("tr");
        rows.forEach(function (row) {
            row.addEventListener("click", function () {
                const ext = row.getAttribute("data-extractor");
                selectedModel = defaultData.find(function (m) { return m.extractor === ext; }) || selectedModel;
                renderChart();
                renderTable();
                renderDetail();
            });
        });
    }

    function renderDetail() {
        if (!selectedModel) return;
        const m = selectedModel;
        detailCard.innerHTML = `
            <div class="pareto-card-header">
                <div>
                    <div class="card-title">${m.displayName}</div>
                    <div class="card-subtitle">${m.architecture} &bull; ${m.profile}</div>
                </div>
                <div class="card-metric-badge" style="border-color: ${m.color}; color: ${m.color};">
                    ${m.badge}
                </div>
            </div>
            <div class="pareto-card-grid">
                <div class="card-stat">
                    <span class="title">Top-1 Accuracy</span>
                    <span class="val" style="color: ${m.color};">${m.top1_acc.toFixed(1)}%</span>
                </div>
                <div class="card-stat">
                    <span class="title">p95 Latency</span>
                    <span class="val">${m.p95_latency_ms.toFixed(1)} ms</span>
                </div>
                <div class="card-stat">
                    <span class="title">Throughput</span>
                    <span class="val">${m.throughput_fps.toFixed(0)} FPS</span>
                </div>
                <div class="card-stat">
                    <span class="title">Serving Cost</span>
                    <span class="val">$${m.cost_per_million_usd.toFixed(2)} / 1M</span>
                </div>
                <div class="card-stat">
                    <span class="title">Parameters</span>
                    <span class="val">${m.parameters_m.toFixed(1)}M</span>
                </div>
                <div class="card-stat">
                    <span class="title">Peak VRAM</span>
                    <span class="val">${m.vram_peak_mb.toFixed(0)} MB</span>
                </div>
            </div>
        `;
    }

    filterBtns.forEach(function (btn) {
        btn.addEventListener("click", function () {
            filterBtns.forEach(function (b) { b.classList.remove("active"); });
            btn.classList.add("active");
            currentFamily = btn.getAttribute("data-family") || "all";
            renderChart();
            renderTable();
        });
    });

    tableHeaders.forEach(function (th) {
        th.addEventListener("click", function () {
            const col = th.getAttribute("data-sort");
            if (sortColumn === col) {
                sortAsc = !sortAsc;
            } else {
                sortColumn = col;
                sortAsc = false;
            }
            renderTable();
        });
    });

    // Attempt to fetch from pareto_results.json if hosted
    fetch("assets/pareto_results.json")
        .then(function (res) { return res.json(); })
        .then(function (jsonData) {
            if (Array.isArray(jsonData) && jsonData.length > 0) {
                // Merge if available
                jsonData.forEach(function (item) {
                    const match = defaultData.find(function (d) { return d.extractor === item.extractor; });
                    if (match) {
                        if (item.p95_latency_ms) match.p95_latency_ms = item.p95_latency_ms;
                        if (item.throughput_fps) match.throughput_fps = item.throughput_fps;
                        if (item.cost_per_million_usd) match.cost_per_million_usd = item.cost_per_million_usd;
                    }
                });
                renderChart();
                renderTable();
                renderDetail();
            }
        })
        .catch(function () {
            // Keep defaultData fallback
        });

    renderChart();
    renderTable();
    renderDetail();
}
