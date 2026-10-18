/**
 * TaxonVision-MLOps Interactive Widgets & Visualizer Controller
 */

document.addEventListener("DOMContentLoaded", function () {
    initFullscreenHandlers();
    initConformalSimulators();
    initCodebaseExplorer();
});

// Support dynamic navigation in Material for MkDocs / Zensical
if (typeof document$ !== "undefined") {
    document$.subscribe(function () {
        initFullscreenHandlers();
        initConformalSimulators();
        initCodebaseExplorer();
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
