/**
 * TaxonVision-MLOps Interactive 8-Stage Epistemic Diamond DVC DAG Explorer
 */
(function () {
/**
 * Interactive 8-Stage Epistemic Diamond DVC DAG & Live Metrics Inspector (Part 2)
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
            metrics: "N/A (Metadata manifest stage: 2,000 raw observations)",
            algorithm: [
                "Parses Level 1 perimeter taxa catalog from config/taxa_catalog.yaml.",
                "Executes paginated HTTP queries to iNaturalist v1 REST API (per_page=200).",
                "Merges verified human-triage corrections from data/feedback/verified_feedback.parquet when available.",
                "Serialises metadata into Snappy-compressed columnar Parquet format (< 500 KB on disk)."
            ],
            triggers: "Triggered when config/taxa_catalog.yaml, inat_client.py, ingest.py, or ingestion.* parameters change.",
            downstreamRadius: "Invalidates: Stage 2 (audit_license), Stage 3 (extract_features), Stage 3.5 (data_drift), Stage 4 (train), Stage 5 (export), Stage 6 (calibrate), and Stage 7 (eval)."
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
            metrics: "reports/license_audit.json: 1,646 / 2,000 Verified (82.3%), 354 Violations Screened, Status: PASSED",
            algorithm: [
                "Reads raw_manifest.parquet into zero-copy Apache Arrow table.",
                "Evaluates each observation's license_code against the OpenLicense StrEnum.",
                "Rejects unlicensed images, proprietary copyrights, and non-derivative media (354 records purged).",
                "Retains photographer attribution strings and writes 1,646 verified records to dataset_manifest.parquet.",
                "Emits reports/license_audit.json tracking compliance percentage."
            ],
            triggers: "Triggered when raw_manifest.parquet, license_filter.py, domain/license.py, or audit_license.py changes.",
            downstreamRadius: "Invalidates: Stage 3 (extract_features) and all downstream drift, training, calibration, and evaluation stages."
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
                "data/embeddings/train_embeddings.pt (1,153 x 384)",
                "data/embeddings/val_embeddings.pt (164 x 384)",
                "data/embeddings/cal_embeddings.pt (164 x 384)",
                "data/embeddings/test_embeddings.pt (164 x 384)"
            ],
            metrics: "1,645 streamed observations partitioned into 4 non-overlapping stratified tensors (384-D)",
            algorithm: [
                "Instantiates frozen foundation backbone (vit_small_patch16_dinov3) via BackboneFactory.",
                "Streams raw observation JPEG bytes on demand from s3://inaturalist-open-data with local LRU caching.",
                "Normalises images (ImageNet mean & std) and computes 384-dimensional representation vectors.",
                "Executes deterministic 4-way stratified hash split (70% train = 1,153, 10% val = 164, 10% cal = 164, 10% test = 164, seed=42).",
                "Persists PyTorch .pt tensors to DagsHub S3 origin and in-cluster SeaweedFS cache."
            ],
            triggers: "Triggered when model.extractor, model.image_size, split ratios, dataset_manifest.parquet, or backbone factory code changes.",
            downstreamRadius: "Invalidates: Stage 3.5 (data_drift), Stage 4 (train), Stage 5 (export), Stage 6 (calibrate), and Stage 7 (eval). Heavy compute boundary."
        },
        {
            id: "data_drift",
            number: "3.5",
            title: "3.5. Training Data Drift Gate (Two-Tier Representation Audit)",
            dvcStage: "data_drift",
            cmd: "pixi run -e dev python -m taxon_vision.pipelines.data_drift --reference-embeddings data/embeddings/reference_train_embeddings.pt --current-embeddings data/embeddings/train_embeddings.pt --drift-config config/drift_config.yaml --out reports/data_drift.json",
            deps: [
                "config/drift_config.yaml",
                "data/embeddings/reference_train_embeddings.pt",
                "data/embeddings/train_embeddings.pt",
                "src/taxon_vision/monitoring/drift.py",
                "src/taxon_vision/pipelines/data_drift.py"
            ],
            params: [
                "drift.mmd_alpha = 0.01",
                "drift.num_permutations = 500",
                "drift.fail_on_drift = false"
            ],
            outs: [
                "reports/data_drift.json (metrics)"
            ],
            metrics: "reports/data_drift.json: W1: 0.01075 (<= tau_w1 0.12812), MMD^2: -0.00031 (p = 0.7413), Drift: FALSE",
            algorithm: [
                "Loads frozen baseline reference_train_embeddings.pt (N_ref=1,182) and candidate train_embeddings.pt (N_cur=1,153).",
                "Computes Tier-1 1D Wasserstein-1 distance over L2 embedding norms against calibrated threshold tau_w1 = 0.12812.",
                "Computes Tier-2 unbiased squared Maximum Mean Discrepancy (MMD_u^2) with multi-bandwidth RBF kernel and 500 permutations.",
                "Emits reports/data_drift.json and logs representation shift telemetry before training begins."
            ],
            triggers: "Triggered when train_embeddings.pt, reference_train_embeddings.pt, config/drift_config.yaml, or monitoring/drift.py changes.",
            downstreamRadius: "Gates Stage 4 (train) upon representation covariate shift."
        },
        {
            id: "train",
            number: "4",
            title: "4. Linear Probe Training (Class-Balanced Loss)",
            dvcStage: "train",
            cmd: "pixi run -e dev python -m taxon_vision.pipelines.train --extractor dinov3 --epochs 30 --batch-size 32 --lr 0.001 --train-embeddings data/embeddings/train_embeddings.pt --val-embeddings data/embeddings/val_embeddings.pt --checkpoint-path models/checkpoints/head.pt --metrics-out reports/train_metrics.json",
            deps: [
                "data/embeddings/train_embeddings.pt",
                "data/embeddings/val_embeddings.pt",
                "reports/data_drift.json",
                "src/taxon_vision/models/loss.py",
                "src/taxon_vision/models/training/",
                "src/taxon_vision/pipelines/train.py"
            ],
            params: [
                "model.extractor = dinov3",
                "training.batch_size = 32",
                "training.cb_beta = 0.9999",
                "training.epochs = 30",
                "training.learning_rate = 0.001",
                "training.tune = false",
                "training.weight_decay = 0.0001"
            ],
            outs: [
                "models/checkpoints/head.pt (pushed to origin/DagsHub & SeaweedFS)"
            ],
            metrics: "reports/train_metrics.json: Train Loss: 0.2597, Val Loss: 0.2683, Val Accuracy: 93.29%, Duration: 3.61s (30 Epochs)",
            algorithm: [
                "Loads pre-extracted train_embeddings.pt (N=1,153) and val_embeddings.pt (N=164) via models.training.dataset.",
                "Calculates Class-Balanced effective number weights: E_n = (1 - beta) / (1 - beta^n) with beta=0.9999.",
                "Executes 30 mini-batch SGD epochs via models.training.loops and models.training.embeddings with AdamW.",
                "Tracks loss, Macro PR-AUC, and validation accuracy per epoch via RAII mlflow_run_scope().",
                "Exports trained cosine classification head to models/checkpoints/head.pt."
            ],
            triggers: "Triggered when training hyperparameters (lr, epochs, cb_beta), training embeddings, data_drift.json, or loss code changes.",
            downstreamRadius: "Invalidates: Stage 5 (export), Stage 6 (calibrate), and Stage 7 (eval). Does NOT invalidate Stage 3 feature extraction!"
        },
        {
            id: "export",
            number: "5",
            title: "5. Graph Fusion & INT8 PTQ (ONNX Runtime)",
            dvcStage: "export",
            cmd: "pixi run -e dev python -m taxon_vision.pipelines.export_onnx --extractor dinov3 --image-size 224 --checkpoint models/checkpoints/head.pt --quantize-int8 --quantize-mode dynamic --quantized-output models/checkpoints/model_int8.onnx --export-head",
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
                "models/checkpoints/model_int8.onnx (fused backbone + head)",
                "models/checkpoints/head.onnx & head_int8.onnx (dedicated linear probe head)"
            ],
            metrics: "N/A (Optimised INT8 runtime graph pushed to DVC remote)",
            algorithm: [
                "Instantiates full computational graph fusing the DINOv3 ViT-S/16 backbone and trained cosine head.",
                "Exports dual-output PyTorch model (logits [B, 10] + embeddings [B, 384]) to ONNX Opset 17 with dynamic batch axes.",
                "Exports dedicated standalone linear probe head graph (head.onnx and dynamic INT8 head_int8.onnx) for sub-millisecond (< 0.2 ms) scoring.",
                "Applies logit-preserving dynamic INT8 Post-Training Quantisation (protecting final logit projection nodes via find_logit_nodes).",
                "Validates numerical parity and exports models/checkpoints/model_int8.onnx."
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
            metrics: "reports/conformal_metrics.json: Empirical Cal Coverage: 96.77%, q_hat: 0.6337, tau_ood: -2.4029 (N_cal = 155 inliers)",
            algorithm: [
                "Runs ONNX Runtime inference over unseen cal_embeddings.pt (164 samples; 155 inliers retained after 95% TPR OOD gating).",
                "Calculates true-class non-conformity scores: s_i = 1 - f(x_i)_{y_i}.",
                "Computes finite-sample inflated quantile q_hat = 0.6337 at level ceil((n+1)(1-alpha))/n.",
                "Computes Helmholtz Free Energy scores E(x) = -T * log sum(exp(logits/T)).",
                "Establishes OOD decision threshold tau_ood = -2.4029 anchoring In-Distribution TPR to 95%.",
                "Writes calibrated parameters to config/conformal_config.yaml."
            ],
            triggers: "Triggered when cal_embeddings.pt, model_int8.onnx, conformal.alpha, or conformal.ood_tpr changes.",
            downstreamRadius: "Invalidates: Stage 7 (eval)."
        },
        {
            id: "eval",
            number: "7",
            title: "7. Pareto Evaluation & Four-Gate Promotion (Deployment Gate)",
            dvcStage: "eval",
            cmd: "pixi run -e dev python -m taxon_vision.pipelines.evaluate --onnx-model models/checkpoints/model_int8.onnx --embeddings data/embeddings/test_embeddings.pt --conformal-config config/conformal_config.yaml --out docs/assets/pareto_results.json --metrics-out reports/pareto_evaluation.json --latency-sla-ms 120.0",
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
                "evaluation.latency_sla_ms = 120.0"
            ],
            outs: [
                "docs/assets/pareto_results.json"
            ],
            metrics: "reports/pareto_evaluation.json: Test Accuracy: 92.07%, Test Coverage: 91.46% (Cal: 96.77%), Mean Set Size: 1.01, p95: 50.38 ms (20.9 FPS)",
            algorithm: [
                "Executes ONNX Runtime engine over holdout test_embeddings.pt (N=164).",
                "Forms prediction sets C(X) = {y : f(x)_y >= 1 - q_hat} (threshold 1 - 0.6337 = 0.3663).",
                "Verifies Gate 1 (Accuracy 92.07%, INT8 drop 1.22% <= 3.0%), Gate 2 (Test coverage 91.46% >= 90.0% finite-sample bound), Gate 3 (p95 latency 50.38 ms <= 120.0 ms SLA), and Gate 4 (Worst-class AP 92.55% >= 25.0%).",
                "Promotes qualifying candidate to @Production in MLflow Registry and writes reports/pareto_evaluation.json."
            ],
            triggers: "Triggered when test_embeddings.pt, model_int8.onnx, conformal_config.yaml, or latency_sla_ms changes.",
            downstreamRadius: "Terminal stage. Blocks @Production alias promotion upon statistical coverage failure or SLO breach."
        }
    ];

    let currentStage = stages[0];

    // Render skeleton
    mountPoint.innerHTML = `
        <div class="dvc-explorer">
            <div class="dvc-metrics-summary">
                <div class="dvc-metric-pill">
                    <span class="title">Stage 2: License Audit</span>
                    <span class="value">1,646 / 2,000</span>
                    <span class="subtext">82.3% Open Licensed (354 Violations Rejected)</span>
                </div>
                <div class="dvc-metric-pill">
                    <span class="title">Stage 4: Probe Training</span>
                    <span class="value">93.29% Val Acc</span>
                    <span class="subtext">Loss: 0.2597 → 0.2683 (30 Epochs in 3.61s)</span>
                </div>
                <div class="dvc-metric-pill">
                    <span class="title">Stage 6: Conformal Cal</span>
                    <span class="value">96.77% Cal Cov</span>
                    <span class="subtext">q̂: 0.6337 | τ_ood: -2.4029 (α = 0.05, N=155)</span>
                </div>
                <div class="dvc-metric-pill">
                    <span class="title">Stage 7: Holdout Eval</span>
                    <span class="value">92.07% Test Acc</span>
                    <span class="subtext">Set Size: 1.01 | CPU p95: 50.38 ms (20.9 FPS)</span>
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

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", initDVCExplorer);
    } else {
        initDVCExplorer();
    }

    if (typeof document$ !== "undefined" && typeof document$.subscribe === "function") {
        document$.subscribe(initDVCExplorer);
    }
})();
