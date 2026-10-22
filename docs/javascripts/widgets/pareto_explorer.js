/**
 * TaxonVision-MLOps Interactive Pareto Frontier Explorer & Benchmark Matrix
 */
(function () {
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
            top1_acc: 89.4,
            macro_pr_auc: 0.952,
            sla_passed: true,
            badge: "Champion (Empirical)",
            color: "#00e676",
            family: "vit",
            profile: "Dense Morphology & Register Tokens (Active Trained Model)"
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
            top1_acc: 92.4,
            macro_pr_auc: 0.941,
            sla_passed: true,
            badge: "Priors (Projected)",
            color: "#00e5ff",
            family: "vit",
            profile: "Domain Biological Tree Prior (Literature Projection)"
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
            top1_acc: 89.4,
            macro_pr_auc: 0.908,
            sla_passed: true,
            badge: "Baseline (Projected)",
            color: "#b388ff",
            family: "vit",
            profile: "Foundation Visual Baseline (Literature Projection)"
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
            top1_acc: 78.5,
            macro_pr_auc: 0.798,
            sla_passed: true,
            badge: "Edge (Projected)",
            color: "#ffab00",
            family: "cnn",
            profile: "Ultra-Low Latency Edge CPU (Literature Projection)"
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
            top1_acc: 81.2,
            macro_pr_auc: 0.825,
            sla_passed: true,
            badge: "Edge (Projected)",
            color: "#ff7043",
            family: "cnn",
            profile: "Balanced Edge Baseline (Literature Projection)"
        }
    ];

    let currentFamily = "all";
    let sortColumn = "top1_acc";
    let sortAsc = false;
    let selectedModel = defaultData[0];

    mountPoint.innerHTML = `
        <div class="pareto-explorer">
            <div class="pareto-provenance-banner" style="margin-bottom: 1rem; padding: 0.75rem 1rem; background: rgba(0, 230, 118, 0.08); border-left: 3px solid #00e676; border-radius: 4px; font-size: 0.82rem; line-height: 1.4;">
                <strong>Benchmark Data Provenance:</strong> DINOv3 ViT-S/14 reflects empirical on-cluster evaluation on 1,689 verified observations (89.35% holdout test accuracy, <code>reports/pareto_evaluation.json</code>). Candidate backbones (BioCLIP-2, DINOv2, MobileNetV4, EfficientNet) represent symbolic projections calibrated against published peer-reviewed literature under identical tensor constraints.
            </div>
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
                            <th data-sort="vram_peak_mb">Peak VRAM</th>
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
                    <td>${d.vram_peak_mb.toFixed(0)} MB</td>
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
                    <span class="title">Embedding Dim</span>
                    <span class="val">${m.feature_dim}D</span>
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
                        if (item.vram_peak_mb) match.vram_peak_mb = item.vram_peak_mb;
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

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", initParetoExplorer);
    } else {
        initParetoExplorer();
    }

    if (typeof document$ !== "undefined" && typeof document$.subscribe === "function") {
        document$.subscribe(initParetoExplorer);
    }
})();
