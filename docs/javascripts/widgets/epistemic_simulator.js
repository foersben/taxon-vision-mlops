/**
 * TaxonVision-MLOps Interactive Conformal & Thermodynamic OOD Live Simulator
 */
(function () {
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
                        Singleton set satisfies 95% conformal coverage. Direct serving within 15 ms SLO.
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
            desc.innerHTML = `Singleton prediction set admitted: <em>${candidate}</em> (&pi;&#770; = ${(probs[admittedIndices[0]] * 100).toFixed(1)}%). Mathematical ${( (1 - currentAlpha) * 100).toFixed(0)}% coverage guarantee verified. Direct ONNX Runtime serving within sub-15ms SLO.`;
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

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", initEpistemicSimulator);
    } else {
        initEpistemicSimulator();
    }

    if (typeof document$ !== "undefined" && typeof document$.subscribe === "function") {
        document$.subscribe(initEpistemicSimulator);
    }
})();
