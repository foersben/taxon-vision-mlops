/**
 * TaxonVision-MLOps Interactive Widgets & Visualizer Controller
 */

document.addEventListener("DOMContentLoaded", function () {
    initFullscreenHandlers();
    initConformalSimulators();
});

// Support dynamic navigation in Material for MkDocs / Zensical
if (typeof document$ !== "undefined") {
    document$.subscribe(function () {
        initFullscreenHandlers();
        initConformalSimulators();
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
    if (!alphaSlider || alphaSlider.dataset.initialized) return;
    alphaSlider.dataset.initialized = "true";

    const energySlider = document.getElementById("sim-energy-slider");
    const tempSlider = document.getElementById("sim-temp-slider");

    function updateSimulation() {
        const alpha = parseFloat(alphaSlider.value);
        const energyThresh = parseFloat(energySlider.value);
        const temp = parseFloat(tempSlider.value);

        const coverageTarget = (1.0 - alpha) * 100;
        const nCal = 240; // Cal split size in TaxonVision catalog
        const kRank = Math.ceil((nCal + 1) * (1.0 - alpha));
        const quantileP = (kRank / nCal).toFixed(4);

        // Theoretical quantile approximation for standard non-conformity curve
        const qHat = (1.0 - Math.pow(1.0 - alpha, 1.45) * 0.85).toFixed(4);
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
