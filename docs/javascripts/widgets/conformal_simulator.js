/**
 * TaxonVision-MLOps Interactive Conformal Coverage & Helmholtz Energy Simulator
 */
(function () {
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
            const energyThresh = energySlider ? parseFloat(energySlider.value) : -2.40;
            const temp = tempSlider ? parseFloat(tempSlider.value) : 1.0;

            const coverageTarget = (1.0 - alpha) * 100;
            const qHat = (1.0 - alpha * 7.326).toFixed(4);
            const avgSetSize = (1.0 + Math.pow(alpha, -0.15) * 0.01).toFixed(2);
            const triageRate = (alpha * 100 * 0.36 + (energyThresh < -12.0 ? 5.2 : 0.03)).toFixed(1);

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

        if (alphaSlider.dataset.initialized !== "true") {
            alphaSlider.dataset.initialized = "true";
            alphaSlider.addEventListener("input", updateSimulation);
            if (energySlider) energySlider.addEventListener("input", updateSimulation);
            if (tempSlider) tempSlider.addEventListener("input", updateSimulation);
        }

        updateSimulation();
    }

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", initConformalSimulators);
    } else {
        initConformalSimulators();
    }

    if (typeof document$ !== "undefined" && typeof document$.subscribe === "function") {
        document$.subscribe(initConformalSimulators);
    }
})();
