/**
 * TaxonVision-MLOps Fullscreen Toggle Controller
 */
(function () {
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
    }

    // Escape key listener to exit fullscreen (registered once at module scope)
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

    if (typeof document$ !== "undefined") {
        document$.subscribe(initFullscreenHandlers);
    } else {
        document.addEventListener("DOMContentLoaded", initFullscreenHandlers);
    }
})();
