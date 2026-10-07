function prepareMermaidBlocks() {
  const candidates = document.querySelectorAll("pre.mermaid, div.mermaid");

  for (const node of candidates) {
    if (node.dataset.mermaidPrepared === "true") {
      continue;
    }

    if (node.tagName === "PRE") {
      const code = node.querySelector("code");
      const text = code ? code.textContent ?? "" : node.textContent ?? "";
      const replacement = document.createElement("div");
      replacement.className = "mermaid";
      replacement.textContent = text;
      replacement.dataset.mermaidPrepared = "true";
      replacement.dataset.mermaidSource = text;
      node.replaceWith(replacement);
      continue;
    }

    if (!node.dataset.mermaidSource) {
      node.dataset.mermaidSource = node.textContent ?? "";
    }
    node.dataset.mermaidPrepared = "true";
  }
}

function ensureMermaidStyles() {
  if (document.getElementById("mermaid-custom-theme-style")) {
    return;
  }
  const style = document.createElement("style");
  style.id = "mermaid-custom-theme-style";
  style.textContent = `
    /* High-contrast crisp styling for all Mermaid diagrams */
    .mermaid svg {
      font-family: inherit !important;
      max-width: 100% !important;
      height: auto !important;
    }
    [data-md-color-scheme="slate"] .mermaid text {
      fill: #f8fafc !important;
    }
    [data-md-color-scheme="slate"] .mermaid .messageText {
      fill: #f8fafc !important;
      stroke: none !important;
    }
    [data-md-color-scheme="slate"] .mermaid .actor {
      stroke: #ff6e40 !important;
      fill: #1e2230 !important;
    }
    [data-md-color-scheme="slate"] .mermaid .actor-line {
      stroke: #94a3b8 !important;
    }
    [data-md-color-scheme="slate"] .mermaid .labelBox {
      stroke: #ff6e40 !important;
      fill: #1e2230 !important;
    }
    [data-md-color-scheme="slate"] .mermaid .labelText {
      fill: #f8fafc !important;
    }
    [data-md-color-scheme="slate"] .mermaid .loopText {
      fill: #f8fafc !important;
    }
    [data-md-color-scheme="slate"] .mermaid .node rect,
    [data-md-color-scheme="slate"] .mermaid .node circle,
    [data-md-color-scheme="slate"] .mermaid .node polygon {
      stroke: #ff6e40 !important;
      fill: #1e2230 !important;
    }
    [data-md-color-scheme="slate"] .mermaid .cluster rect {
      stroke: #ff6e40 !important;
      fill: #141824 !important;
    }
    [data-md-color-scheme="slate"] .mermaid .edgePath .path {
      stroke: #e2e8f0 !important;
      stroke-width: 1.5px !important;
    }
    [data-md-color-scheme="slate"] .mermaid .edgeLabel {
      background-color: #1e2230 !important;
      color: #f8fafc !important;
    }
    [data-md-color-scheme="slate"] .mermaid .edgeLabel rect {
      fill: #1e2230 !important;
      opacity: 0.85 !important;
    }
  `;
  document.head.appendChild(style);
}

function renderMermaidDiagrams() {
  if (typeof mermaid === "undefined") {
    return;
  }

  ensureMermaidStyles();
  const scheme = document.body ? document.body.getAttribute("data-md-color-scheme") : "slate";
  const isDark = scheme === "slate";

  prepareMermaidBlocks();
  mermaid.initialize({
    startOnLoad: false,
    securityLevel: "loose",
    theme: isDark ? "dark" : "default",
    flowchart: {
      htmlLabels: true,
      curve: "basis",
      useMaxWidth: true,
    },
    sequence: {
      useMaxWidth: true,
      showSequenceNumbers: true,
      actorMargin: 50,
      boxMargin: 10,
      boxTextMargin: 5,
      noteMargin: 10,
      messageMargin: 35,
    },
    themeVariables: isDark
      ? {
          // General base tokens
          background: "transparent",
          primaryColor: "#1e2230",
          primaryTextColor: "#f8fafc",
          primaryBorderColor: "#ff6e40",
          secondaryColor: "#2a2d3d",
          secondaryTextColor: "#f8fafc",
          secondaryBorderColor: "#ff6e40",
          tertiaryColor: "#141824",
          tertiaryTextColor: "#f8fafc",
          tertiaryBorderColor: "#ff6e40",
          lineColor: "#e2e8f0",
          textColor: "#f8fafc",
          mainBkg: "#1e2230",
          errorBkgColor: "#7f1d1d",
          errorTextColor: "#fecaca",

          // Flowchart tokens
          nodeBorder: "#ff6e40",
          nodeTextColor: "#f8fafc",
          clusterBkg: "#141824",
          clusterBorder: "#ff6e40",
          defaultLinkColor: "#e2e8f0",
          titleColor: "#ffcc80",
          edgeLabelBackground: "#1e2230",

          // Sequence Diagram tokens
          signalColor: "#e2e8f0",
          signalTextColor: "#f8fafc",
          actorBkg: "#1e2230",
          actorBorder: "#ff6e40",
          actorTextColor: "#f8fafc",
          actorLineColor: "#94a3b8",
          labelBoxBkgColor: "#1e2230",
          labelBoxBorderColor: "#ff6e40",
          labelTextColor: "#f8fafc",
          loopTextColor: "#f8fafc",
          noteBkgColor: "#2d221e",
          noteTextColor: "#ffcc80",
          noteBorderColor: "#ff6e40",
          activationBorderColor: "#ff6e40",
          activationBkgColor: "#374151",
          sequenceNumberColor: "#ffffff",

          // Class Diagram tokens
          classText: "#f8fafc",

          // State Diagram tokens
          labelColor: "#f8fafc",
          altBackground: "#141824",
        }
      : {
          background: "transparent",
          primaryColor: "#f8fafc",
          primaryTextColor: "#0f172a",
          primaryBorderColor: "#ff6e40",
          secondaryColor: "#f1f5f9",
          secondaryTextColor: "#0f172a",
          secondaryBorderColor: "#ff6e40",
          tertiaryColor: "#ffffff",
          tertiaryTextColor: "#0f172a",
          tertiaryBorderColor: "#ff6e40",
          lineColor: "#334155",
          textColor: "#0f172a",
          mainBkg: "#ffffff",

          // Flowchart tokens
          nodeBorder: "#ff6e40",
          nodeTextColor: "#0f172a",
          clusterBkg: "#f8fafc",
          clusterBorder: "#ff6e40",
          defaultLinkColor: "#334155",
          titleColor: "#c2410c",
          edgeLabelBackground: "#f8fafc",

          // Sequence Diagram tokens
          signalColor: "#1e293b",
          signalTextColor: "#0f172a",
          actorBkg: "#ffffff",
          actorBorder: "#ff6e40",
          actorTextColor: "#0f172a",
          actorLineColor: "#94a3b8",
          labelBoxBkgColor: "#ffffff",
          labelBoxBorderColor: "#ff6e40",
          labelTextColor: "#0f172a",
          loopTextColor: "#0f172a",
          noteBkgColor: "#fff7ed",
          noteTextColor: "#9a3412",
          noteBorderColor: "#ff6e40",
          activationBorderColor: "#ff6e40",
          activationBkgColor: "#e2e8f0",
          sequenceNumberColor: "#ffffff",

          // Class Diagram tokens
          classText: "#0f172a",

          // State Diagram tokens
          labelColor: "#0f172a",
          altBackground: "#ffffff",
        },
  });

  const nodes = document.querySelectorAll("div.mermaid[data-mermaid-prepared='true']");
  if (nodes.length > 0) {
    mermaid.run({ nodes });
  }
}

function handlePaletteToggle() {
  const nodes = document.querySelectorAll("div.mermaid");
  for (const node of nodes) {
    if (node.dataset.mermaidSource) {
      node.removeAttribute("data-processed");
      node.innerHTML = "";
      node.textContent = node.dataset.mermaidSource;
      node.dataset.mermaidPrepared = "true";
    }
  }
  renderMermaidDiagrams();
}

if (typeof MutationObserver !== "undefined" && document.body) {
  const observer = new MutationObserver((mutations) => {
    for (const m of mutations) {
      if (m.type === "attributes" && m.attributeName === "data-md-color-scheme") {
        handlePaletteToggle();
        break;
      }
    }
  });
  observer.observe(document.body, { attributes: true, attributeFilter: ["data-md-color-scheme"] });
}

if (typeof document$ !== "undefined") {
  document$.subscribe(renderMermaidDiagrams);
} else {
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", renderMermaidDiagrams);
  } else {
    renderMermaidDiagrams();
  }
}
