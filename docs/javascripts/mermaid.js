function prepareMermaidBlocks() {
  const candidates = document.querySelectorAll("pre.mermaid, div.mermaid");

  for (const node of candidates) {
    if (node.dataset.mermaidPrepared === "true") {
      continue;
    }

    if (node.tagName === "PRE") {
      const code = node.querySelector("code");
      const replacement = document.createElement("div");
      replacement.className = "mermaid";
      replacement.textContent = code ? code.textContent ?? "" : node.textContent ?? "";
      replacement.dataset.mermaidPrepared = "true";
      node.replaceWith(replacement);
      continue;
    }

    node.dataset.mermaidPrepared = "true";
  }
}

function renderMermaidDiagrams() {
  if (typeof mermaid === "undefined") {
    return;
  }

  const scheme = document.body ? document.body.getAttribute("data-md-color-scheme") : "slate";
  const isDark = scheme === "slate";

  prepareMermaidBlocks();
  mermaid.initialize({
    startOnLoad: false,
    securityLevel: "loose",
    theme: isDark ? "dark" : "default",
    flowchart: { htmlLabels: true },
    sequence: {
      useMaxWidth: true,
      showSequenceNumbers: true,
    },
    themeVariables: isDark
      ? {
          primaryColor: "#1e2230",
          primaryTextColor: "#f8fafc",
          primaryBorderColor: "#ff6e40",
          lineColor: "#e2e8f0",
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
        }
      : {
          primaryColor: "#f8fafc",
          primaryTextColor: "#0f172a",
          primaryBorderColor: "#ff6e40",
          lineColor: "#334155",
          signalColor: "#1e293b",
          signalTextColor: "#0f172a",
          actorBkg: "#ffffff",
          actorBorder: "#ff6e40",
          actorTextColor: "#0f172a",
          actorLineColor: "#94a3b8",
        },
  });

  const nodes = document.querySelectorAll("div.mermaid[data-mermaid-prepared='true']");
  if (nodes.length > 0) {
    mermaid.run({ nodes });
  }
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
