---
type: Tool
title: Interactive Architecture & Knowledge Graph Hub
status: stable
stale_after: "2027-01-01T00:00:00Z"
version: 1.0
description: Interactive visual exploration portal embedding the TaxonVision architecture visualizer and OKF v0.2 knowledge graph.
tags: [interactive, visualizer, architecture, okf, knowledge-graph, cytoscape]
generated: {by: process:docs-librarian, at: "2026-10-18T12:00:00Z"}
verified: {by: process:docs-librarian, at: "2026-10-18T12:00:00Z"}
---

# Interactive Architecture & Knowledge Graph Hub

Welcome to the **TaxonVision-MLOps Interactive Hub**. This portal provides real-time, interactive graph visualizations of the end-to-end production architecture, DVC DAG data lineages, zero-allocation runtime pathways, and the formal Open Knowledge Format (OKF v0.2) dependency network.

<div class="metric-grid">
    <div class="metric-card">
        <div class="metric-header">Architecture Explorer</div>
        <div class="metric-value">Cytoscape.js</div>
        <div class="metric-desc">Interactive physics-driven topology graph mapping pipelines, ONNX engines, and Kubernetes workloads.</div>
    </div>
    <div class="metric-card">
        <div class="metric-header">Knowledge Graph</div>
        <div class="metric-value">OKF v0.2</div>
        <div class="metric-desc">Machine-verified semantic graph of all repository concepts, system patterns, and invariants.</div>
    </div>
    <div class="metric-card">
        <div class="metric-header">Pipeline DAG</div>
        <div class="metric-value">7 Stages</div>
        <div class="metric-desc">The complete Epistemic Diamond lifecycle from raw citizen-science ingest to calibrated inference.</div>
    </div>
</div>

---

## 1. System Architecture & Epistemic Diamond Visualizer

The interactive canvas below models the complete TaxonVision-MLOps ecosystem. You can pan, zoom, click nodes to inspect tensor dimensions and execution SLOs, and filter across the four operational tiers: **Ingestion & S3**, **Training & DVC**, **ONNX Runtime Serving**, and **Epistemic Safety Gates**.

<div class="visualizer-wrapper" id="vis-arch-wrapper">
    <div class="visualizer-header">
        <div class="title">
            <span class="badge-tag info">Live Engine</span>
            TaxonVision-MLOps Architecture &amp; Epistemic Intelligence Visualizer
        </div>
        <div class="visualizer-actions">
            <button class="btn-fullscreen" data-target="vis-arch-wrapper" type="button"><i class="fa-solid fa-expand"></i> Fullscreen</button>
        </div>
    </div>
    <iframe class="visualizer-iframe" src="architecture_visualizer.html" title="TaxonVision Architecture Visualizer" allow="fullscreen"></iframe>
</div>

### Visualizer Features & Navigation

* **Interactive Node Inspection:** Click any architectural block (e.g., `FastAPI /metrics`, `ONNX Engine`, `Split Conformal Gate`) to open its parameter card, tensor shape signature, and latency SLO.
* **Physics & Layout Controls:** Use the top control bar within the visualizer to toggle between hierarchical DAG view, circular layout, and force-directed clustering.
* **Invariant Badges:** Trace cryptographic SHA-256 validation bounds and distribution-free coverage thresholds across the inference path.

---

## 2. OKF v0.2 Knowledge Graph Explorer

The visualizer below renders the machine-attested knowledge graph maintained by `scripts/governance/validate_okf.py` and `scripts/governance/visualize_okf.py`. Every node represents an audited concept document or reference manual, linked by verified semantic dependencies.

<div class="visualizer-wrapper" id="vis-okf-wrapper">
    <div class="visualizer-header">
        <div class="title">
            <span class="badge-tag success">Verified OKF v0.2</span>
            Open Knowledge Graph &amp; Agentic Governance Explorer
        </div>
        <div class="visualizer-actions">
            <button class="btn-fullscreen" data-target="vis-okf-wrapper" type="button"><i class="fa-solid fa-expand"></i> Fullscreen</button>
        </div>
    </div>
    <iframe class="visualizer-iframe" src="viz.html" title="OKF Knowledge Graph Explorer" allow="fullscreen"></iframe>
</div>

### Knowledge Graph Interaction Guide

* **Graph Navigation:** Click and drag nodes to reorganize semantic clusters. Scroll to zoom in and out of specific subdomains.
* **Metadata Sidebar:** Select any concept node to inspect its trust tier (§5.3), freshness state (§5.5), and author verification receipt.
* **Search & Filter:** Type keywords into the search box to highlight matching architectural components and their upstream dependencies.
