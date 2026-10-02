---
type: Concept
title: Concept Design & Architectural Specification
status: stable
stale_after: "2027-01-01T00:00:00Z"
version: 1.0
description: Conceptual design and LaTeX documentation specification for TaxonVision-MLOps across 6 architectural chapters.
tags: [concept-design, architecture, latex, mlops]
generated: {by: process:concept-architect, at: "2026-10-02T10:00:00Z"}
verified: {by: process:concept-architect, at: "2026-10-02T10:00:00Z"}
---

# Concept Design & Architectural Specification

This document details the architectural concept design, operational levels, and mathematical invariants for TaxonVision-MLOps, serving as the Open Knowledge Format (OKF v0.2) counterpart to `docs/latex/concept_design/concept_design.tex`.

## Architectural Tenets

The platform prioritizes the operational production chain over isolated model performance: a versioned, monitored, deployed, and retrainable service is superior to complex unmanaged notebook prototypes.

## Six-Chapter Architectural Architecture

* **Chapter 1: Introduction & MLOps Maturity**
    * Paradigm shift from Level 0 to Level 4 safety.
* **Chapter 2: Infrastructure & Compute Topology**
    * Bare-metal hardware, Tailscale mesh, Hybrid CI/CD (GitHub Actions + Local GPU).
* **Chapter 3: The Open-Source MLOps Toolchain**
    * DagsHub, DVC, and MLflow integrated with Optuna for efficient hyperparameter sweeps.
* **Chapter 4: The Automated ML Pipeline**
    * Full pipeline mapping from Git Push to FastAPI serving.
* **Chapter 5: Epistemic Safety & Active Learning**
    * Split Conformal Prediction ($1 - \alpha$), Energy-based OOD scoring, and human-in-the-loop triage.
* **Chapter 6: Observability, Drift & Testing**
    * Prometheus telemetry, GitHub Actions + DVC orchestration, and robust testing (Unit, Integration, Invariants).

## Artifact Manifest

* `concept_design.tex`: Formal LaTeX manuscript suitable for academic or technical reporting.
* `concept_design.md`: OKF v0.2 architectural reference document.
