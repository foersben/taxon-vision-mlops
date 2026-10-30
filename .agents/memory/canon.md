---
type: Agent Memory
title: TaxonVision Architectural Canon Memory
status: stable
stale_after: "2027-01-01T00:00:00Z"
version: 1.0
description: Immutable architectural canon and hard invariants learned from prior engineering cycles.
tags: [canon, invariants, memory, decisions]
generated: {by: process:okf-updater, at: "2026-10-30T00:00:00Z"}
verified: {by: process:okf-updater, at: "2026-10-30T00:00:00Z"}
---

# TaxonVision Architectural Canon Memory

## 2026-10-25 - Hardware Invariant Policy (Rule 07)

* **Learning:** Never invent or substitute cloud GPU types (A100, H100, T4) or consumer hardware variations.
* **Invariant:** When hardware specifications are stated, cite exclusively the verified physical machine: Intel Core i7-14700K (20 cores / 28 threads), NVIDIA GeForce RTX 5070 Ti (16 GB GDDR7 VRAM), 128 GB DDR5 RAM, 2 TB PCIe Gen4 NVMe. Mention hardware only when functionally necessary to justify architectural constraints (e.g. thread pinning, PyTorch memory budgeting $f=0.7$, non-CoW filesystem attributes).

## 2026-10-27 - 4-Part Google Docstring Standard (Rule 08)

* **Learning:** Superficial single-line docstrings fail to communicate architectural trade-offs, physical hardware bounds, and zero-allocation invariants to LLM agents.
* **Invariant:** All non-trivial public classes, modules, and functions must implement the 4-part docstring pattern: *Summary*, *Why (Architectural Rationale)*, *How (Algorithmic Mechanism & Complexity)*, and *Standard Fields* (Args/Returns/Raises/References).

## 2026-10-28 - Conformal Prediction Coverage Guarantee (Rule 06)

* **Learning:** Static softmax confidence thresholds are uncalibrated and yield silent misclassifications on long-tail observations.
* **Invariant:** Split Conformal Prediction must mathematically guarantee user-specified coverage $1 - \alpha$. Observations with prediction set size $|C(X)| > k_{\text{max}}$ or empty sets $|C(X)| = 0$ must route to human triage without exception.

## 2026-10-29 - Clean Toolchain & Async Event Loop Protection

* **Learning:** Synchronous disk I/O (Parquet reads, SQLite/DuckDB queries) or ONNX CPU inferences inside FastAPI `async def` routes block the main event loop, severely degrading concurrent request throughput.
* **Invariant:** All CPU-bound computations and synchronous I/O operations inside `async def` endpoints must be offloaded via `await anyio.to_thread.run_sync(...)`.
