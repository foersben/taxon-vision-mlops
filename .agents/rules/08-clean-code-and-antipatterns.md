---
type: Agent Rule
title: "Rule 08: Clean Code & Anti-Pattern Prevention"
status: stable
stale_after: "2027-01-01T00:00:00Z"
version: 1.0
description: "Standards for 4-part architectural docstrings, parameter containers, event-loop non-blocking I/O, RAII resource lifecycle, and fail-fast observability."
tags: [clean-code, architecture, quality, async]
generated: {by: process:scaffold-init, at: "2026-10-27T17:40:00Z"}
verified: {by: process:scaffold-init, at: "2026-10-27T17:40:00Z"}
trigger: always_on
severity: critical
---

# Mandates: Clean Code & Anti-Pattern Prevention

All software components in TaxonVision must strictly adhere to the following clean code mandates to prevent technical debt, runtime regressions, and concurrency bottlenecks.

## 1. Mandatory 4-Part Architectural Docstring Standard

Superficial one-line summaries are strictly prohibited for non-trivial public classes, modules, and functions. All components must implement the elongated 4-part Google-style documentation pattern:

* **Summary:** A concise 1-2 sentence definition of the component.
* **Why (Architectural Rationale & Trade-offs):** Concrete justification of why this pattern was chosen over alternatives, referencing physical hardware ceilings, latency budgets, memory limits, or legal compliance invariants.
* **How (Algorithmic Mechanism & Invariants):** Clear description of the internal data flow, mathematical formulations, runtime/space complexity $\mathcal{O}(\cdot)$, and zero-allocation or statistical guarantees.
* **Standard Fields:** Typed `Args`, `Returns`, `Raises`, and literature `References`.

## 2. Parameter Encapsulation & Tuple Blindness Ban

* **Parameter Arity Limit:** Functions and methods must never accept more than 4 primitive positional parameters. Any parameter list exceeding 4 arguments must be encapsulated into a typed Pydantic model, a frozen dataclass, or a config container.
* **Zero Tuple Blindness:** Returning untyped raw tuples for domain concepts (e.g. `tuple[str, float, int, list[float]]`) is forbidden. Methods returning composite entities must define an explicit `@dataclass(frozen=True)` or `NamedTuple` with descriptive field names and type annotations.

## 3. Asynchronous Concurrency & Event Loop Protection

* **Non-Blocking Invariant:** The primary asyncio event loop in FastAPI must never be blocked by synchronous operations.
* **Threadpool Offloading:** Any CPU-bound computation (image preprocessing, ONNX matrix math, statistical hypothesis testing) or synchronous disk/database I/O (SQLite queries, Parquet loading, file writes) called from within an `async def` route must be offloaded via `await anyio.to_thread.run_sync(...)` or dispatched via `BackgroundTasks`.

## 4. RAII Lifecycle Management for External Resources

* **Deterministic Context Scoping:** Stateful external sessions, database transactions, MLflow tracking runs, and temporary file locks must always be encapsulated in native Python context managers (`with` statements or `@contextmanager`).
* **Zero Orphaned Runs:** Manual tracking via boolean state flags (such as `is_active: bool`) is prohibited. The context manager must guarantee deterministic teardown, flush pending writes, and explicitly mark failed runs with an error status on unhandled exceptions.

## 5. Fail-Fast Observability & Metric Integrity

* **Zero Silent Error Masking:** Broad catch-all exception handlers (`except Exception:`) that silently swallow critical failures or substitute synthetic dummy outputs in production paths are strictly forbidden. All unexpected exceptions must be logged with full tracebacks (`logger.exception`). Synthetic fallbacks are permitted only when gated by an explicit configuration invariant (e.g. `allow_untrained_fallback: true` for cold container bootstrapping).
* **Metric Timing & Increment Invariant:** Prometheus success counters (`status="success"`) must be incremented strictly after the target operation executes without error. Error counters (`status="error"`) must be recorded within the corresponding exception handler. Latency histograms must accurately scope only the critical execution block.

## 6. Structural Sibling Topology

* When refactoring monolithic files exceeding 500 lines of code, decompose logic into focused single-responsibility sibling modules within the same directory (e.g. `dataset.py`, `promotion.py`, `visualizations.py`) rather than introducing nested subpackages. This preserves a flat package topology and prevents circular or eager submodule imports under Rule 00.
