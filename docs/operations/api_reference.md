---
type: Reference
title: FastAPI Service & OpenAPI Contract Reference
status: stable
stale_after: "2027-01-01T00:00:00Z"
version: 1.0
description: REST endpoints, request schemas, and response contracts.
tags: [api, fastapi, openapi, endpoints]
generated: {by: process:scaffold-init, at: "2026-10-02T10:00:00Z"}
verified: {by: process:scaffold-init, at: "2026-10-02T10:00:00Z"}
---

# FastAPI Service & OpenAPI Contract Reference

The TaxonVision REST API is built on FastAPI and designed to serve high-throughput, low-latency requests in a zero-allocation inference environment.

## 1. Architectural Setup: Non-Blocking Threadpool Offloading

**What we chose:**
All synchronous CPU/GPU bound operations (e.g., PyTorch training, ONNX inference matrix multiplications, forward-hooked Class Activation Mapping, and synchronous Redis cache queries) are explicitly offloaded to an AnyIO worker threadpool using `await anyio.to_thread.run_sync(...)`.

**Why we chose it (Strategic Validation):**
FastAPI uses a single-threaded `asyncio` event loop. A common anti-pattern in MLOps is placing heavy synchronous PyTorch calls, image decode routines, or Redis socket calls directly inside `async def` route handlers. When this happens, the single asyncio thread blocks completely until the computation or network wait finishes. During this time, the server cannot accept new connections or respond to `/health` liveness probes, causing Kubernetes to falsely assume the pod is dead and terminate it. Offloading to AnyIO ensures the main event loop remains free to handle incoming connections and health checks, guaranteeing robust scaling under load.

## 2. Zero-Allocation Inference Hot Path

**What we chose:**
Image preprocessing transforms (`torchvision.transforms.Compose`) are pre-allocated and cached using the `@lru_cache` decorator on the `get_image_transform()` method.

**Why we chose it (Semantic Validation):**
Re-instantiating transform pipelines on every incoming request triggers repeated memory allocations on the heap and forces the Python Garbage Collector into constant churn. For a service bound by strict sub-25ms inference SLAs, dynamic allocations on the hot path are unacceptable. Caching the immutable transform pipeline ensures zero-allocation routing, eliminating GC pauses and maintaining constant p95 latency percentiles.

## 3. OpenAPI REST & HTMX Endpoints Reference

The FastAPI service exposes machine REST API endpoints alongside server-rendered HTMX partials:

* **`POST /api/v1/predict`** (alias: `POST /predict`)
    * **Description:** High-throughput species classification endpoint. Preprocesses image, runs ONNX Runtime inference, evaluates non-conformity scores against calibrated $\hat{q}$, evaluates Helmholtz free energy against $\tau_{\text{ood}}$, and optionally renders visual attribution CAM overlays.
    * **Request:** `multipart/form-data` containing `file: UploadFile` and optional query flag `explain: bool = false`.
    * **Response Schema:** JSON object containing `predictions: list[TaxonPrediction]`, `conformal_set: list[int]`, `ood_score: float`, `is_ood: bool`, `routing_tier: str`, and optional `cam_overlay_base64: str | None`.
    * **Latency SLA:** < 25.0 ms p95 on GPU / < 50.0 ms p95 on CPU.
    * **Error Status Codes:**
        * `400 Bad Request` (Corrupted image format or unparsable payload)
        * `422 Unprocessable Entity` (File upload missing or empty)
        * `500 Internal Server Error` (Failures explicitly logged with full traceback and reported to Prometheus)
* **`POST /api/v1/explain`**
    * **Description:** On-demand visual explainability via forward-hooked Class Activation Mapping (CAM). Resolves image bytes either directly from multipart upload or by looking up `image_hash` from the distributed Redis ephemeral cache (`redis-ephemeral:6379`), eliminating multi-pod split-brain errors.
    * **Request:** Query parameters `image_hash: str | None`, `target_class_index: int | None`, or optional multipart `file: UploadFile`.
    * **Response Schema:**
        * JSON Client (`Accept: application/json`): `ExplainResponse` containing `image_hash: str`, `target_class_index: int`, `cam_overlay_base64: str`, and `execution_time_ms: float`.
        * HTMX Client (`HX-Request: true`): HTML drawer fragment (`partials/cam_result.html`) displaying the rendered CAM overlay heatmap and activation legend.
    * **Error Status Codes:**
        * `422 Unprocessable Entity` (Image hash not found in cache or expired beyond 15-minute TTL; renders inline warning fragment `partials/cam_error.html` for HTMX clients)
* **`POST /ui/predict-htmx`**
    * **Description:** Server-rendered prediction endpoint powering the operator dashboard. Executes ONNX inference, stores the specimen image in the distributed cache, and returns the interactive prediction card HTML partial with lazy-loading CAM inspection drawers.
    * **Request:** `multipart/form-data` containing `file: UploadFile`.
    * **Response Schema:** HTML fragment (`partials/prediction_card.html`).
* **`POST /api/v1/train`** (aliases: `POST /api/v1/training`, `POST /train`, `POST /training`)
    * **Description:** Asynchronous training trigger endpoint. Dispatches model classification head training onto the AnyIO background worker pool, preventing blocking of the web service.
    * **Request:** Query parameters `epochs: int = 15`, `tune: bool = false`, `extractor: str = "dinov3"`.
    * **Response Schema:** JSON object `{"status": "training_dispatched", "task": "background_task", "params": {...}}`.
    * **Latency SLA:** Immediate HTTP 200/202 dispatch response (< 10 ms).
* **`GET /health`**
    * **Description:** Basic Kubernetes liveness probe endpoint.
    * **Request Schema:** None.
    * **Response Schema:** JSON object `{"status": "healthy", "service": "taxon-vision"}`.
    * **Latency SLA:** < 2 ms.
* **`GET /health/ready`**
    * **Description:** Kubernetes readiness probe verifying model checkpoint availability, taxa catalog integrity, and runtime engine pre-warming.
    * **Request Schema:** None.
    * **Response Schema:** JSON object `{"status": "ready", "model_loaded": true, "taxa_count": 10}`.
    * **Latency SLA:** < 5 ms.
* **`GET /metrics`**
    * **Description:** Prometheus exposition scrape target exposing latency histograms, conformal set cardinality counters, and HTTP error metrics.
    * **Request Schema:** None.
    * **Response Schema:** Prometheus exposition text format.
    * **Latency SLA:** < 10 ms.
