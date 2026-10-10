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
All synchronous CPU/GPU bound operations (e.g., PyTorch training and inference matrix multiplications) are explicitly offloaded to an AnyIO worker threadpool using `await anyio.to_thread.run_sync(...)`.

**Why we chose it (Strategic Validation):**
FastAPI uses a single-threaded `asyncio` event loop. A common anti-pattern in MLOps is placing heavy synchronous PyTorch calls directly inside `async def` route handlers. When this happens, the single asyncio thread blocks completely until the matrix multiplication finishes. During this time, the server cannot accept new connections or respond to `/health` liveness probes, causing Kubernetes to falsely assume the pod is dead and terminate it. Offloading to AnyIO ensures the main event loop remains free to handle incoming connections and health checks, guaranteeing robust scaling under load.

## 2. Zero-Allocation Inference Hot Path

**What we chose:**
Image preprocessing transforms (`torchvision.transforms.Compose`) are pre-allocated and cached using the `@lru_cache` decorator on the `get_image_transform()` method.

**Why we chose it (Semantic Validation):**
Re-instantiating transform pipelines on every incoming request triggers repeated memory allocations on the heap and forces the Python Garbage Collector into constant churn. For a service bound by strict sub-25ms inference SLAs, dynamic allocations on the hot path are unacceptable. Caching the immutable transform pipeline ensures zero-allocation routing, eliminating GC pauses and maintaining constant p95 latency percentiles.

## OpenAPI Endpoints Specifications

* **`POST /predict`**
    * **Description:** Accepts multipart image uploads, offloads preprocessing to threadpool, and returns top-k predictions and conformal sets.
    * **Request Schema:** `multipart/form-data` containing an `image` file.
    * **Response Schema:** JSON object containing an array of `predictions` (class, probability) and `conformal_set`.
    * **Latency Budget:** < 50ms p95.
    * **Error Status Codes:**
        * `400 Bad Request` (Invalid image format)
        * `500 Internal Server Error` (Errors are explicitly bubbled up rather than masked, ensuring metric fidelity)
* **`POST /conformal/calibrate`**
    * **Description:** Calibrates the conformal predictor with holdout data to guarantee empirical marginal coverage.
    * **Request Schema:** JSON object containing `dataset_id`.
    * **Response Schema:** JSON object containing updated `calibration_score`.
    * **Latency Budget:** Asynchronous execution, immediate 202 Accepted response.
    * **Error Status Codes:**
        * `404 Not Found` (Dataset ID not found)
* **`GET /health/ready`**
    * **Description:** Asynchronous liveness probe (never blocked by inference).
    * **Request Schema:** None.
    * **Response Schema:** JSON object `{"status": "ok"}`.
    * **Latency Budget:** < 5ms.
* **`GET /metrics`**
    * **Description:** Prometheus telemetry scrape target.
    * **Request Schema:** None.
    * **Response Schema:** Prometheus exposition format plain text.
    * **Latency Budget:** < 10ms.
