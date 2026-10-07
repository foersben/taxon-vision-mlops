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

## Endpoints

* `POST /api/v1/predict`: Accepts multipart image uploads, offloads preprocessing to threadpool, and returns top-k predictions and conformal sets. Errors are explicitly bubbled up (HTTP 500) rather than masked, ensuring metric fidelity.
* `POST /api/v1/explain`: Returns activation-weighted CAM visual heatmaps without backpropagation.
* `POST /api/v1/train`: Triggers the training pipeline on a background thread, preventing request starvation.
* `GET /health`: Asynchronous liveness probe (never blocked by inference).
* `GET /metrics`: Prometheus telemetry scrape target.
