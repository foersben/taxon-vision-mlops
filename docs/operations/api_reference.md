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

* `POST /api/v1/predict`: Accepts multipart image upload, returns top-k predictions and conformal sets.
* `POST /api/v1/explain`: Returns Grad-CAM visual attribution heatmaps.
* `POST /api/v1/feedback`: Ingests citizen reviews for validation and retraining loops.
* `GET /health`: Liveness probe.
* `GET /metrics`: Prometheus telemetry scrape target.
