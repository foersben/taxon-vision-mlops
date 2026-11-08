---
type: Reference
title: FastAPI Service & OpenAPI Contract Reference
status: stable
stale_after: "2027-01-01T00:00:00Z"
version: 1.1
description: Complete REST API and HTMX hypermedia endpoint reference, threadpool concurrency invariants, and authentication contracts.
tags: [api, fastapi, openapi, endpoints, htmx, drift, retrain, passport]
generated: {by: "process:docs-librarian", at: "2026-11-08T17:40:00Z"}
verified: {by: "process:docs-librarian", at: "2026-11-08T17:40:00Z"}
sources:
  - resource: "src/taxon_vision/service/api.py"
  - resource: "src/taxon_vision/service/routes/predict.py"
  - resource: "src/taxon_vision/service/routes/explain.py"
  - resource: "src/taxon_vision/service/routes/views.py"
  - resource: "src/taxon_vision/service/routes/drift.py"
  - resource: "src/taxon_vision/service/routes/retrain.py"
  - resource: "src/taxon_vision/service/schemas.py"
---

# FastAPI Service & OpenAPI Contract Reference

The TaxonVision-MLOps web service ([`src/taxon_vision/service/api.py`](file:///home/benni/Documents/antigravity_workspace/taxon-vision-mlops/src/taxon_vision/service/api.py)) unifies the machine-to-machine REST API, the server-rendered Jinja2/HTMX operator cockpit, the on-demand statistical drift evaluation runner, and the Grafana-to-Airflow retraining webhook relay inside a single high-concurrency ASGI application.

---

## 1. Architectural Invariants: Event-Loop Protection & Zero-Allocation Caching

### 1.1 Non-Blocking AnyIO Threadpool Offloading (Rule 08)

FastAPI runs on a single-threaded `asyncio` event loop per worker process. Executing synchronous CPU-bound operations (image decoding, ONNX Runtime matrix math, Wasserstein-1 / MMD² permutation testing, forward-hooked CAM rendering) or synchronous I/O (PostgreSQL queries, Redis cache lookups, local JSON report reads) directly inside an `async def` route handler would stall the event loop, causing Kubernetes `/health` liveness probes to time out.

Every route in `src/taxon_vision/service/routes/` enforces non-blocking concurrency via two mechanisms:

* **`await anyio.to_thread.run_sync(...)`:** Offloads synchronous inference (`run_prediction`), CAM generation, drift computation (`evaluate_recent_drift`), Redis caching (`cache.store` / `cache.get`), and Model Passport compilation (`load_model_passport`) onto the AnyIO worker threadpool.
* **FastAPI `BackgroundTasks`:** Dispatches PostgreSQL prediction telemetry persistence (`get_telemetry_store().record(rec)`) after the HTTP response payload is constructed, ensuring database inserts add $0.0\,\text{ms}$ to client-perceived `/api/v1/predict` latency.

### 1.2 Pre-Allocated Transform & Engine Singleton Caching (Rule 02)

Re-instantiating `torchvision.transforms.Compose` pipelines or ONNX Runtime sessions on every request triggers heap fragmentation and Python garbage collection pauses. Both `get_image_transform()` and the active `ONNXInferenceEngine` are cached in memory via `@lru_cache`, while the Active Model Passport (`load_model_passport()`) is cached with a 60-second monotonic TTL.

---

## 2. Complete Endpoint Reference Matrix

| HTTP Method & Path | Route Module | Auth Guard | Primary Purpose | Latency SLO |
| :--- | :--- | :--- | :--- | :--- |
| `GET /` | [`routes/views.py`](file:///home/benni/Documents/antigravity_workspace/taxon-vision-mlops/src/taxon_vision/service/routes/views.py) | Public | Renders the Jinja2/HTMX operator dashboard (`index.html`) with the frosted-glass **Active Model Passport** governance strip. | $< 15\,\text{ms}$ (`< 0.1 ms` on passport cache hit) |
| `POST /ui/predict-htmx` | [`routes/views.py`](file:///home/benni/Documents/antigravity_workspace/taxon-vision-mlops/src/taxon_vision/service/routes/views.py) | Public | Processes uploaded specimen image, caches raw bytes in Redis (`taxon:img:{sha256}`), and returns the HTMX `partials/prediction_card.html` fragment. | $< 35\,\text{ms}$ p95 |
| `POST /api/v1/predict` (alias `/predict`) | [`routes/predict.py`](file:///home/benni/Documents/antigravity_workspace/taxon-vision-mlops/src/taxon_vision/service/routes/predict.py) | Public | Machine REST classification endpoint returning top-1 taxon, Split Conformal set $C(X)$, Helmholtz OOD energy, and async telemetry dispatch. | $< 25\,\text{ms}$ p95 |
| `POST /api/v1/explain` | [`routes/explain.py`](file:///home/benni/Documents/antigravity_workspace/taxon-vision-mlops/src/taxon_vision/service/routes/explain.py) | Public | Generates forward-hooked Class Activation Mapping (CAM) overlay from `image_hash` (Redis lookup) or uploaded file; returns JSON or HTMX partial. | $< 40\,\text{ms}$ p95 |
| `POST /api/v1/drift/evaluate` | [`routes/drift.py`](file:///home/benni/Documents/antigravity_workspace/taxon-vision-mlops/src/taxon_vision/service/routes/drift.py) | Bearer `TAXON_INTERNAL_API_TOKEN` | Executes Tier 1 ($W_1$) and Tier 2 (RBF $\text{MMD}^2$) drift evaluation over the latest 500 PostgreSQL predictions, logs to MLflow, and records `drift_runs`. | $< 500\,\text{ms}$ (threadpool offload) |
| `GET /api/v1/drift/latest` | [`routes/drift.py`](file:///home/benni/Documents/antigravity_workspace/taxon-vision-mlops/src/taxon_vision/service/routes/drift.py) | Public | Retrieves the most recent `DriftRunSummary` from the PostgreSQL `drift_runs` table without recomputing statistics. | $< 5\,\text{ms}$ |
| `POST /api/v1/retrain/webhook` | [`routes/retrain.py`](file:///home/benni/Documents/antigravity_workspace/taxon-vision-mlops/src/taxon_vision/service/routes/retrain.py) | Bearer `TAXON_RETRAIN_WEBHOOK_SECRET` | Receives Grafana Alertmanager webhooks, enforces a 6-hour cooldown lock, and triggers the `model_retraining` DAG in Airflow 3.0. | $< 25\,\text{ms}$ |
| `POST /api/v1/train` | [`routes/train.py`](file:///home/benni/Documents/antigravity_workspace/taxon-vision-mlops/src/taxon_vision/service/routes/train.py) | Public / Internal | Dispatches asynchronous classification head training onto background workers. | $< 10\,\text{ms}$ dispatch |
| `GET /health` & `GET /health/ready` | [`routes/health.py`](file:///home/benni/Documents/antigravity_workspace/taxon-vision-mlops/src/taxon_vision/service/routes/health.py) | Public | Kubernetes liveness and readiness probes verifying ONNX checkpoint and taxa catalogue readiness. | $< 5\,\text{ms}$ |
| `GET /metrics` | [`api.py`](file:///home/benni/Documents/antigravity_workspace/taxon-vision-mlops/src/taxon_vision/service/api.py) | Public (Cluster) | Prometheus scrape target exposing latency histograms, conformal counters, and `DriftMetricsCollector` gauges. | $< 10\,\text{ms}$ |

---

## 3. Detailed Endpoint Contracts

### 3.1 `GET /` (Operator Cockpit & Active Model Passport)

* **Execution Flow:** Offloads [`load_model_passport()`](file:///home/benni/Documents/antigravity_workspace/taxon-vision-mlops/src/taxon_vision/service/passport.py#L168-L220) via `await anyio.to_thread.run_sync(load_model_passport)`.
* **Bound Template Context (`ModelPassport`):**
    * `model_name` (`str`, e.g. `"taxon_vision_onnx@Production"`), `model_version` (`str`), `mlflow_model_id` (`str | None`), `backbone` (`str`, e.g. `"dinov3 INT8"`).
    * `nominal_coverage` (`float`, `0.95`), `empirical_test_coverage` (`float`), `mean_set_size` (`float`), `singleton_rate` (`float`), `p95_latency_ms` (`float`).
    * `exchangeability_status` (`"Healthy"` | `"Drift Warning"` | `"Warming Up"`) and `mmd2_p_value` (`float | None`).

### 3.2 `POST /api/v1/predict` (Species Classification & Epistemic Triage)

* **Request:** `multipart/form-data` with `file: UploadFile` (JPEG, PNG, or WebP $\le 15\,\text{MB}$) and optional query parameter `explain: bool = false`.
* **Response (`PredictionResponse`):**
    * `observation_id` (`UUID`), `taxon_id` (`int`), `scientific_name` (`str`), `common_name` (`str`), `confidence` (`float`).
    * `conformal_set` (`list[str]` of admissible taxa satisfying $1 - \hat{\pi}_k(\mathbf{x}) \le \hat{q}_\alpha$).
    * `ood_energy_score` (`float`), `is_ood` (`bool`), `requires_human_review` (`bool`), `routing_tier` (`"auto_accepted"` | `"human_triage"` | `"ood_rejected"`).
    * `latency_ms` (`float`) and optional `cam_heatmap_base64` (`str | None`).
* **Asynchronous Telemetry Side-Effect:** Queues `PredictionRecord` (including the 384-D pre-normalised `embedding`) into FastAPI `BackgroundTasks` for insertion into PostgreSQL `predictions`.

### 3.3 `POST /api/v1/drift/evaluate` & `GET /api/v1/drift/latest`

* **Authentication (`POST /api/v1/drift/evaluate`):** Validated by `verify_internal_token()` against `TAXON_INTERNAL_API_TOKEN` (`HTTP 401 Unauthorized` on mismatch).
* **Response (`DriftEvaluationResponse`):**

```json
{
  "drift_detected": false,
  "status": "evaluated",
  "w1_distance": 0.0684,
  "w1_threshold": 0.15,
  "mmd2": 0.0041,
  "mmd2_p_value": 0.4129,
  "window_size": 500,
  "n_reference": 169,
  "mlflow_run_id": "a8f39c12e4b04891a2c3d4e5f6071829",
  "report_path": "reports/drift_report.html"
}
```

### 3.4 `POST /api/v1/retrain/webhook` (Grafana-to-Airflow Relay)

* **Authentication:** Validated by `verify_retrain_token()` against `TAXON_RETRAIN_WEBHOOK_SECRET` using constant-time `hmac.compare_digest` (`HTTP 401 Unauthorized` on mismatch).
* **Request (`GrafanaWebhookPayload`):** Standard Grafana Alertmanager webhook JSON body containing `status`, `alerts[]`, and `commonLabels` (mapped via Pydantic alias to `common_labels`).
* **Response (`RetrainWebhookResponse`, HTTP 200 OK):**
    * `status`: `"triggered"` (Airflow `model_retraining` DAG run created), `"suppressed"` (6-hour cooldown lock active), or `"ignored"` (no firing alert with `action="retrain"`).
    * `detail`: Human-readable explanation (including remaining cooldown seconds when suppressed).
    * `dag_run_id`: Dispatched Airflow DAG run identifier (or `null` when suppressed/ignored).
