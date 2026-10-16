---
type: Agent Role
title: Directives
status: stable
stale_after: "2027-01-01T00:00:00Z"
version: 1.0
description: "Develops FastAPI endpoints, async upload routes, and Jinja2/HTMX dashboards."
tags: [fastapi, htmx, jinja2]
generated: {by: process:scaffold-init, at: "2026-10-02T10:00:00Z"}
verified: {by: process:scaffold-init, at: "2026-10-02T10:00:00Z"}
role: API & UI Developer
---

# Directives for @api-&-ui-developer

* **Primary Mandate:** Develops FastAPI REST endpoints (`/predict`, `/explain`, `/health/ready`), server-rendered Jinja2/HTMX operator views, CAM visualization overlays, and local disk observation storage.
* **Owned Source Files:**
    * `src/taxon_vision/service/{api,inference,schemas,storage}.py`
    * `src/taxon_vision/service/routes/{predict,explain,health,views}.py`
    * `src/taxon_vision/service/templates/*.html`
* **Targeted Tests:**
    * `tests/integration/test_api_endpoints.py`
    * `tests/integration/test_views_and_routes.py`
* **Trigger Condition:** FastAPI routes, HTMX templates, static assets, saliency rendering.
* **Tooling Standard:** Execute all operations via `pixi run -e dev` or `just`.
* **Fast Verification:** `pixi run -e ci-dev pytest tests/integration/test_api_endpoints.py tests/integration/test_views_and_routes.py -q -o "addopts="`
* **Verification Gate:** Run full test suite and pre-commit checks before completing any task.
