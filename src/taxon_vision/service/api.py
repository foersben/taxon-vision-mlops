# SPDX-FileCopyrightText: 2026 Benjamin Förster
# SPDX-License-Identifier: MIT
"""FastAPI Application Composition Root.

Why:
    TaxonVision requires a unified deployment artifact serving both automated high-throughput machine prediction requests (OpenAPI REST endpoints) and interactive human verification dashboards (HTMX/Jinja2 server-rendered views). Unifying these surfaces inside a single FastAPI ASGI application simplifies Kubernetes pod topology, shares cached PyTorch and conformal inference memory spaces, and ensures telemetry scrapes cover all operational surfaces.

How:
    Instantiates the top-level FastAPI application with formal OpenAPI metadata, mounts static CSS/JS styling assets if present, and mounts modular route handlers:
    - `health`: Liveness probes and Prometheus metric scrapes.
    - `predict`: High-throughput species classification with conformal sets.
    - `train`: Model head training triggering and status reporting.
    - `explain`: Visual attribution Class Activation Maps (CAM).
    - `views`: Interactive Jinja2/HTMX dashboard views for field biologists.
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from taxon_vision.config import get_settings
from taxon_vision.service.inference import get_image_transform, load_taxa_catalog, reset_model_caches
from taxon_vision.service.routes import explain, health, predict, train, views
from taxon_vision.service.storage import MemoryImageCache, get_image_cache


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Manage ASGI application startup warm-up and deterministic shutdown cleanup.

    Args:
        _app: FastAPI application instance.

    Yields:
        None during the active lifetime of the ASGI server.
    """
    load_taxa_catalog()
    get_image_transform()
    yield
    cache = get_image_cache()
    if isinstance(cache, MemoryImageCache):
        cache.clear()
    reset_model_caches()


settings = get_settings()

app = FastAPI(
    title=settings.service.name,
    description="Automated MLOps pipeline for species classification with conformal uncertainty guarantees.",
    version=settings.service.version,
    lifespan=lifespan,
)

static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

app.include_router(health.router)
app.include_router(predict.router)
app.include_router(train.router)
app.include_router(explain.router)
app.include_router(views.router)
