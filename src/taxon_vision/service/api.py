# SPDX-FileCopyrightText: 2026 Benjamin Förster
# SPDX-License-Identifier: MIT
"""FastAPI Application Composition Root."""

from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from taxon_vision.service.routes import explain, health, predict, train, views

app = FastAPI(
    title="TaxonVision Species Identification Service",
    description="Automated MLOps pipeline for species classification with conformal uncertainty guarantees.",
    version="0.1.0",
)

static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

app.include_router(health.router)
app.include_router(predict.router)
app.include_router(train.router)
app.include_router(explain.router)
app.include_router(views.router)
