# SPDX-FileCopyrightText: 2026 Benjamin Förster
# SPDX-License-Identifier: MIT
"""Server-rendered Jinja2/HTMX web interface."""

from pathlib import Path
from typing import Annotated

import anyio
from fastapi import APIRouter, File, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from taxon_vision.service.inference import run_prediction

router = APIRouter()
templates_dir = Path(__file__).parent.parent / "templates"
templates = Jinja2Templates(directory=str(templates_dir))


@router.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    """Render the primary operator dashboard home view.

    Why:
        Provides human taxonomists, field ecologists, and MLOps operators with a responsive, lightweight visual cockpit for testing model predictions, viewing conformal prediction bounds, and monitoring real-time inference latency without needing external Single-Page Application (SPA) client frameworks or complex build toolchains.

    How:
        Binds the incoming HTTP request context to the server-side Jinja2 environment, rendering `index.html` with HTMX attributes (`hx-post`, `hx-target`, `hx-swap`) that enable partial HTML swaps for seamless UI updates.

    Args:
        request: FastAPI HTTP request context for Jinja2 template rendering.

    Returns:
        Rendered HTML page with HTMX-enabled observation upload form.
    """
    return templates.TemplateResponse(request=request, name="index.html", context={"title": "TaxonVision Dashboard"})


@router.post("/ui/predict-htmx", response_class=HTMLResponse)
async def predict_htmx(request: Request, file: Annotated[UploadFile, File(...)]) -> HTMLResponse:
    """Process an uploaded observation image and return an HTMX prediction card partial.

    Why:
        In HTMX-driven hypermedia architectures, form submissions exchange small HTML snippets rather than full-page refreshes or client-side JSON parsing. Executing the visual inference pipeline off the main event loop ensures the server can handle concurrent UI interactions smoothly while returning rich visual indicators of confidence, conformal set coverage, and human triage referral flags.

    How:
        1. Asynchronously reads image bytes from the uploaded multipart file stream.
        2. Offloads synchronous inference execution to an AnyIO threadpool worker via `run_sync`.
        3. Formats resulting taxonomic classifications, confidence levels, conformal sets, and measured latencies into the template rendering context.
        4. Renders and returns the HTML partial `partials/prediction_card.html` for targeted DOM insertion.

    Args:
        request: FastAPI HTTP request context.
        file: Multipart uploaded organism image file.

    Returns:
        Rendered HTML partial fragment containing prediction badges and conformal sets.
    """
    image_bytes = await file.read()

    result = await anyio.to_thread.run_sync(run_prediction, image_bytes)

    context = {
        "filename": file.filename,
        "scientific_name": result["scientific_name"],
        "common_name": result["common_name"],
        "confidence": result["confidence"],
        "conformal_set": result["conformal_set"],
        "requires_human_review": result["requires_human_review"],
        "latency_ms": result["latency_ms"],
    }
    return templates.TemplateResponse(request=request, name="partials/prediction_card.html", context=context)
