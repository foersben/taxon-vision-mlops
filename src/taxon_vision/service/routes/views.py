# SPDX-FileCopyrightText: 2026 Benjamin Förster
# SPDX-License-Identifier: MIT
"""Server-rendered Jinja2 and HTMX operator cockpit views.

Summary:
    Provides server-rendered HTML views and interactive HTMX endpoints powering the human-in-the-loop taxonomic classification cockpit, model passport display, and diagnostic explainability workflows.

Why (Architectural Rationale & Trade-offs):
    Deploying a server-rendered hypermedia UI with HTMX avoids client-side Single-Page Application (SPA) bundle overhead, node build toolchains, and state synchronization drift between frontend TypeScript schemas and backend Pydantic models. Serving clean HTML fragments keeps client memory footprints negligible on field tablets and desktop browsers whilst maintaining sub-50ms interaction round-trips.

How (Algorithmic Mechanism & Invariants):
    1. The primary GET / dashboard route compiles and renders an active ModelPassport asynchronously via anyio.to_thread.run_sync, preventing blocking disk or database reads on the asyncio event loop.
    2. The POST /ui/predict-htmx route accepts multipart image uploads, offloads ONNX inference and conformal prediction set generation to worker threads, caches raw image bytes in memory for follow-up explainability calls, and returns partial HTML cards for targeted DOM swapping.

Standard Fields:
    References: Fielding (2000) 'Architectural Styles and the Design of Network-based Software Architectures' (REST / HATEOAS).
"""

from pathlib import Path
from typing import Annotated

import anyio
from fastapi import APIRouter, File, HTTPException, Request, UploadFile, status
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from taxon_vision.config import get_settings
from taxon_vision.service.inference import (
    InvalidImageError,
    ModelNotReadyError,
    load_taxa_catalog,
    run_prediction,
)
from taxon_vision.service.passport import load_model_passport
from taxon_vision.service.presentation import build_top_candidates
from taxon_vision.service.storage import get_image_cache

router = APIRouter()
templates_dir = Path(__file__).parent.parent / "templates"
templates = Jinja2Templates(directory=str(templates_dir))


@router.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    """Render the primary operator dashboard home view with active model governance strip.

    Summary:
        Renders the server-side Jinja2 operator workbench, binding the active ModelPassport governance metadata and file upload form.

    Why (Architectural Rationale & Trade-offs):
        Provides human taxonomists, field ecologists, and MLOps operators with a responsive, lightweight visual cockpit for testing model predictions, inspecting conformal coverage guarantees, and monitoring inference latency without external Single-Page Application (SPA) client frameworks or complex build toolchains.

    How (Algorithmic Mechanism & Invariants):
        Compiles the active ModelPassport off the main event loop via anyio.to_thread.run_sync and binds the result to the server-side Jinja2 environment, rendering index.html with HTMX attributes (hx-post, hx-target, hx-swap) for partial DOM updates.

    Complexity:
        O(1) memory lookup when passport cache is valid; O(1) file reads on cache miss.

    Args:
        request: FastAPI HTTP request context for Jinja2 template rendering.

    Returns:
        Rendered HTML page with HTMX-enabled observation upload form and model passport strip.
    """
    passport = await anyio.to_thread.run_sync(load_model_passport)
    return templates.TemplateResponse(
        request=request, name="index.html", context={"title": "TaxonVision Dashboard", "passport": passport}
    )


@router.post("/ui/predict-htmx", response_class=HTMLResponse)
async def predict_htmx(request: Request, file: Annotated[UploadFile, File(...)]) -> HTMLResponse:
    """Process an uploaded observation image and return an HTMX prediction card partial.

    Why:
        In HTMX-driven hypermedia architectures, form submissions exchange small HTML snippets rather than full-page refreshes or client-side JSON parsing. Executing the visual inference pipeline off the main event loop ensures the server can handle concurrent UI interactions smoothly while returning rich visual indicators of confidence, conformal set coverage, and human triage referral flags.

    How:
        1. Asynchronously reads image bytes from the uploaded multipart file stream.
        2. Validates payload non-emptiness and size constraints.
        3. Offloads synchronous inference execution to an AnyIO threadpool worker via `run_sync`.
        4. Caches the raw bytes in the ephemeral in-memory image cache and exposes the SHA-256 `image_hash`
           to the template, so the CAM button can reference the image without re-uploading it.
        5. Builds the ranked top-K candidate list and epistemic telemetry (energy, entropy, thresholds).
        6. Renders and returns the HTML partial `partials/prediction_card.html` for targeted DOM insertion.

        CAM attribution is deliberately *not* computed here: it is requested on demand via
        `/api/v1/explain?image_hash=...`, which keeps the upload round-trip free of a second forward pass.

    Complexity:
        O(1) async thread dispatch plus O(K) template formatting.

    Args:
        request: FastAPI HTTP request context.
        file: Multipart uploaded organism image file.

    Returns:
        Rendered HTML partial fragment containing prediction badges and conformal sets.

    Raises:
        HTTPException: HTTP 413 for oversized payloads, 422 for undecodable images, 503 if model is unavailable.
    """
    image_bytes = await file.read()
    settings = get_settings()

    if len(image_bytes) == 0:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="Invalid or corrupted image upload.",
        )

    if len(image_bytes) > settings.service.max_image_upload_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Image upload exceeds maximum allowed size of {settings.service.max_image_upload_bytes} bytes.",
        )

    try:
        result = await anyio.to_thread.run_sync(run_prediction, image_bytes)
    except InvalidImageError as err:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(err)) from err
    except ModelNotReadyError as err:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(err)) from err

    cache = get_image_cache()
    image_hash = await anyio.to_thread.run_sync(cache.store, image_bytes)
    catalog = load_taxa_catalog()
    top_candidates = build_top_candidates(
        probabilities=result["probabilities"],
        catalog=catalog,
        conformal_set=result["conformal_set"],
        top_k=settings.service.ui_top_k_candidates,
    )

    context = {
        "filename": file.filename,
        "image_hash": image_hash,
        "scientific_name": result["scientific_name"],
        "common_name": result["common_name"],
        "confidence": result["confidence"],
        "conformal_set": result["conformal_set"],
        "routing_tier": result["routing_tier"],
        "is_ood": result["is_ood"],
        "requires_human_review": result["requires_human_review"],
        "latency_ms": result["latency_ms"],
        "top_taxon_id": result["taxon_id"],
        "top_candidates": top_candidates,
        "ood_energy_score": result["ood_energy_score"],
        "ood_energy_threshold": result["ood_energy_threshold"],
        "shannon_entropy": result["shannon_entropy"],
        "ood_entropy_threshold": result["ood_entropy_threshold"],
    }
    return templates.TemplateResponse(request=request, name="partials/prediction_card.html", context=context)
