# SPDX-FileCopyrightText: 2026 Benjamin Förster
# SPDX-License-Identifier: MIT
"""Server-rendered Jinja2/HTMX web interface."""

from pathlib import Path
from typing import Annotated

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

    Args:
        request: FastAPI HTTP request context for Jinja2 template rendering.

    Returns:
        Rendered HTML page with HTMX-enabled observation upload form.
    """
    return templates.TemplateResponse(request=request, name="index.html", context={"title": "TaxonVision Dashboard"})


@router.post("/ui/predict-htmx", response_class=HTMLResponse)
async def predict_htmx(request: Request, file: Annotated[UploadFile, File(...)]) -> HTMLResponse:
    """Process an uploaded observation image and return an HTMX prediction card partial.

    Args:
        request: FastAPI HTTP request context.
        file: Multipart uploaded organism image file.

    Returns:
        Rendered HTML partial fragment containing prediction badges and conformal sets.
    """
    image_bytes = await file.read()

    result = run_prediction(image_bytes)

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
