# SPDX-FileCopyrightText: 2026 Benjamin Förster
# SPDX-License-Identifier: MIT
"""Server-rendered Jinja2/HTMX web interface."""

from pathlib import Path

from fastapi import APIRouter, File, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

router = APIRouter()
templates_dir = Path(__file__).parent.parent / "templates"
templates = Jinja2Templates(directory=str(templates_dir))


@router.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(request=request, name="index.html", context={"title": "TaxonVision Dashboard"})


@router.post("/ui/predict-htmx", response_class=HTMLResponse)
async def predict_htmx(request: Request, file: UploadFile = File(...)) -> HTMLResponse:
    # Render HTMX partial card
    context = {
        "filename": file.filename,
        "scientific_name": "Danaus plexippus",
        "common_name": "Monarch Butterfly",
        "confidence": 0.942,
        "conformal_set": ["Danaus plexippus"],
        "requires_human_review": False,
        "latency_ms": 16.4,
    }
    return templates.TemplateResponse(request=request, name="partials/prediction_card.html", context=context)
