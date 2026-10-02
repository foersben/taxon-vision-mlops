# SPDX-FileCopyrightText: 2026 Benjamin Förster
# SPDX-License-Identifier: MIT
"""Visual explanation endpoint."""

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(prefix="/api/v1")


class ExplainResponse(BaseModel):
    heatmap_status: str
    latency_ms: float


@router.post("/explain", response_model=ExplainResponse)
async def explain_prediction() -> ExplainResponse:
    return ExplainResponse(heatmap_status="generated", latency_ms=14.5)
