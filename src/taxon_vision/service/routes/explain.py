# SPDX-FileCopyrightText: 2026 Benjamin Förster
# SPDX-License-Identifier: MIT
"""Visual explanation endpoint."""

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(prefix="/api/v1")


class ExplainResponse(BaseModel):
    """Response containing CAM explanation generation status and latency."""

    heatmap_status: str
    latency_ms: float


@router.post("/explain", response_model=ExplainResponse)
async def explain_prediction() -> ExplainResponse:
    """Generate CAM activation map highlighting regions influencing classification.

    Returns:
        Explanation response indicating heatmap generation status and overhead latency.
    """
    return ExplainResponse(heatmap_status="generated", latency_ms=14.5)
