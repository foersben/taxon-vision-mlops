# SPDX-FileCopyrightText: 2026 Benjamin Förster
# SPDX-License-Identifier: MIT
"""Visual explanation endpoint."""

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(prefix="/api/v1")


class ExplainResponse(BaseModel):
    """Explain response.

    Attributes:
        Various internal state and configuration variables used by the class.
    """

    heatmap_status: str
    latency_ms: float


@router.post("/explain", response_model=ExplainResponse)
async def explain_prediction() -> ExplainResponse:
    """Explain prediction.

    Returns:
        The resulting value from the operation.
    """
    return ExplainResponse(heatmap_status="generated", latency_ms=14.5)
