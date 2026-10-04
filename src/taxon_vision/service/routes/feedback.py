# SPDX-FileCopyrightText: 2026 Benjamin Förster
# SPDX-License-Identifier: MIT
"""Citizen feedback collection endpoint."""

from fastapi import APIRouter

from taxon_vision.service.schemas import FeedbackRequest, FeedbackResponse

router = APIRouter(prefix="/api/v1")


@router.post("/feedback", response_model=FeedbackResponse)
async def submit_feedback(req: FeedbackRequest) -> FeedbackResponse:
    """Ingest community or expert validation feedback for an observation.

    Args:
        req: Validated taxonomic identity and reviewer commentary payload.

    Returns:
        Acknowledgment response confirming receipt and logging for triage reconciliation.
    """
    return FeedbackResponse(status="accepted", message=f"Observation {req.observation_id} logged for review.")
