# SPDX-FileCopyrightText: 2026 Benjamin Förster
# SPDX-License-Identifier: MIT
"""Citizen feedback collection endpoint."""

from fastapi import APIRouter

from taxon_vision.service.schemas import FeedbackRequest, FeedbackResponse

router = APIRouter(prefix="/api/v1")


@router.post("/feedback", response_model=FeedbackResponse)
async def submit_feedback(req: FeedbackRequest) -> FeedbackResponse:
    """Submit feedback.

    Args:
        req: The req parameter.

    Returns:
        The resulting value from the operation.
    """
    return FeedbackResponse(status="accepted", message=f"Observation {req.observation_id} logged for review.")
