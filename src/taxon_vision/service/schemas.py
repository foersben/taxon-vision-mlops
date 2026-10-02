# SPDX-FileCopyrightText: 2026 Benjamin Förster
# SPDX-License-Identifier: MIT
"""Pydantic v2 schemas for API contracts."""

from __future__ import annotations

from pydantic import BaseModel


class TaxonPrediction(BaseModel):
    """Taxon prediction.

    Attributes:
        Various internal state and configuration variables used by the class.
    """

    taxon_id: int
    scientific_name: str
    common_name: str
    confidence: float


class PredictionResponse(BaseModel):
    """Prediction response.

    Attributes:
        Various internal state and configuration variables used by the class.
    """

    top_prediction: TaxonPrediction
    top_candidates: list[TaxonPrediction]
    conformal_prediction_set: list[str]
    is_conformal_ambiguous: bool
    is_ood_flagged: bool
    requires_human_review: bool
    latency_ms: float


class FeedbackRequest(BaseModel):
    """Feedback request.

    Attributes:
        Various internal state and configuration variables used by the class.
    """

    observation_id: str
    validated_taxon_id: int
    reviewer_name: str
    comments: str = ""


class FeedbackResponse(BaseModel):
    """Feedback response.

    Attributes:
        Various internal state and configuration variables used by the class.
    """

    status: str
    message: str
