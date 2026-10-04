# SPDX-FileCopyrightText: 2026 Benjamin Förster
# SPDX-License-Identifier: MIT
"""Pydantic v2 schemas for API contracts."""

from __future__ import annotations

from pydantic import BaseModel


class TaxonPrediction(BaseModel):
    """Predicted taxonomic identity with confidence score."""

    taxon_id: int
    scientific_name: str
    common_name: str
    confidence: float


class PredictionResponse(BaseModel):
    """Structured inference response containing top predictions, conformal sets, and flags."""

    top_prediction: TaxonPrediction
    top_candidates: list[TaxonPrediction]
    conformal_prediction_set: list[str]
    is_conformal_ambiguous: bool
    is_ood_flagged: bool
    requires_human_review: bool
    latency_ms: float


class FeedbackRequest(BaseModel):
    """Payload submitted by domain experts reviewing model predictions."""

    observation_id: str
    validated_taxon_id: int
    reviewer_name: str
    comments: str = ""


class FeedbackResponse(BaseModel):
    """Confirmation response returned upon recording human review feedback."""

    status: str
    message: str
