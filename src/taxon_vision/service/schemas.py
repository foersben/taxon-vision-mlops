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


class TrainingRequest(BaseModel):
    """Parameters for model training."""

    extractor: str = "mobilenetv4_conv_small"
    epochs: int = 5
    batch_size: int = 16
    learning_rate: float = 0.001
    background: bool = False


class TrainingResponse(BaseModel):
    """Result and metrics from model training run."""

    status: str
    extractor: str
    epochs_trained: int
    final_train_loss: float | None = None
    final_val_loss: float | None = None
    final_val_accuracy: float | None = None
    checkpoint_path: str
    duration_seconds: float
    message: str
