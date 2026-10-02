# SPDX-FileCopyrightText: 2026 Benjamin Förster
# SPDX-License-Identifier: MIT
"""Inference prediction endpoint."""

import time

from fastapi import APIRouter, File, UploadFile

from taxon_vision.monitoring.telemetry import PREDICTION_COUNTER
from taxon_vision.service.schemas import PredictionResponse, TaxonPrediction
from taxon_vision.uncertainty.conformal import ConformalPredictionEngine

router = APIRouter(prefix="/api/v1")
conformal_engine = ConformalPredictionEngine(q_hat=0.85, alpha=0.05, k_max=3)


@router.post("/predict", response_model=PredictionResponse)
async def predict_species(file: UploadFile = File(...)) -> PredictionResponse:
    """Predict species.

    Args:
        file: The file parameter.

    Returns:
        The resulting value from the operation.
    """
    t0 = time.perf_counter()
    PREDICTION_COUNTER.labels(status="success").inc()

    # Simulated prediction for Level 1 baseline
    top = TaxonPrediction(
        taxon_id=1, scientific_name="Danaus plexippus", common_name="Monarch Butterfly", confidence=0.92
    )
    candidates = [
        top,
        TaxonPrediction(taxon_id=2, scientific_name="Apis mellifera", common_name="Western Honey Bee", confidence=0.05),
    ]
    pred_set = ["Danaus plexippus"]
    latency = (time.perf_counter() - t0) * 1000.0

    return PredictionResponse(
        top_prediction=top,
        top_candidates=candidates,
        conformal_prediction_set=pred_set,
        is_conformal_ambiguous=False,
        is_ood_flagged=False,
        requires_human_review=False,
        latency_ms=latency,
    )
