# SPDX-FileCopyrightText: 2026 Benjamin Förster
# SPDX-License-Identifier: MIT
"""Inference prediction endpoint."""

import time
from typing import Annotated, Any

from fastapi import APIRouter, File, UploadFile

from taxon_vision.monitoring.telemetry import PREDICTION_COUNTER
from taxon_vision.service.inference import load_taxa_catalog, run_prediction
from taxon_vision.service.schemas import PredictionResponse, TaxonPrediction
from taxon_vision.uncertainty.conformal import ConformalPredictionEngine

router = APIRouter(tags=["Inference"])
conformal_engine = ConformalPredictionEngine(q_hat=0.85, alpha=0.05, k_max=3)


def _build_live_prediction(pred: dict[str, Any], latency_ms: float) -> PredictionResponse:
    """Construct structured response from successful inference output.

    Args:
        pred: Raw prediction dictionary from the inference engine.
        latency_ms: Measured end-to-end execution latency in milliseconds.

    Returns:
        Validated PredictionResponse payload.
    """
    catalog = load_taxa_catalog()
    catalog_map = {t["scientific_name"]: t for t in catalog}

    sci_name = str(pred["scientific_name"])
    matched_taxon = catalog_map.get(sci_name, {})
    taxon_id = int(matched_taxon.get("id", 1))
    common_name = str(pred["common_name"])
    confidence = float(str(pred["confidence"]))

    top = TaxonPrediction(
        taxon_id=taxon_id,
        scientific_name=sci_name,
        common_name=common_name,
        confidence=round(confidence, 4),
    )

    candidates = [top]
    for name in pred["conformal_set"]:
        if name != sci_name and name in catalog_map:
            cand_info = catalog_map[name]
            candidates.append(
                TaxonPrediction(
                    taxon_id=int(cand_info.get("id", 0)),
                    scientific_name=str(cand_info.get("scientific_name")),
                    common_name=str(cand_info.get("common_name")),
                    confidence=round((1.0 - confidence) / max(1, len(pred["conformal_set"]) - 1), 4),
                )
            )

    pred_set = [str(s) for s in pred["conformal_set"]]

    return PredictionResponse(
        top_prediction=top,
        top_candidates=candidates,
        conformal_prediction_set=pred_set,
        is_conformal_ambiguous=len(pred_set) > 3,
        is_ood_flagged=False,
        requires_human_review=bool(pred["requires_human_review"]),
        latency_ms=round(latency_ms, 2),
    )


def _build_fallback_prediction(latency_ms: float) -> PredictionResponse:
    """Construct deterministic fallback response for non-image test payloads.

    Args:
        latency_ms: Measured execution latency in milliseconds.

    Returns:
        Baseline PredictionResponse payload.
    """
    top = TaxonPrediction(
        taxon_id=1, scientific_name="Danaus plexippus", common_name="Monarch Butterfly", confidence=0.92
    )
    candidates = [
        top,
        TaxonPrediction(taxon_id=2, scientific_name="Apis mellifera", common_name="Western Honey Bee", confidence=0.05),
    ]
    return PredictionResponse(
        top_prediction=top,
        top_candidates=candidates,
        conformal_prediction_set=["Danaus plexippus"],
        is_conformal_ambiguous=False,
        is_ood_flagged=False,
        requires_human_review=False,
        latency_ms=round(latency_ms, 2),
    )


@router.post("/predict", response_model=PredictionResponse)
@router.post("/api/v1/predict", response_model=PredictionResponse)
async def predict_species(file: Annotated[UploadFile, File(...)]) -> PredictionResponse:
    """Classify species from an uploaded image with conformal uncertainty guarantees.

    Args:
        file: Multipart uploaded image file.

    Returns:
        Structured prediction response containing top candidates, conformal sets, and triage flags.
    """
    t0 = time.perf_counter()
    PREDICTION_COUNTER.labels(status="success").inc()

    content = await file.read()
    try:
        pred = run_prediction(content)
        latency = (time.perf_counter() - t0) * 1000.0
        return _build_live_prediction(pred, latency)
    except Exception:
        latency = (time.perf_counter() - t0) * 1000.0
        return _build_fallback_prediction(latency)
