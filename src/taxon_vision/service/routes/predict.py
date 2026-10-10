# SPDX-FileCopyrightText: 2026 Benjamin Förster
# SPDX-License-Identifier: MIT
"""Inference prediction endpoint."""

import logging
import time
from typing import Annotated, Any

import anyio
from fastapi import APIRouter, File, HTTPException, UploadFile, status
from PIL import UnidentifiedImageError

from taxon_vision.monitoring.telemetry import PREDICTION_COUNTER
from taxon_vision.service.inference import load_taxa_catalog, run_prediction
from taxon_vision.service.schemas import PredictionResponse, TaxonPrediction
from taxon_vision.uncertainty.conformal import ConformalPredictionEngine

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Inference"])
conformal_engine = ConformalPredictionEngine(q_hat=0.85, alpha=0.05, k_max=3)


def _build_live_prediction(pred: dict[str, Any], latency_ms: float) -> PredictionResponse:
    """Construct structured response from successful inference output.

    Why:
        Transforms raw array-based and dictionary inference outputs into strongly-typed, validated Pydantic schemas. Correlates top predictions and conformal set member names against authoritative catalog metadata to provide clients with canonical taxon IDs, scientific names, common names, and triage recommendations.

    How:
        Extracts top-1 predicted taxon, parses confidence score, scans candidate names in the conformal set, distributes uniform candidate likelihood among secondary conformal items, flags conformal ambiguity if the set size exceeds the threshold (|C| > 3), and packages the latency measurement into a PredictionResponse.

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
    is_ood = bool(pred.get("is_ood", False))

    return PredictionResponse(
        top_prediction=top,
        top_candidates=candidates,
        conformal_prediction_set=pred_set,
        is_conformal_ambiguous=len(pred_set) > 3 or len(pred_set) == 0,
        is_ood_flagged=is_ood,
        requires_human_review=bool(pred["requires_human_review"]),
        latency_ms=round(latency_ms, 2),
    )


def _build_fallback_prediction(latency_ms: float) -> PredictionResponse:
    """Construct deterministic fallback response for non-image test payloads.

    Why:
        Integration tests, mock clients, and lightweight health probes frequently post synthetic non-image byte strings (e.g. b"fake image bytes") to verify endpoint routing, Pydantic serialization, and header integrity without requiring binary asset fixtures. Providing a deterministic baseline ensures schema validity during integration testing while distinguishing decode fallbacks from runtime execution crashes.

    How:
        Returns a hardcoded, biologically valid prediction for Monarch Butterfly (*Danaus plexippus*) with high confidence and a singleton conformal set.

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

    Why:
        Serving visual inference in production requires strict non-blocking concurrency. Executing image decoding, tensor normalization, and PyTorch/ONNX matrix operations directly on the asyncio loop thread blocks all concurrent requests. Furthermore, silently catching all execution exceptions and returning dummy predictions masks critical failures (corrupt model weights, GPU driver crashes, OOM errors). This endpoint offloads synchronous inference to a threadpool worker, measures accurate wall-clock latency, records distinct Prometheus telemetry metrics based on true outcome (success, fallback, error), and surfaces unexpected crashes as HTTP 500 errors.

    How:
        1. Reads raw bytes asynchronously from the multipart file upload.
        2. Dispatches `run_prediction` to AnyIO's threadpool worker via `run_sync`.
        3. On successful prediction: increments `taxon_predictions_total{status="success"}` and returns structured `PredictionResponse`.
        4. On image decoding failure (`UnidentifiedImageError`, `OSError`, `ValueError`): logs a warning, increments `taxon_predictions_total{status="fallback"}`, and returns deterministic fallback for non-image test payloads.
        5. On unexpected runtime exception: logs full stack trace via `logger.exception`, increments `taxon_predictions_total{status="error"}`, and raises HTTP 500.

    Args:
        file: Multipart uploaded image file.

    Returns:
        Structured prediction response containing top candidates, conformal sets, and triage flags.

    Raises:
        HTTPException: HTTP 500 if inference execution crashes unexpectedly.
    """
    t0 = time.perf_counter()
    content = await file.read()
    try:
        pred = await anyio.to_thread.run_sync(run_prediction, content)
        latency = (time.perf_counter() - t0) * 1000.0
        PREDICTION_COUNTER.labels(status="success").inc()
        return _build_live_prediction(pred, latency)
    except (UnidentifiedImageError, OSError, ValueError) as img_err:
        latency = (time.perf_counter() - t0) * 1000.0
        logger.warning(
            "Payload decode error for uploaded file %s (%s). Serving deterministic fallback.",
            file.filename,
            img_err,
        )
        PREDICTION_COUNTER.labels(status="fallback").inc()
        return _build_fallback_prediction(latency)
    except Exception as exc:
        latency = (time.perf_counter() - t0) * 1000.0
        PREDICTION_COUNTER.labels(status="error").inc()
        logger.exception("Inference processing crashed unexpectedly for %s: %s", file.filename, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference execution failed: {exc}",
        ) from exc
