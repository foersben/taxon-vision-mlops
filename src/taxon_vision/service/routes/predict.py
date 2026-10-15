# SPDX-FileCopyrightText: 2026 Benjamin Förster
# SPDX-License-Identifier: MIT
"""Inference prediction endpoint."""

import logging
import time
from typing import Annotated, Any

import anyio
from fastapi import APIRouter, File, HTTPException, UploadFile, status

from taxon_vision.config import get_settings
from taxon_vision.monitoring.telemetry import PREDICTION_COUNTER
from taxon_vision.service.inference import InvalidImageError, ModelNotReadyError, load_taxa_catalog, run_prediction
from taxon_vision.service.schemas import PredictionResponse, TaxonPrediction

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Inference"])


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

    settings = get_settings()
    pred_set = [str(s) for s in pred["conformal_set"]]
    is_ood = bool(pred.get("is_ood", False))

    return PredictionResponse(
        top_prediction=top,
        top_candidates=candidates,
        conformal_prediction_set=pred_set,
        is_conformal_ambiguous=len(pred_set) > settings.conformal.k_max or len(pred_set) == 0,
        is_ood_flagged=is_ood,
        requires_human_review=bool(pred["requires_human_review"]),
        latency_ms=round(latency_ms, 2),
    )


@router.post("/predict", response_model=PredictionResponse)
@router.post("/api/v1/predict", response_model=PredictionResponse)
async def predict_species(file: Annotated[UploadFile, File(...)]) -> PredictionResponse:
    """Classify species from an uploaded image with conformal uncertainty guarantees.

    Why:
        Serving visual inference in production requires strict non-blocking concurrency. Executing image decoding, tensor normalization, and PyTorch/ONNX matrix operations directly on the asyncio loop thread blocks all concurrent requests. Furthermore, silently catching execution exceptions and returning dummy predictions masks critical failures (corrupt model weights, missing artefacts, GPU driver crashes, OOM errors) and would feed fabricated labels into the human review queue. This endpoint offloads synchronous inference to a threadpool worker, measures accurate wall-clock latency, records distinct Prometheus telemetry per outcome, and reports each failure class with its proper HTTP status.

    How:
        1. Reads raw bytes asynchronously from the multipart file upload.
        2. Dispatches `run_prediction` to AnyIO's threadpool worker via `run_sync`.
        3. On success: increments `taxon_predictions_total{status="success"}` and returns the structured `PredictionResponse`.
        4. On an undecodable image (`InvalidImageError`): increments `status="invalid_input"` and raises HTTP 400.
        5. On a missing or incompatible model artefact (`ModelNotReadyError`): increments `status="unavailable"` and raises HTTP 503.
        6. On any other exception: logs the stack trace, increments `status="error"`, and raises HTTP 500.

    Args:
        file: Multipart uploaded image file.

    Returns:
        Structured prediction response containing top candidates, conformal sets, and triage flags.

    Raises:
        HTTPException: HTTP 400 for undecodable images, 503 if the model is not ready, 500 if inference crashes unexpectedly.
    """
    t0 = time.perf_counter()
    content = await file.read()
    settings = get_settings()
    if len(content) > settings.service.max_image_upload_bytes:
        PREDICTION_COUNTER.labels(status="invalid_input").inc()
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Image upload exceeds maximum allowed size of {settings.service.max_image_upload_bytes} bytes.",
        )
    try:
        pred = await anyio.to_thread.run_sync(run_prediction, content)
        latency = (time.perf_counter() - t0) * 1000.0
        PREDICTION_COUNTER.labels(status="success").inc()
        return _build_live_prediction(pred, latency)
    except InvalidImageError as img_err:
        logger.warning("Rejected undecodable upload %s: %s", file.filename, img_err)
        PREDICTION_COUNTER.labels(status="invalid_input").inc()
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(img_err)) from img_err
    except ModelNotReadyError as model_err:
        logger.error("Model not ready for %s: %s", file.filename, model_err)
        PREDICTION_COUNTER.labels(status="unavailable").inc()
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(model_err)) from model_err
    except Exception as exc:
        PREDICTION_COUNTER.labels(status="error").inc()
        logger.exception("Inference processing crashed unexpectedly for %s: %s", file.filename, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference execution failed: {exc}",
        ) from exc
