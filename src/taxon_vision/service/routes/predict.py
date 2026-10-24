# SPDX-FileCopyrightText: 2026 Benjamin Förster
# SPDX-License-Identifier: MIT
"""Inference prediction endpoint."""

import io
import logging
import time
import uuid
from typing import Annotated, Any

import anyio
from fastapi import APIRouter, File, HTTPException, UploadFile, status
from PIL import Image

from taxon_vision.config import get_settings
from taxon_vision.inference.cam import generate_cam_overlay
from taxon_vision.monitoring.telemetry import PREDICTION_COUNTER
from taxon_vision.service.inference import InvalidImageError, ModelNotReadyError, load_taxa_catalog, run_prediction
from taxon_vision.service.schemas import ConformalRoutingTier, PredictionResponse, TaxonPrediction
from taxon_vision.service.storage import persist_flagged_observation

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Inference"])


def _resolve_routing_tier(
    raw_tier: str,
    pred_set: list[str],
    confidence: float,
    min_confidence: float,
    k_max: int,
) -> ConformalRoutingTier:
    """Safely map raw prediction tier string or infer from prediction set metrics."""
    try:
        return ConformalRoutingTier(raw_tier)
    except ValueError:
        cardinality = len(pred_set)
        if cardinality == 0:
            return ConformalRoutingTier.ABSTENTION
        if cardinality == 1:
            return (
                ConformalRoutingTier.DECISIVE if confidence >= min_confidence else ConformalRoutingTier.SEVERE_AMBIGUITY
            )
        if cardinality <= k_max:
            return ConformalRoutingTier.MODERATE_AMBIGUITY
        return ConformalRoutingTier.SEVERE_AMBIGUITY


def _build_live_prediction(
    pred: dict[str, Any],
    latency_ms: float,
    cam_heatmap_base64: str | None = None,
) -> PredictionResponse:
    """Construct structured response from successful inference output.

    Why:
        Transforms raw array-based and dictionary inference outputs into strongly-typed,
        validated Pydantic schemas. Correlates top predictions and conformal set member
        names against authoritative catalog metadata to provide clients with canonical
        taxon IDs, scientific names, common names, and triage recommendations.

    How:
        Extracts top-1 predicted taxon, parses confidence score, scans candidate names
        in the conformal set, distributes uniform candidate likelihood among secondary
        conformal items, flags conformal ambiguity if the set size exceeds the threshold
        (|C| > 3 or |C| == 0), and packages the latency measurement into a PredictionResponse.

    Complexity:
        O(K) where K is the cardinality of the conformal candidate set.

    Args:
        pred: Raw prediction dictionary from the inference engine.
        latency_ms: Measured end-to-end execution latency in milliseconds.
        cam_heatmap_base64: Optional base64-encoded CAM saliency overlay image.

    Returns:
        Validated PredictionResponse payload.
    """
    catalog = load_taxa_catalog()
    catalog_map = {t["scientific_name"]: t for t in catalog}

    sci_name = str(pred["scientific_name"])
    matched_taxon = catalog_map.get(sci_name, {})
    taxon_id = int(matched_taxon.get("id", 0))
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

    routing_tier = _resolve_routing_tier(
        raw_tier=str(pred.get("routing_tier", "")),
        pred_set=pred_set,
        confidence=confidence,
        min_confidence=settings.conformal.min_confidence_threshold,
        k_max=settings.conformal.k_max,
    )

    return PredictionResponse(
        top_prediction=top,
        top_candidates=candidates,
        conformal_prediction_set=pred_set,
        routing_tier=routing_tier,
        is_conformal_ambiguous=len(pred_set) > settings.conformal.k_max or len(pred_set) == 0,
        is_ood_flagged=is_ood,
        requires_human_review=bool(pred["requires_human_review"]),
        latency_ms=round(latency_ms, 2),
        cam_heatmap_base64=cam_heatmap_base64,
    )


@router.post("/predict", response_model=PredictionResponse)
@router.post("/api/v1/predict", response_model=PredictionResponse)
async def predict_species(
    file: Annotated[UploadFile, File(...)],
    explain: bool = False,
) -> PredictionResponse:
    """Classify species from an uploaded image with conformal uncertainty guarantees.

    Why:
        Serving visual inference in production requires strict non-blocking concurrency.
        Executing image decoding, tensor normalization, and PyTorch/ONNX matrix operations
        directly on the asyncio loop thread blocks all concurrent requests. Furthermore,
        silently catching execution exceptions and returning dummy predictions masks critical
        failures (corrupt model weights, missing artefacts, GPU driver crashes, OOM errors).
        This endpoint offloads synchronous inference to a threadpool worker, measures accurate
        wall-clock latency, persists out-of-distribution or ambiguous records for subsequent
        human review, and reports each failure class with its proper HTTP status.

    How:
        1. Reads raw bytes asynchronously from the multipart file upload.
        2. Validates payload non-emptiness and size constraints.
        3. Dispatches `run_prediction` to AnyIO's threadpool worker via `run_sync`.
        4. If flagged as OOD or requiring human review, persists observation metadata and embeddings.
        5. If `explain=True`, computes forward-only CAM attribution for top-1 taxon and encodes as base64 PNG.
        6. On success: increments `taxon_predictions_total{status="success"}` and returns `PredictionResponse`.
        7. On an undecodable/empty image (`InvalidImageError`): logs a warning, increments `status="invalid_input"` and raises HTTP 422.
        8. On missing model artefacts or unexpected runtime inference failure: logs exception, increments counter, and raises HTTP 503.

    Complexity:
        O(1) async dispatch plus O(K) schema serialization.

    Args:
        file: Multipart uploaded image file.
        explain: Optional boolean flag to request forward-only CAM saliency visualization.

    Returns:
        Structured prediction response containing top candidates, conformal sets, and triage flags.

    Raises:
        HTTPException: HTTP 413 for payload too large, 422 for undecodable/empty images, 503 if model inference fails.
    """
    t0 = time.perf_counter()
    content = await file.read()
    settings = get_settings()

    if len(content) == 0:
        logger.warning("Rejected empty image upload for %s", file.filename)
        PREDICTION_COUNTER.labels(status="invalid_input").inc()
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="Invalid or corrupted image upload.",
        )

    if len(content) > settings.service.max_image_upload_bytes:
        PREDICTION_COUNTER.labels(status="invalid_input").inc()
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Image upload exceeds maximum allowed size of {settings.service.max_image_upload_bytes} bytes.",
        )

    try:
        pred = await anyio.to_thread.run_sync(run_prediction, content)
        latency = (time.perf_counter() - t0) * 1000.0

        if pred.get("is_ood") or pred.get("requires_human_review"):
            obs_id = str(uuid.uuid4())
            persist_flagged_observation(obs_id, pred, embedding=pred.get("embedding"))

        cam_base64: str | None = None
        if explain:
            try:
                img = Image.open(io.BytesIO(content)).convert("RGB")
                top1_idx = pred.get("top1_idx")
                cam_base64 = await anyio.to_thread.run_sync(generate_cam_overlay, img, None, top1_idx, 0.5)
            except Exception as cam_err:
                logger.warning("CAM overlay generation failed for %s: %s", file.filename, cam_err)
                cam_base64 = None

        PREDICTION_COUNTER.labels(status="success").inc()
        return _build_live_prediction(pred, latency, cam_base64)
    except InvalidImageError as img_err:
        logger.warning("Rejected undecodable upload %s: %s", file.filename, img_err)
        PREDICTION_COUNTER.labels(status="invalid_input").inc()
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="Invalid or corrupted image upload.",
        ) from img_err
    except ModelNotReadyError as model_err:
        logger.exception("Model not ready for %s: %s", file.filename, model_err)
        PREDICTION_COUNTER.labels(status="unavailable").inc()
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model inference failed.",
        ) from model_err
    except Exception as exc:
        PREDICTION_COUNTER.labels(status="error").inc()
        logger.exception("Inference processing crashed unexpectedly for %s: %s", file.filename, exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model inference failed.",
        ) from exc
