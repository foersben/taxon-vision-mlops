# SPDX-FileCopyrightText: 2026 Benjamin Förster
# SPDX-License-Identifier: MIT
"""Shared upload validation, Prometheus telemetry, and PostgreSQL recording for prediction routes.

Summary:
    Centralises multipart image payload validation and post-inference observability (Prometheus latency/counter metrics, human referral triage persistence, and asynchronous TelemetryStore embedding writes) across REST JSON and HTMX UI inference routes.

Why (Architectural Rationale & Trade-offs):
    Duplicating upload validation, Prometheus instrumentation, and PostgreSQL TelemetryStore writes across `predict.py` and `views.py` causes observability drift—historically leaving HTMX web uploads invisible to Grafana p95 latency panels and drift detection windows. Extracting a single shared helper guarantees metric and telemetry parity whilst keeping each FastAPI route handler below the cognitive complexity ceiling (< 15) and isolating non-blocking BackgroundTasks dispatch.

How (Algorithmic Mechanism & Invariants):
    `read_validated_upload` enforces non-empty and byte-size limits before image decoding, incrementing `taxon_predictions_total{status="invalid_input"}` on rejection. `record_prediction_outcome` observes `LATENCY_HISTOGRAM` strictly after inference succeeds, increments `REFERRAL_RATE` and dispatches `handle_flagged_prediction_if_needed` when OOD or ambiguous, constructs an immutable `PredictionRecord` with pre-normalised feature embeddings, queues `TelemetryStore.record` on `BackgroundTasks`, and increments `taxon_predictions_total{status="success"}`.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from typing import Any

from fastapi import BackgroundTasks, HTTPException, UploadFile, status

from taxon_vision.config.loader import get_settings
from taxon_vision.monitoring.telemetry import LATENCY_HISTOGRAM, PREDICTION_COUNTER, REFERRAL_RATE
from taxon_vision.service.storage import (
    ContributorConsent,
    FlaggedObservationPayload,
    handle_flagged_prediction_if_needed,
)
from taxon_vision.service.telemetry_store import PredictionRecord, get_telemetry_store

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PredictionTelemetryContext:
    """Typed payload encapsulating a completed inference outcome for shared telemetry and triage recording.

    Why:
        Bundles prediction outputs, raw image bytes, wall-clock duration, and contributor open-license consent into a frozen value object so `record_prediction_outcome` complies with the Rule 08 parameter arity limit (<= 4 positional arguments).

    How:
        Constructed by `/api/v1/predict` and `/ui/predict-htmx` handlers immediately after synchronous threadpool inference returns without error.

    Attributes:
        prediction_result: Structured prediction dictionary emitted by `run_prediction`.
        image_bytes: Raw validated upload payload bytes.
        duration_sec: Measured inference wall-clock duration in seconds.
        consent: Optional open-license opt-in and attribution metadata from the contributor.
    """

    prediction_result: dict[str, Any]
    image_bytes: bytes
    duration_sec: float
    consent: ContributorConsent | None = None


async def read_validated_upload(file: UploadFile) -> bytes:
    """Read and validate an uploaded multipart image payload against emptiness and size ceilings.

    Why:
        Both JSON and HTMX inference routes must reject empty or oversized uploads identically before spawning threadpool image decoders, recording `invalid_input` rejections uniformly in Prometheus.

    How:
        Reads the asynchronous `UploadFile` stream, checks zero-length and `service.max_image_upload_bytes` bounds, increments `PREDICTION_COUNTER{status="invalid_input"}` on violation, and raises HTTP 422 or HTTP 413.

    Args:
        file: FastAPI `UploadFile` handle from the incoming multipart request.

    Returns:
        Raw non-empty upload byte payload within the configured size limit.

    Raises:
        HTTPException: HTTP 422 if the upload is empty, or HTTP 413 if it exceeds `max_image_upload_bytes`.
    """
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

    return content


async def record_prediction_outcome(
    ctx: PredictionTelemetryContext,
    background_tasks: BackgroundTasks,
) -> str:
    """Record Prometheus metrics, triage persistence, and asynchronous TelemetryStore writes for a prediction.

    Why:
        Ensures every successful classification—whether submitted via the JSON REST API or the HTMX operator cockpit—contributes identically to Prometheus SLO histograms, referral gauges, and PostgreSQL drift windows without blocking the asyncio event loop.

    How:
        Observes `LATENCY_HISTOGRAM`, checks OOD/review flags to increment `REFERRAL_RATE` and persist flagged crops via `handle_flagged_prediction_if_needed`, appends a `PredictionRecord` write to `background_tasks`, and increments `PREDICTION_COUNTER{status="success"}`.

    Args:
        ctx: Frozen `PredictionTelemetryContext` holding prediction results, bytes, duration, and consent.
        background_tasks: FastAPI `BackgroundTasks` scheduler for non-blocking I/O dispatch.

    Returns:
        Generated unique `observation_id` UUID string for the recorded prediction.
    """
    settings = get_settings()
    pred = ctx.prediction_result
    latency_ms = ctx.duration_sec * 1000.0
    LATENCY_HISTOGRAM.observe(ctx.duration_sec)

    obs_id = str(uuid.uuid4())
    if pred.get("is_ood") or pred.get("requires_human_review"):
        REFERRAL_RATE.inc()
        await handle_flagged_prediction_if_needed(
            FlaggedObservationPayload(
                observation_id=obs_id,
                prediction_result=pred,
                embedding=pred.get("embedding"),
                image_bytes=ctx.image_bytes,
                consent=ctx.consent,
            ),
            background_tasks=background_tasks,
        )

    rec = PredictionRecord(
        observation_id=obs_id,
        extractor=settings.model.default_extractor,
        model_version=settings.service.version,
        top1_taxon=str(pred.get("scientific_name")),
        top1_confidence=float(pred.get("confidence", 0.0)),
        conformal_set=[str(s) for s in pred.get("conformal_set", [])],
        ood_energy=float(pred["ood_energy_score"]) if pred.get("ood_energy_score") is not None else None,
        is_ood=bool(pred.get("is_ood", False)),
        requires_review=bool(pred.get("requires_human_review", False)),
        latency_ms=round(latency_ms, 2),
        embedding=pred.get("embedding"),
    )
    background_tasks.add_task(get_telemetry_store().record, rec)
    PREDICTION_COUNTER.labels(status="success").inc()
    return obs_id
