# SPDX-FileCopyrightText: 2026 Benjamin Förster
# SPDX-License-Identifier: MIT
"""FastAPI route handlers for statistical drift monitoring and evaluation."""

from __future__ import annotations

import logging
import os
from typing import Annotated

import anyio
from fastapi import APIRouter, Depends, HTTPException, Security, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from taxon_vision.monitoring.drift_runner import evaluate_recent_drift
from taxon_vision.service.schemas import DriftEvaluationResponse, DriftRunSummary
from taxon_vision.service.telemetry_store import get_telemetry_store

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/drift", tags=["Drift Monitoring"])
security = HTTPBearer(auto_error=False)


def verify_internal_token(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Security(security)],
) -> None:
    """Validate internal bearer token for protected drift evaluation endpoints.

    Why:
        Prevent unauthenticated external callers from triggering CPU-intensive MMD² permutation cycles or generating unsolicited MLflow runs.

    How:
        Compares incoming bearer token against TAXON_INTERNAL_API_TOKEN environment variable. If the variable is unset (local dev/test), permits the request whilst logging a debug notice.

    Args:
        credentials: Authorization credentials extracted from HTTP headers.

    Raises:
        HTTPException: If the token is missing or does not match the configured secret.
    """
    expected_token = os.getenv("TAXON_INTERNAL_API_TOKEN")
    if not expected_token:
        return

    if credentials is None or credentials.credentials != expected_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing internal authorization token",
            headers={"WWW-Authenticate": "Bearer"},
        )


@router.post(
    "/evaluate",
    response_model=DriftEvaluationResponse,
    status_code=status.HTTP_200_OK,
    dependencies=[Depends(verify_internal_token)],
    summary="Trigger on-demand drift evaluation across recent predictions",
)
async def evaluate_drift_endpoint() -> DriftEvaluationResponse:
    """Execute end-to-end drift evaluation cycle over the latest prediction window.

    Why:
        Allows external orchestrators (Airflow DAGs, CI test suites, Grafana webhook alerts) to trigger statistical hypothesis testing without blocking the ASGI event loop.

    How:
        Offloads synchronous CPU-bound matrix mathematics (W1, MMD² permutations, PCA projections) to a worker thread via `anyio.to_thread.run_sync`.
    """
    try:
        verdict = await anyio.to_thread.run_sync(evaluate_recent_drift)
        return DriftEvaluationResponse(
            drift_detected=verdict.drift_detected,
            status=verdict.status,
            w1_distance=verdict.w1_distance,
            w1_threshold=verdict.w1_threshold,
            mmd2=verdict.mmd2,
            mmd2_p_value=verdict.mmd2_p_value,
            window_size=verdict.window_size,
            n_reference=verdict.n_reference,
            mlflow_run_id=verdict.mlflow_run_id,
            report_path=verdict.report_path,
        )
    except Exception as exc:
        logger.exception("Failed to execute on-demand drift evaluation")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal drift evaluation failed",
        ) from exc


@router.get(
    "/latest",
    response_model=DriftRunSummary | None,
    status_code=status.HTTP_200_OK,
    summary="Retrieve the most recent drift evaluation run outcome",
)
async def get_latest_drift_endpoint() -> DriftRunSummary | None:
    """Retrieve the latest drift run record from the persistent telemetry repository.

    Why:
        Enables web operator views and status dashboards to query the current distribution health without invoking heavy statistical re-computation.
    """
    store = get_telemetry_store()
    latest = await anyio.to_thread.run_sync(store.latest_drift_run)
    if latest is None:
        return None

    return DriftRunSummary(
        id=latest.id,
        created_at=latest.created_at.isoformat() if latest.created_at else None,
        window_size=latest.window_size,
        w1_distance=latest.w1_distance,
        w1_threshold=latest.w1_threshold,
        mmd2=latest.mmd2,
        mmd2_p_value=latest.mmd2_p_value,
        drift_detected=latest.drift_detected,
        mlflow_run_id=latest.mlflow_run_id,
    )
