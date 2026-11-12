# SPDX-FileCopyrightText: 2026 Benjamin Förster
# SPDX-License-Identifier: MIT
"""FastAPI route handlers for statistical drift monitoring and evaluation."""

from __future__ import annotations

import hmac
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


def verify_bearer_env_secret(
    credentials: HTTPAuthorizationCredentials | None,
    env_var: str,
    unauthorized_detail: str,
) -> None:
    """Validate incoming HTTP Bearer credentials against a configured environment secret using constant-time comparison.

    Why:
        Centralises fail-closed Bearer token verification across protected operational routes, preventing timing side-channels and ensuring unconfigured production deployments reject requests explicitly instead of falling back to permissive or static dev tokens.

    How:
        Reads the expected secret from ``env_var``, raises HTTP 503 if unset or empty, and performs a timing-safe ``hmac.compare_digest`` check yielding HTTP 401 on missing or mismatched credentials.

    Args:
        credentials: Parsed HTTP Bearer credentials extracted by FastAPI Security.
        env_var: Name of the environment variable supplying the expected shared secret.
        unauthorized_detail: Detail message emitted with HTTP 401 Unauthorized responses.

    Raises:
        HTTPException: 503 when ``env_var`` is unconfigured, or 401 when credentials are missing or invalid.
    """
    expected_secret = os.getenv(env_var, "").strip()
    if not expected_secret:
        logger.error("Protected endpoint denied request because %s is not configured.", env_var)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Authentication secret '{env_var}' is not configured on this server.",
        )

    if credentials is None or not hmac.compare_digest(credentials.credentials, expected_secret):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=unauthorized_detail,
            headers={"WWW-Authenticate": "Bearer"},
        )


def verify_internal_token(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Security(security)],
) -> None:
    """Validate internal bearer token for protected operational endpoints.

    Why:
        Prevent unauthenticated external callers from triggering CPU/GPU-intensive training runs or MMD² permutation cycles.

    How:
        Delegates fail-closed constant-time verification to ``verify_bearer_env_secret`` using ``TAXON_INTERNAL_API_TOKEN``.

    Args:
        credentials: Authorization credentials extracted from HTTP headers.

    Raises:
        HTTPException: If ``TAXON_INTERNAL_API_TOKEN`` is unset (503) or the bearer token is missing or invalid (401).
    """
    verify_bearer_env_secret(
        credentials,
        "TAXON_INTERNAL_API_TOKEN",
        "Invalid or missing internal authorization token",
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
