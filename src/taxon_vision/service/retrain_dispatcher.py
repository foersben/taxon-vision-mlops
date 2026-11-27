# SPDX-FileCopyrightText: 2026 Benjamin Förster
# SPDX-License-Identifier: MIT
"""Airflow 3 retraining dispatcher, cooldown lock, and Grafana alert filter helpers.

Summary:
    Encapsulates Grafana Alertmanager `action=retrain` alert filtering, thread-safe in-memory cooldown rate limiting, and asynchronous Airflow 3 OAuth2 JWT + v2/v1 DAG run dispatching.

Why (Architectural Rationale & Trade-offs):
    Extracting orchestration transport, authentication exchange, and alert cooldown state out of `routes/retrain.py` prevents HTTP router bloat, keeps the FastAPI handler thin (< 120 LOC), and allows isolated transport unit testing against `httpx.MockTransport` without importing ASGI router wiring.

How (Algorithmic Mechanism & Invariants):
    Scans `GrafanaWebhookPayload` alerts for firing `action=retrain` entries, enforces an atomic monotonic/epoch cooldown via `threading.Lock`, authenticates against Airflow 3 `/auth/token` (with controlled 404/405 fallback to legacy v1 Basic Auth), and posts `model_retraining` DAG run requests using non-blocking `httpx.AsyncClient`.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import NamedTuple

import httpx
from fastapi import HTTPException, status

from taxon_vision.service.schemas import GrafanaWebhookPayload

logger = logging.getLogger(__name__)

_last_retrain_timestamp: float = 0.0
_cooldown_lock = threading.Lock()


class CooldownAcquisitionResult(NamedTuple):
    """Structured outcome of checking and acquiring the retraining cooldown window.

    Attributes:
        acquired: Boolean indicator whether the retraining slot was acquired.
        remaining_seconds: Remaining cooldown duration in seconds if suppressed, otherwise 0.0.
    """

    acquired: bool
    remaining_seconds: float


def reset_retrain_cooldown() -> None:
    """Reset the in-memory cooldown timestamp for deterministic test fixture execution."""
    global _last_retrain_timestamp
    with _cooldown_lock:
        _last_retrain_timestamp = 0.0


def extract_firing_retrain_alert(payload: GrafanaWebhookPayload) -> str | None:
    """Extract the first firing alert name associated with an action=retrain instruction.

    Why:
        Grafana delivers aggregated alert batches where some alerts may be resolved or unrelated to automated retraining; only actionable firing alerts should trigger pipeline execution.

    How:
        Iterates across `payload.alerts` verifying `status == 'firing'` and `labels['action'] == 'retrain'`, falling back to `common_labels` if batch-level labels are defined.

    Args:
        payload: Parsed Grafana webhook payload.

    Returns:
        Alert name string if an active firing retrain alert is present, otherwise None.
    """
    for alert in payload.alerts:
        if alert.status == "firing" and alert.labels.get("action") == "retrain":
            return alert.labels.get("alertname", "unnamed_alert")

    if payload.status == "firing" and payload.common_labels.get("action") == "retrain":
        return payload.common_labels.get("alertname", "unnamed_alert")

    return None


def acquire_retrain_cooldown(cooldown_seconds: float) -> CooldownAcquisitionResult:
    """Check and atomically acquire the retraining cooldown slot.

    Why:
        Avoids parallel or redundant GPU retraining runs caused by repeated alert deliveries during the same distribution drift event.

    How:
        Acquires threading lock, calculates elapsed time since last trigger, and updates timestamp atomically if cooldown has expired.

    Args:
        cooldown_seconds: Minimum seconds required between consecutive retraining runs.

    Returns:
        `CooldownAcquisitionResult` containing acquired flag and remaining cooldown seconds.
    """
    global _last_retrain_timestamp
    now = time.time()
    with _cooldown_lock:
        elapsed = now - _last_retrain_timestamp
        if elapsed < cooldown_seconds:
            return CooldownAcquisitionResult(acquired=False, remaining_seconds=cooldown_seconds - elapsed)
        _last_retrain_timestamp = now
        return CooldownAcquisitionResult(acquired=True, remaining_seconds=0.0)


def _extract_dag_run_id(resp: httpx.Response) -> str:
    """Extract a normalised DAG run identifier from an Airflow REST response.

    Why:
        Centralises response payload parsing so v2 FastAPI and legacy v1 endpoints return a consistent identifier without duplicating dictionary lookups.

    How:
        Parses JSON safely and prefers `dag_run_id`, falling back to `id` or the literal `"dispatched"` sentinel when omitted.

    Args:
        resp: Successful HTTP response returned by the Airflow DAG run endpoint.

    Returns:
        Normalised DAG run identifier string.
    """
    data = resp.json()
    return str(data.get("dag_run_id") or data.get("id") or "dispatched")


async def _fetch_airflow_jwt(
    client: httpx.AsyncClient,
    airflow_url: str,
    auth: tuple[str, str],
) -> str | None:
    """Authenticate against the Airflow 3 token endpoint to obtain a Bearer JWT.

    Why:
        Airflow 3 FastAPI routes require OAuth2 Bearer tokens minted by `POST /auth/token` rather than HTTP Basic authentication; returning `None` on 404/405 allows controlled fallback to legacy Airflow 2 deployments whilst failing fast on invalid credentials.

    How:
        Dispatches JSON credentials via `LoginBody` schema, extracts `access_token` from `200`/`201` responses, returns `None` only when `/auth/token` is absent, and raises `HTTPException` (502) on authentication errors.

    Args:
        client: Reusable asynchronous HTTPX client session.
        airflow_url: Base URL of the Airflow API server without trailing slash.
        auth: Username and password tuple for token exchange.

    Returns:
        Signed JWT access token string, or `None` if the endpoint is unavailable (404/405).

    Raises:
        HTTPException: If credentials are rejected or the token payload omits `access_token`.
    """
    token_endpoint = f"{airflow_url}/auth/token"
    resp = await client.post(token_endpoint, json={"username": auth[0], "password": auth[1]})
    if resp.status_code in (status.HTTP_200_OK, status.HTTP_201_CREATED):
        token = resp.json().get("access_token")
        if not token:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="Airflow /auth/token response omitted access_token",
            )
        return str(token)

    if resp.status_code in (status.HTTP_404_NOT_FOUND, status.HTTP_405_METHOD_NOT_ALLOWED):
        return None

    logger.error("Airflow /auth/token failed with status %d: %s", resp.status_code, resp.text)
    raise HTTPException(
        status_code=status.HTTP_502_BAD_GATEWAY,
        detail=f"Airflow authentication returned status code {resp.status_code}",
    )


async def _dispatch_airflow_run(
    client: httpx.AsyncClient,
    airflow_url: str,
    auth: tuple[str, str],
    conf_payload: dict[str, str],
) -> str:
    """Submit a DAG run trigger payload to Airflow 3 v2 or legacy v1 REST APIs.

    Why:
        Airflow 3 `TriggerDAGRunPostBody` enforces an explicit `logical_date` key and Bearer JWT header, whereas legacy instances accept Basic authentication on `/api/v1` without `logical_date`.

    How:
        Exchanges credentials for a Bearer JWT via `_fetch_airflow_jwt`, dispatches to `/api/v2/dags/model_retraining/dagRuns` with `logical_date: None` (or falls back to `/api/v1` if no token endpoint exists), and extracts the run identifier via a single response check.

    Args:
        client: Reusable asynchronous HTTPX client session.
        airflow_url: Base URL of the Airflow API server without trailing slash.
        auth: Username and password tuple for Airflow authentication.
        conf_payload: Trigger provenance metadata passed under the DAG run `conf` key.

    Returns:
        Dispatched DAG run identifier.

    Raises:
        HTTPException: If Airflow returns a non-success HTTP status code.
    """
    dag_id = "model_retraining"
    token = await _fetch_airflow_jwt(client, airflow_url, auth)
    if token is not None:
        resp = await client.post(
            f"{airflow_url}/api/v2/dags/{dag_id}/dagRuns",
            json={"logical_date": None, "conf": conf_payload},
            headers={"Authorization": f"Bearer {token}"},
        )
    else:
        resp = await client.post(
            f"{airflow_url}/api/v1/dags/{dag_id}/dagRuns",
            json={"conf": conf_payload},
            auth=auth,
        )

    if resp.status_code in (status.HTTP_200_OK, status.HTTP_201_CREATED):
        return _extract_dag_run_id(resp)

    logger.error("Airflow API returned unexpected status %d: %s", resp.status_code, resp.text)
    raise HTTPException(
        status_code=status.HTTP_502_BAD_GATEWAY,
        detail=f"Airflow API server returned status code {resp.status_code}",
    )


async def dispatch_airflow_retraining(alert_name: str) -> str:
    """Dispatch DAG run execution request to the Airflow 3 REST API.

    Why:
        Triggers the containerised retraining pipeline asynchronously without blocking the FastAPI event loop whilst authenticating against Airflow 3's JWT security model.

    How:
        Resolves connection parameters from environment variables, short-circuits cleanly if simulation mode is enabled, exchanges credentials for a Bearer token, and posts to the `model_retraining` DAG runs endpoint with legacy v1 fallback and fail-fast HTTP 502 error propagation.

    Args:
        alert_name: Name of the triggering Grafana alert.

    Returns:
        Dispatched DAG run identifier.

    Raises:
        HTTPException: If Airflow API server is unreachable, rejects authentication, or returns an error response.
    """
    if os.getenv("TAXON_AIRFLOW_SIMULATION", "false").lower() in ("true", "1"):
        simulated_id = f"simulated-run-{int(time.time())}"
        logger.info("Airflow simulation mode active; recorded simulated run %s", simulated_id)
        return simulated_id

    airflow_url = os.getenv("TAXON_AIRFLOW_URL", "http://airflow-apiserver:8080").rstrip("/")
    airflow_user = os.getenv("TAXON_AIRFLOW_USER", "").strip()
    airflow_pass = os.getenv("TAXON_AIRFLOW_PASSWORD", "").strip()
    if not airflow_user or not airflow_pass:
        logger.error("Airflow API credentials (TAXON_AIRFLOW_USER / TAXON_AIRFLOW_PASSWORD) are not configured.")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Airflow orchestration credentials are not configured.",
        )
    conf_payload = {"trigger_source": "grafana", "alert": alert_name}

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            return await _dispatch_airflow_run(client, airflow_url, (airflow_user, airflow_pass), conf_payload)
    except httpx.RequestError as exc:
        logger.exception("Failed to connect to Airflow API server at %s", airflow_url)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Could not reach Airflow API server at {airflow_url}",
        ) from exc
