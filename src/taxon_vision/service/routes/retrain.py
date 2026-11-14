# SPDX-FileCopyrightText: 2026 Benjamin Förster
# SPDX-License-Identifier: MIT
"""FastAPI route handler bridging Grafana Alertmanager webhooks to Airflow 3 DAG orchestrator.

Summary:
    Exposes an authenticated webhook receiver endpoint that validates Grafana Alertmanager payloads, enforces a cooldown rate-limiting window, and dispatches an asynchronous execution request to the Airflow 3 REST API for the model retraining pipeline.

Why:
    Grafana Alertmanager dispatches static Bearer token webhooks upon detected statistical drift. Direct execution of retraining pipelines requires GPU coordination and stateful tracking in Airflow. The relay bridge decouples alerting from orchestration, shields the cluster from alert storms via strict cooldown bounds, and records telemetry metrics for Prometheus.

How:
    Authenticates callers via constant-time token comparison against TAXON_RETRAIN_WEBHOOK_SECRET, inspects alert items to verify that at least one alert is firing with label action=retrain, checks an atomic in-memory cooldown timestamp against TAXON_RETRAIN_COOLDOWN_SECONDS (default 6 hours), dispatches an asynchronous HTTP POST request to Airflow's REST API, and increments the Prometheus counter taxon_retrain_triggers_total.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import Annotated, NamedTuple

import httpx
from fastapi import APIRouter, Depends, HTTPException, Security, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from taxon_vision.monitoring.telemetry import RETRAIN_TRIGGERS_TOTAL
from taxon_vision.service.routes.drift import verify_bearer_env_secret
from taxon_vision.service.schemas import GrafanaWebhookPayload, RetrainWebhookResponse

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/retrain", tags=["Retraining Relay"])
security = HTTPBearer(auto_error=False)

_last_retrain_timestamp: float = 0.0
_cooldown_lock = threading.Lock()


def reset_retrain_cooldown() -> None:
    """Reset the in-memory cooldown timestamp for deterministic test fixture execution."""
    global _last_retrain_timestamp
    with _cooldown_lock:
        _last_retrain_timestamp = 0.0


def verify_retrain_token(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Security(security)],
) -> None:
    """Validate incoming Bearer token using fail-closed constant-time string comparison.

    Why:
        Prevent timing-attack side channels and reject requests cleanly when TAXON_RETRAIN_WEBHOOK_SECRET is unconfigured, avoiding fallback to weak default tokens.

    How:
        Delegates constant-time comparison and fail-closed verification against TAXON_RETRAIN_WEBHOOK_SECRET to verify_bearer_env_secret.

    Args:
        credentials: Authorization header credentials.

    Raises:
        HTTPException: 503 if TAXON_RETRAIN_WEBHOOK_SECRET is unset, or 401 if bearer credentials are missing or invalid.
    """
    verify_bearer_env_secret(
        credentials,
        "TAXON_RETRAIN_WEBHOOK_SECRET",
        "Invalid or missing retrain webhook authorization token",
    )


def extract_firing_retrain_alert(payload: GrafanaWebhookPayload) -> str | None:
    """Extract the first firing alert name associated with an action=retrain instruction.

    Why:
        Grafana delivers aggregated alert batches where some alerts may be resolved or unrelated to automated retraining; only actionable firing alerts should trigger pipeline execution.

    How:
        Iterates across payload.alerts verifying status == 'firing' and labels['action'] == 'retrain', falling back to common_labels if batch-level labels are defined.

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


class CooldownAcquisitionResult(NamedTuple):
    """Structured outcome of checking and acquiring the retraining cooldown window.

    Attributes:
        acquired: Boolean indicator whether the retraining slot was acquired.
        remaining_seconds: Remaining cooldown duration in seconds if suppressed, otherwise 0.0.
    """

    acquired: bool
    remaining_seconds: float


def acquire_retrain_cooldown(cooldown_seconds: float) -> CooldownAcquisitionResult:
    """Check and atomically acquire the retraining cooldown slot.

    Why:
        Avoid parallel or redundant GPU retraining runs caused by repeated alert deliveries during the same distribution drift event.

    How:
        Acquires threading lock, calculates elapsed time since last trigger, and updates timestamp atomically if cooldown has expired.

    Args:
        cooldown_seconds: Minimum seconds required between consecutive retraining runs.

    Returns:
        CooldownAcquisitionResult containing acquired flag and remaining cooldown seconds.
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
        Parses JSON safely and prefers ``dag_run_id``, falling back to ``id`` or the literal ``"dispatched"`` sentinel when omitted.

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
        Airflow 3 FastAPI routes require OAuth2 Bearer tokens minted by ``POST /auth/token`` rather than HTTP Basic authentication; returning ``None`` on 404/405 allows controlled fallback to legacy Airflow 2 deployments whilst failing fast on invalid credentials.

    How:
        Dispatches JSON credentials via ``LoginBody`` schema, extracts ``access_token`` from ``200``/``201`` responses, returns ``None`` only when ``/auth/token`` is absent, and raises ``HTTPException`` (502) on authentication errors.

    Args:
        client: Reusable asynchronous HTTPX client session.
        airflow_url: Base URL of the Airflow API server without trailing slash.
        auth: Username and password tuple for token exchange.

    Returns:
        Signed JWT access token string, or ``None`` if the endpoint is unavailable (404/405).

    Raises:
        HTTPException: If credentials are rejected or the token payload omits ``access_token``.
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
        Airflow 3 ``TriggerDAGRunPostBody`` enforces an explicit ``logical_date`` key and Bearer JWT header, whereas legacy instances accept Basic authentication on ``/api/v1`` without ``logical_date``.

    How:
        Exchanges credentials for a Bearer JWT via ``_fetch_airflow_jwt``, dispatches to ``/api/v2/dags/model_retraining/dagRuns`` with ``logical_date: None`` (or falls back to ``/api/v1`` if no token endpoint exists), and extracts the run identifier via a single response check.

    Args:
        client: Reusable asynchronous HTTPX client session.
        airflow_url: Base URL of the Airflow API server without trailing slash.
        auth: Username and password tuple for Airflow authentication.
        conf_payload: Trigger provenance metadata passed under the DAG run ``conf`` key.

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
        Resolves connection parameters from environment variables, short-circuits cleanly if simulation mode is enabled, exchanges credentials for a Bearer token, and posts to the ``model_retraining`` DAG runs endpoint with legacy v1 fallback and fail-fast HTTP 502 error propagation.

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


@router.post(
    "/webhook",
    response_model=RetrainWebhookResponse,
    status_code=status.HTTP_200_OK,
    dependencies=[Depends(verify_retrain_token)],
    summary="Receive Grafana Alertmanager webhooks and dispatch Airflow retraining DAG",
)
async def retrain_webhook_endpoint(
    payload: GrafanaWebhookPayload,
) -> RetrainWebhookResponse:
    """Process incoming Grafana webhook notification and trigger model retraining if applicable.

    Why:
        Serves as the single operational gateway between telemetry alerting and retraining automation.

    How:
        Verifies token, filters for firing retrain actions, checks cooldown bounds, and dispatches the DAG.

    Args:
        payload: Inbound Grafana Alertmanager notification payload.

    Returns:
        Outcome response indicating whether retraining was triggered, suppressed, or ignored.
    """
    alert_name = extract_firing_retrain_alert(payload)
    if not alert_name:
        RETRAIN_TRIGGERS_TOTAL.labels(source="grafana", outcome="ignored").inc()
        return RetrainWebhookResponse(
            status="ignored",
            detail="No active firing alert with label action=retrain found in notification payload",
        )

    cooldown_seconds = float(os.getenv("TAXON_RETRAIN_COOLDOWN_SECONDS", "21600.0"))
    cooldown = acquire_retrain_cooldown(cooldown_seconds)
    if not cooldown.acquired:
        RETRAIN_TRIGGERS_TOTAL.labels(source="grafana", outcome="suppressed").inc()
        return RetrainWebhookResponse(
            status="suppressed",
            detail=f"Retraining suppressed by cooldown ({int(cooldown.remaining_seconds)}s remaining)",
        )

    try:
        dag_run_id = await dispatch_airflow_retraining(alert_name)
        RETRAIN_TRIGGERS_TOTAL.labels(source="grafana", outcome="triggered").inc()
        return RetrainWebhookResponse(
            status="triggered",
            detail=f"Successfully dispatched model_retraining DAG for alert '{alert_name}'",
            dag_run_id=dag_run_id,
        )
    except HTTPException:
        RETRAIN_TRIGGERS_TOTAL.labels(source="grafana", outcome="error").inc()
        raise
