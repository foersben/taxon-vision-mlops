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

import hmac
import logging
import os
import threading
import time
from typing import Annotated, NamedTuple

import httpx
from fastapi import APIRouter, Depends, HTTPException, Security, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from taxon_vision.monitoring.telemetry import RETRAIN_TRIGGERS_TOTAL
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
    """Validate incoming Bearer token using constant-time string comparison.

    Why:
        Prevent timing-attack side channels whilst ensuring only authorized Grafana Alertmanager instances can trigger retraining workflows.

    How:
        Extracts credentials from the HTTP Authorization header and compares them against TAXON_RETRAIN_WEBHOOK_SECRET via hmac.compare_digest.

    Args:
        credentials: Authorization header credentials.

    Raises:
        HTTPException: If credentials are missing or token does not match the configured secret.
    """
    expected_secret = os.getenv("TAXON_RETRAIN_WEBHOOK_SECRET", "dev-secret-token")
    if credentials is None or not hmac.compare_digest(credentials.credentials, expected_secret):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing retrain webhook authorization token",
            headers={"WWW-Authenticate": "Bearer"},
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


async def dispatch_airflow_retraining(alert_name: str) -> str:
    """Dispatch DAG run execution request to the Airflow 3 REST API.

    Why:
        Triggers the containerized retraining pipeline asynchronously without blocking the FastAPI event loop.

    How:
        Dispatches HTTP POST request with alert payload configuration to Airflow's REST API endpoint, with fallback to legacy endpoint and offline simulation support.

    Args:
        alert_name: Name of the triggering Grafana alert.

    Returns:
        Dispatched DAG run identifier.

    Raises:
        HTTPException: If Airflow API server is unreachable or returns an error response.
    """
    airflow_url = os.getenv("TAXON_AIRFLOW_URL", "http://airflow-apiserver:8080").rstrip("/")
    airflow_user = os.getenv("TAXON_AIRFLOW_USER", "admin")
    airflow_pass = os.getenv("TAXON_AIRFLOW_PASSWORD", "admin")
    dag_id = "model_retraining"

    if os.getenv("TAXON_AIRFLOW_SIMULATION", "false").lower() in ("true", "1"):
        simulated_id = f"simulated-run-{int(time.time())}"
        logger.info("Airflow simulation mode active; recorded simulated run %s", simulated_id)
        return simulated_id

    endpoint = f"{airflow_url}/api/v2/dags/{dag_id}/dagRuns"
    payload = {"conf": {"trigger_source": "grafana", "alert": alert_name}}

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(endpoint, json=payload, auth=(airflow_user, airflow_pass))
            if resp.status_code in (status.HTTP_200_OK, status.HTTP_201_CREATED):
                data = resp.json()
                return str(data.get("dag_run_id") or data.get("id") or "dispatched")

            v1_endpoint = f"{airflow_url}/api/v1/dags/{dag_id}/dagRuns"
            resp_v1 = await client.post(v1_endpoint, json=payload, auth=(airflow_user, airflow_pass))
            if resp_v1.status_code in (status.HTTP_200_OK, status.HTTP_201_CREATED):
                data_v1 = resp_v1.json()
                return str(data_v1.get("dag_run_id") or data_v1.get("id") or "dispatched")

            logger.error("Airflow API returned unexpected status %d: %s", resp.status_code, resp.text)
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"Airflow API server returned status code {resp.status_code}",
            )
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
