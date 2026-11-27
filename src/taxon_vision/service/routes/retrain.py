# SPDX-FileCopyrightText: 2026 Benjamin Förster
# SPDX-License-Identifier: MIT
"""FastAPI route handler bridging Grafana Alertmanager webhooks to Airflow 3 DAG orchestrator.

Summary:
    Exposes the authenticated `/api/v1/retrain/webhook` endpoint that validates Grafana Alertmanager payloads, enforces cooldown gating, and delegates Airflow 3 DAG run dispatching to `taxon_vision.service.retrain_dispatcher`.

Why (Architectural Rationale & Trade-offs):
    Keeping HTTP routing, Bearer authentication, and Prometheus outcome accounting separated from Airflow JWT/HTTP transport preserves thin routers whilst protecting the cluster from drift alert storms via deterministic cooldown bounds.

How (Algorithmic Mechanism & Invariants):
    Authenticates callers via constant-time token comparison against `TAXON_RETRAIN_WEBHOOK_SECRET`, filters actionable firing alerts via `extract_firing_retrain_alert`, acquires cooldown via `acquire_retrain_cooldown`, dispatches via `dispatch_airflow_retraining`, and increments `taxon_retrain_triggers_total` strictly by terminal outcome.
"""

from __future__ import annotations

import os
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Security, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from taxon_vision.monitoring.telemetry import RETRAIN_TRIGGERS_TOTAL
from taxon_vision.service.retrain_dispatcher import (
    acquire_retrain_cooldown,
    dispatch_airflow_retraining,
    extract_firing_retrain_alert,
)
from taxon_vision.service.routes.drift import verify_bearer_env_secret
from taxon_vision.service.schemas import GrafanaWebhookPayload, RetrainWebhookResponse

router = APIRouter(prefix="/api/v1/retrain", tags=["Retraining Relay"])
security = HTTPBearer(auto_error=False)


def verify_retrain_token(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Security(security)],
) -> None:
    """Validate incoming Bearer token using fail-closed constant-time string comparison.

    Why:
        Prevents timing-attack side channels and rejects requests cleanly when `TAXON_RETRAIN_WEBHOOK_SECRET` is unconfigured, avoiding fallback to weak default tokens.

    How:
        Delegates constant-time comparison and fail-closed verification against `TAXON_RETRAIN_WEBHOOK_SECRET` to `verify_bearer_env_secret`.

    Args:
        credentials: Authorization header credentials.

    Raises:
        HTTPException: 503 if `TAXON_RETRAIN_WEBHOOK_SECRET` is unset, or 401 if bearer credentials are missing or invalid.
    """
    verify_bearer_env_secret(
        credentials,
        "TAXON_RETRAIN_WEBHOOK_SECRET",
        "Invalid or missing retrain webhook authorization token",
    )


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
