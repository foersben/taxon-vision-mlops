# SPDX-FileCopyrightText: 2026 Benjamin Förster
# SPDX-License-Identifier: MIT
"""Health and telemetry routes."""

from typing import Any

from fastapi import APIRouter, HTTPException, status
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from starlette.responses import Response

from taxon_vision.service.inference import check_readiness, get_classifier, load_taxa_catalog

router = APIRouter()


@router.get("/health")
def health() -> dict[str, str]:
    """Perform a lightweight liveness check confirming service availability.

    Why:
        Orchestration platforms (Docker Compose, Kubernetes kubelet, systemd) require a rapid, zero-side-effect probe to determine process liveness and container readiness. Executing heavy model checks or I/O on the liveness probe risks false-positive pod restarts under transient traffic spikes. A decoupled, lightweight endpoint satisfies orchestration SLAs without adding CPU or GPU contention.

    How:
        Returns an immediate JSON payload with operational status "healthy" and the service identifier string, bypassing disk access, database lookups, and model inference.

    Complexity:
        O(1) memory lookup.

    Returns:
        Dictionary reporting health status and service identifier.
    """
    return {"status": "healthy", "service": "taxon-vision-mlops"}


@router.get("/health/ready")
def health_ready() -> dict[str, Any]:
    """Perform a readiness probe validating that the model can serve predictions.

    Why:
        Kubernetes readiness probes require a dedicated endpoint to ensure traffic is only routed to pods that can actually answer requests. A probe that unconditionally reports ready would send traffic to instances with a missing or incompatible model artefact.

    How:
        Loads the taxa catalogue and the serving model via `check_readiness()`.
        If the catalog is empty or the classifier fails to initialize, raises HTTPException 503.
        Otherwise returns a dictionary with operational readiness details.

    Complexity:
        O(1) cached dependency probe.

    Returns:
        Dictionary with readiness status, service identifier, and check results.

    Raises:
        HTTPException: HTTP 503 if the service is not ready.
    """
    try:
        taxa = load_taxa_catalog()
        if not taxa:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Service not ready: taxa catalog is empty",
            )
        get_classifier()
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Service not ready: {exc}",
        ) from exc

    ready, checks = check_readiness()
    if not ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service not ready",
        )
    return {
        "status": "ready",
        "service": "taxon-vision-mlops",
        "checks": checks,
    }


@router.get("/metrics")
def metrics() -> Response:
    """Expose Prometheus telemetry metrics for scraping.

    Why:
        Observability in production MLOps requires real-time insight into prediction throughput, inference latency distributions, and epistemic uncertainty referral rates. Prometheus pull-based scraping provides decoupled metric collection for Grafana dashboards and alerting rules without introducing external telemetry dependencies into the inference path.

    How:
        Invokes Prometheus client registry `generate_latest()`, serializing all registered counters (e.g. `taxon_predictions_total`), histograms (`taxon_inference_latency_seconds`), and gauges (`taxon_human_referrals_active`) into the standard Prometheus exposition format.

    Complexity:
        O(N) where N is the total number of metric collectors and metric series.

    Returns:
        HTTP Response containing formatted metric samples in Prometheus exposition format.
    """
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
