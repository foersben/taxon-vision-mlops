# SPDX-FileCopyrightText: 2026 Benjamin Förster
# SPDX-License-Identifier: MIT
"""Health and telemetry routes."""

from fastapi import APIRouter
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from starlette.responses import Response

router = APIRouter()


@router.get("/health")
def health() -> dict[str, str]:
    """Perform a lightweight liveness check confirming service availability.

    Why:
        Orchestration platforms (Docker Compose, Kubernetes kubelet, systemd) require a
        rapid, zero-side-effect probe to determine process liveness and container readiness.
        Executing heavy model checks or I/O on the liveness probe risks false-positive pod
        restarts under transient traffic spikes. A decoupled, lightweight endpoint satisfies
        orchestration SLAs without adding CPU or GPU contention.

    How:
        Returns an immediate JSON payload with operational status "healthy" and the service
        identifier string, bypassing disk access, database lookups, and model inference.

    Returns:
        Dictionary reporting health status and service identifier.
    """
    return {"status": "healthy", "service": "taxon-vision-mlops"}


@router.get("/metrics")
def metrics() -> Response:
    """Expose Prometheus telemetry metrics for scraping.

    Why:
        Observability in production MLOps requires real-time insight into prediction throughput,
        inference latency distributions, and epistemic uncertainty referral rates. Prometheus
        pull-based scraping provides decoupled metric collection for Grafana dashboards and
        alerting rules without introducing external telemetry dependencies into the inference path.

    How:
        Invokes Prometheus client registry `generate_latest()`, serializing all registered
        counters (e.g. `taxon_predictions_total`), histograms (`taxon_inference_latency_seconds`),
        and gauges (`taxon_human_referrals_active`) into the standard Prometheus exposition format.

    Returns:
        HTTP Response containing formatted metric samples in Prometheus exposition format.
    """
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
