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

    Returns:
        Dictionary reporting health status and service identifier.
    """
    return {"status": "healthy", "service": "taxon-vision-mlops"}


@router.get("/metrics")
def metrics() -> Response:
    """Expose Prometheus telemetry metrics for scraping.

    Returns:
        HTTP Response containing formatted metric samples in Prometheus exposition format.
    """
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
