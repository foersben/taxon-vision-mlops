# SPDX-FileCopyrightText: 2026 Benjamin Förster
# SPDX-License-Identifier: MIT
"""Prometheus telemetry metrics.

This module exposes real-time telemetry metrics via the embedded Prometheus client for monitoring model performance, operational load, and decision routing behavior.

Prometheus collects metrics via HTTP scrapes from a dedicated `/metrics` endpoint exposed by the application (typically integrated into the FastAPI or Streamlit server instance).

Exposed metrics:
    - `taxon_predictions_total[status]`: Counter tracking total prediction calls grouped by status (e.g., 'auto_assigned', 'rejected', 'needs_review'). Supports debugging routing logic and measuring automation rates.
    - `taxon_inference_latency_seconds`: Histogram measuring end-to-end inference duration (preprocessing to final decision), aiding latency optimization and SLO compliance monitoring.
    - `taxon_human_referrals_active`: Gauge reporting current count of observations deferred to human experts, serving as a proxy for automation boundary and fallback load.

Metrics are automatically registered with the global Prometheus registry on import, enabling immediate scraping without manual registration steps. No background threads or server processes are started by this module; Prometheus relies on external scraping agents to poll the metrics endpoint.

Usage:
    - No explicit initialization is required. Import this module once during application startup to register the metric collectors.
    - Application code should call `.inc()` or `.observe()` on the metric objects at appropriate decision points (e.g., after each inference call).

Example usage in prediction pipeline:
    >>> from taxon_vision.monitoring import telemetry
    >>> telemetry.PREDICTION_COUNTER.labels(status="auto_assigned").inc()
    >>> telemetry.LATENCY_HISTOGRAM.observe(inference_duration)

Prometheus server configuration (external):
    - Configure Prometheus to scrape `http://<app-host>:<app-port>/metrics`
    - Set appropriate scrape intervals (e.g., 15-60s) based on desired metric granularity

Example prometheus.yml scrape config:
    scrape_configs:
      - job_name: 'taxon-vision'
        metrics_path: '/metrics'
        static_configs:
          - targets: ['<app-host>:8000']

External monitoring tools (e.g., Grafana) can visualize these metrics by querying Prometheus query language (PromQL).
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Iterator
from typing import TYPE_CHECKING

from prometheus_client import REGISTRY, Counter, Gauge, Histogram
from prometheus_client.core import GaugeMetricFamily

if TYPE_CHECKING:
    from taxon_vision.service.telemetry_store import DriftRunRecord
logger = logging.getLogger(__name__)

PREDICTION_COUNTER = Counter("taxon_predictions_total", "Total species predictions evaluated", ["status"])
LATENCY_HISTOGRAM = Histogram("taxon_inference_latency_seconds", "Inference duration in seconds")
REFERRAL_RATE = Gauge("taxon_human_referrals_active", "Number of observations routed to human review")
TELEMETRY_WRITE_FAILURES = Counter(
    "taxon_telemetry_write_failures_total",
    "Total database write failures when persisting prediction telemetry",
)
RETRAIN_TRIGGERS_TOTAL = Counter(
    "taxon_retrain_triggers_total",
    "Total model retraining triggers evaluated by the relay",
    ["source", "outcome"],
)


class DriftMetricsCollector:
    """Custom Prometheus metric collector for persistent statistical drift telemetry.

    Summary:
        Dynamically extracts and serialises the most recent drift evaluation run from the telemetry repository on Prometheus scrape requests.

    Why:
        In multi-worker ASGI environments (e.g. Uvicorn with multiple worker processes), in-memory Prometheus Gauges are isolated per process, leading to inconsistent metric oscillations across scrape requests. Decoupling drift telemetry from process memory and querying persistent relational storage guarantees consistent metric states across all ASGI worker processes. A 30-second TTL cache prevents redundant database transactions during high-frequency scrape intervals.

    How:
        1. Maintains an in-memory cache of the latest DriftRunRecord alongside a monotonic timestamp.
        2. On `collect()`, acquires a thread lock and checks whether the cache has expired (elapsed time > cache_ttl_seconds).
        3. If expired, queries `store.latest_drift_run()` via lazy acquisition of the TelemetryStore singleton to prevent circular imports.
        4. Yields five GaugeMetricFamily objects: W1 distance, W1 threshold, MMD² permutation p-value, drift detected boolean flag, and last execution timestamp.

    Attributes:
        _cache_ttl_seconds: Duration in seconds to retain queried drift records before re-querying storage.
        _last_scrape_time: Monotonic timestamp of the last successful scrape.
        _cached_record: Cached latest drift evaluation record.
        _lock: Thread lock to ensure thread-safe access to shared state.
    """

    def __init__(self, cache_ttl_seconds: float = 30.0) -> None:
        """Initialise drift collector with a bounded cache TTL.

        Args:
            cache_ttl_seconds: Duration in seconds to retain queried drift records before re-querying storage.
        """
        self._cache_ttl_seconds = cache_ttl_seconds
        self._last_scrape_time: float = 0.0
        self._cached_record: DriftRunRecord | None = None
        self._lock = threading.Lock()

    def collect(self) -> Iterator[GaugeMetricFamily]:
        """Collect and yield drift telemetry gauges on Prometheus scrape.

        Returns:
            Generator yielding GaugeMetricFamily objects for active drift metrics.
        """
        now = time.monotonic()
        with self._lock:
            if now - self._last_scrape_time > self._cache_ttl_seconds or self._cached_record is None:
                try:
                    from taxon_vision.service.telemetry_store import get_telemetry_store

                    store = get_telemetry_store()
                    self._cached_record = store.latest_drift_run()
                    self._last_scrape_time = now
                except Exception:
                    logger.exception("Failed to query latest drift run from telemetry store for Prometheus exposition")
            record = self._cached_record
        if record is None:
            return
        yield GaugeMetricFamily(
            "taxon_drift_w1_distance",
            "Observed 1D Wasserstein-1 distance on embedding norms from latest drift run",
            value=record.w1_distance,
        )
        yield GaugeMetricFamily(
            "taxon_drift_w1_threshold",
            "Calibrated 1D Wasserstein-1 alert threshold from latest drift run",
            value=record.w1_threshold,
        )
        if record.mmd2_p_value is not None:
            yield GaugeMetricFamily(
                "taxon_drift_mmd2_p_value",
                "Multivariate MMD squared permutation test p-value from latest drift run",
                value=record.mmd2_p_value,
            )
        yield GaugeMetricFamily(
            "taxon_drift_detected",
            "Binary indicator flag whether drift was detected in latest run (1.0 = true, 0.0 = false)",
            value=1.0 if record.drift_detected else 0.0,
        )
        if record.created_at is not None:
            yield GaugeMetricFamily(
                "taxon_drift_last_run_timestamp_seconds",
                "UTC UNIX timestamp in seconds of the latest drift run execution",
                value=record.created_at.timestamp(),
            )


# Auto-register the collector with the default Prometheus registry
DRIFT_COLLECTOR = DriftMetricsCollector()
try:
    REGISTRY.register(DRIFT_COLLECTOR)
except Exception:
    # Avoid duplicate timeseries exception during test reloading
    pass
