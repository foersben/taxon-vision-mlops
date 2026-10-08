# SPDX-FileCopyrightText: 2026 Benjamin Förster
# SPDX-License-Identifier: MIT
"""Prometheus telemetry metrics.

This module exposes real-time telemetry metrics via the embedded Prometheus client for monitoring model performance, operational load, and decision routing behavior.

Prometheus collects metrics via HTTP scrapes from a dedicated `/metrics` endpoint exposed by the application (typically integrated into the FastAPI or Streamlit server instance).

Exposed metrics:
    - `taxon_predictions_total[status]`: Counter tracking total prediction calls grouped by status (e.g., 'auto_assigned', 'rejected', 'needs_review'). Supports debugging routing logic and measuring automation rates.
    - `taxon_inference_latency_seconds`: Histogram measuring end-to-end inference duration (preprocessing to final decision), aiding latency optimization and SLA compliance monitoring.
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

from prometheus_client import Counter, Gauge, Histogram

PREDICTION_COUNTER = Counter("taxon_predictions_total", "Total species predictions evaluated", ["status"])
LATENCY_HISTOGRAM = Histogram("taxon_inference_latency_seconds", "Inference duration in seconds")
REFERRAL_RATE = Gauge("taxon_human_referrals_active", "Number of observations routed to human review")
