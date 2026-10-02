# SPDX-FileCopyrightText: 2026 Benjamin Förster
# SPDX-License-Identifier: MIT
"""Prometheus telemetry metrics."""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

PREDICTION_COUNTER = Counter("taxon_predictions_total", "Total species predictions evaluated", ["status"])
LATENCY_HISTOGRAM = Histogram("taxon_inference_latency_seconds", "Inference duration in seconds")
REFERRAL_RATE = Gauge("taxon_human_referrals_active", "Number of observations routed to human review")
