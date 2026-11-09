#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Benjamin Förster
# SPDX-License-Identifier: MIT
"""Synthetic Traffic Generator and Drift Evaluation Maintenance Utility.

Summary:
    Dispatches synthetic observation batches to the active TaxonVision FastAPI microservice and triggers downstream statistical drift evaluations to populate live Prometheus metrics and Grafana telemetry dashboards.

Why:
    Cold-started or newly deployed clusters operate with empty Prometheus timeseries and unpopulated Grafana panels, displaying 'No data' across operational and statistical views. This utility provides an automated, deterministic mechanism to warm Prometheus metric buffers, populate the PostgreSQL telemetry database, and verify end-to-end drift calculation pipelines without manual curl operations.

How:
    Iterates over sample JPEG images in `data/sample/`, formats multipart HTTP POST requests, transmits them to `/api/v1/predict` at a configurable throughput rate, queries the `/api/v1/drift/evaluate` endpoint to calculate Tier 1 Wasserstein-1 distance and Tier 2 MMD² permutation tests, and logs structured verification metrics.
"""

from __future__ import annotations

import argparse
import glob
import logging
import os
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("simulate_traffic")


@dataclass(frozen=True)
class SimulationConfig:
    """Structured parameter configuration for the traffic simulation routine.

    Summary:
        Encapsulates endpoints, batch volumes, and execution pacing parameters.
    """

    base_url: str
    request_count: int
    delay_ms: float
    trigger_drift: bool
    sample_dir: Path


@dataclass(frozen=True)
class SimulationReport:
    """Outcome report detailing request success rates and drift evaluation state.

    Summary:
        Summarises transmission counts, failures, latency bounds, and drift outcomes.
    """

    successful_requests: int
    failed_requests: int
    drift_status: str | None
    drift_detected: bool | None
    w1_distance: float | None


def parse_arguments(argv: list[str] | None = None) -> SimulationConfig:
    """Parse command-line arguments and construct a typed SimulationConfig instance.

    Args:
        argv: Optional argument string sequence (defaults to sys.argv[1:]).

    Returns:
        Validated SimulationConfig dataclass.
    """
    parser = argparse.ArgumentParser(
        description="Simulate inference traffic and trigger statistical drift evaluation.",
    )
    parser.add_argument(
        "--url",
        type=str,
        default=os.getenv("APP_URL", "http://localhost:8001"),
        help="Base URL of the running TaxonVision microservice (default: http://localhost:8001).",
    )
    parser.add_argument(
        "--count",
        type=lambda v: int(v.split("=")[-1]) if "=" in str(v) else int(v),
        default=60,
        help="Total number of prediction requests to dispatch (default: 60).",
    )
    parser.add_argument(
        "--delay-ms",
        type=float,
        default=10.0,
        help="Inter-request delay in milliseconds (default: 10.0).",
    )
    parser.add_argument(
        "--no-drift",
        action="store_true",
        help="Skip triggering the drift evaluation cycle upon request completion.",
    )
    parser.add_argument(
        "--sample-dir",
        type=str,
        default="data/sample",
        help="Directory containing reference sample JPEG images (default: data/sample).",
    )

    args = parser.parse_args(argv)
    return SimulationConfig(
        base_url=args.url.rstrip("/"),
        request_count=args.count,
        delay_ms=args.delay_ms,
        trigger_drift=not args.no_drift,
        sample_dir=Path(args.sample_dir),
    )


def load_sample_images(sample_dir: Path) -> list[bytes]:
    """Load sample JPEG images from disk into memory buffers.

    Why:
        Pre-loading avoids redundant disk I/O bottlenecks during high-frequency request transmission.

    Args:
        sample_dir: Path to directory containing sample JPEG files.

    Returns:
        List of raw JPEG byte buffers.

    Raises:
        FileNotFoundError: If no JPEG images are found in the target directory.
    """
    pattern = str(sample_dir / "*.jpg")
    file_paths = sorted(glob.glob(pattern))
    if not file_paths:
        raise FileNotFoundError(f"No JPEG images found matching pattern: {pattern}")

    buffers: list[bytes] = []
    for fp in file_paths:
        with open(fp, "rb") as f:
            buffers.append(f.read())
    logger.info("Loaded %d sample images into memory from %s", len(buffers), sample_dir)
    return buffers


def dispatch_single_prediction(endpoint_url: str, img_bytes: bytes) -> bool:
    """Send an in-memory image buffer as a multipart form upload to the prediction route.

    Args:
        endpoint_url: Complete HTTP URL of the `/api/v1/predict` endpoint.
        img_bytes: Raw JPEG image bytes.

    Returns:
        True if the request completed with HTTP 200, False otherwise.
    """
    boundary = "----TaxonVisionBoundaryP95Audit"
    headers = {"Content-Type": f"multipart/form-data; boundary={boundary}"}
    body = (
        (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="file"; filename="sample.jpg"\r\n'
            f"Content-Type: image/jpeg\r\n\r\n"
        ).encode()
        + img_bytes
        + f"\r\n--{boundary}--\r\n".encode()
    )

    req = urllib.request.Request(endpoint_url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=5.0) as resp:
            return bool(resp.status == 200)
    except urllib.error.URLError as err:
        logger.warning("Prediction request failed: %s", err)
        return False


def run_traffic_simulation(config: SimulationConfig) -> SimulationReport:
    """Execute end-to-end traffic simulation and optional drift evaluation.

    Args:
        config: Typed parameter configuration.

    Returns:
        SimulationReport detailing operational outcome.
    """
    images = load_sample_images(config.sample_dir)
    predict_url = f"{config.base_url}/api/v1/predict"
    drift_url = f"{config.base_url}/api/v1/drift/evaluate"

    logger.info(
        "Beginning simulation of %d predictions against %s (delay=%.1fms)...",
        config.request_count,
        predict_url,
        config.delay_ms,
    )

    success_count = 0
    failure_count = 0
    delay_sec = max(0.0, config.delay_ms / 1000.0)

    for idx in range(config.request_count):
        img_bytes = images[idx % len(images)]
        if dispatch_single_prediction(predict_url, img_bytes):
            success_count += 1
        else:
            failure_count += 1

        if (idx + 1) % 10 == 0 or (idx + 1) == config.request_count:
            logger.info("Progress: %d/%d predictions dispatched", idx + 1, config.request_count)

        if delay_sec > 0.0:
            time.sleep(delay_sec)

    drift_status = None
    drift_detected = None
    w1_distance = None

    if config.trigger_drift:
        logger.info("Triggering statistical drift evaluation via %s...", drift_url)
        drift_req = urllib.request.Request(drift_url, data=b"", method="POST")
        try:
            with urllib.request.urlopen(drift_req, timeout=30.0) as resp:
                import json

                data = json.loads(resp.read().decode("utf-8"))
                drift_status = data.get("status")
                drift_detected = data.get("drift_detected")
                w1_distance = data.get("w1_distance")
                logger.info(
                    "Drift evaluated successfully: status=%s, detected=%s, W1=%.4f, MMD²_p=%s",
                    drift_status,
                    drift_detected,
                    w1_distance or 0.0,
                    data.get("mmd2_p_value"),
                )
        except urllib.error.URLError as err:
            logger.warning("Drift evaluation request failed: %s", err)

    return SimulationReport(
        successful_requests=success_count,
        failed_requests=failure_count,
        drift_status=drift_status,
        drift_detected=drift_detected,
        w1_distance=w1_distance,
    )


def main(argv: list[str] | None = None) -> int:
    """Main CLI entrypoint.

    Args:
        argv: Command-line arguments.

    Returns:
        Process exit code (0 on success, non-zero on total failure).
    """
    config = parse_arguments(argv)
    report = run_traffic_simulation(config)
    logger.info(
        "Simulation completed: %d succeeded, %d failed.",
        report.successful_requests,
        report.failed_requests,
    )
    return 0 if report.successful_requests > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
