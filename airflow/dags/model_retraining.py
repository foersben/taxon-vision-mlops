# SPDX-FileCopyrightText: 2026 Benjamin Förster
# SPDX-License-Identifier: MIT
"""Event-driven model retraining orchestrator DAG.

Summary:
    Coordinates automated retraining workflows triggered by the hourly drift monitor or the operational Grafana alerting relay, interfacing with Jenkins GPU build pipelines and reporting model promotion outcomes.

Why:
    Retraining vision models on raw imagery requires GPU hardware acceleration and signed CI/CD gates managed by Jenkins in the cluster. Airflow acts as the control-plane orchestrator, ensuring strictly serialised execution (`max_active_runs=1`) and transparent traceability in the UI whilst delegating HTTP transport helpers to `_jenkins_mlflow_client.py`.

How:
    1. `merge_human_feedback`: Appends verified Label Studio annotation exports to the dataset manifest.
    2. `trigger_jenkins_build`: Dispatches a parameterised build request to Jenkins via HTTP API token.
    3. `wait_for_jenkins_build`: Polls the build queue and execution status until terminal success or failure.
    4. `report_promotion_outcome`: Reads the resulting MLflow run metadata and Four-Gate promotion verdict into XCom.
"""

from __future__ import annotations

import logging
import os
from typing import Any

try:
    from airflow.sdk import dag, task
except ImportError:
    from airflow.decorators import dag, task  # type: ignore[no-redef]

try:
    from _jenkins_mlflow_client import (
        build_jenkins_session,
        fetch_jenkins_crumb_headers,
        fetch_mlflow_promotion_summary,
        jenkins_json_request,
        poll_jenkins_queue_and_build,
        resolve_jenkins_context,
        resolve_runtime_path,
    )
except ImportError:
    from airflow.dags._jenkins_mlflow_client import (
        build_jenkins_session,
        fetch_jenkins_crumb_headers,
        fetch_mlflow_promotion_summary,
        jenkins_json_request,
        poll_jenkins_queue_and_build,
        resolve_jenkins_context,
        resolve_runtime_path,
    )

from taxon_vision.active_learning.label_studio_sync import (
    FeedbackSyncPaths,
    sync_and_merge_human_feedback,
)

logger = logging.getLogger(__name__)


@dag(
    dag_id="model_retraining",
    schedule=None,
    catchup=False,
    max_active_runs=1,
    tags=["taxon-vision", "retraining", "pipeline"],
)
def model_retraining() -> None:
    """Define the event-driven Airflow DAG closing active learning feedback and GPU retraining.

    Why:
        Coordinates stateful human-in-the-loop curation (`DagsHubLabelStudioClient`), DVC manifest augmentation, remote bare-metal GPU execution on Jenkins, and immutable MLflow Model Registry auditing under a single serialised `max_active_runs=1` lock so concurrent drift alerts cannot race on manifest mutations or GPU slots.

    How:
        Wires four TaskFlow operators sequentially: `merge_human_feedback -> trigger_jenkins_build -> wait_for_jenkins_build -> report_promotion_outcome`, propagating status and MLflow run verdicts via XCom.
    """

    @task
    def merge_human_feedback() -> dict[str, Any]:
        """Synchronise pending local captures, pull DagsHub annotations, and merge into `raw_manifest.parquet`.

        Why:
            Before GPU training runs on Jenkins, locally cached open-license captures must be uploaded to DagsHub Label Studio and completed expert annotations must be merged into `raw_manifest.parquet` with open-license verification, separated OOD routing, and Outcome B taxonomy expansion tracking.

        How:
            1. Resolves runtime PVC/repo paths for flagged captures, Label Studio export JSON, `raw_manifest.parquet`, and `taxa_catalog.yaml`.
            2. If `DAGSHUB_TOKEN` is configured, flushes unsynchronised consented crops to DagsHub S3/Label Studio sorted by `TriageQueue` uncertainty and exports completed tasks to JSON.
            3. Invokes `merge_feedback_to_manifest` against `raw_manifest.parquet` (preserving Stage 2 `audit_license` as an immutable downstream gate) and returns counts plus `catalog_expanded` to XCom.

        Returns:
            Dictionary summarising merged record counts, OOD audit counts, newly registered taxon IDs, and `catalog_expanded`.
        """
        paths = FeedbackSyncPaths(
            flagged_dir=resolve_runtime_path(
                "TAXON_FLAGGED_DIR",
                "/opt/airflow/data/flagged_observations",
                "data/flagged_observations",
            ),
            export_path=resolve_runtime_path(
                "TAXON_LABEL_STUDIO_EXPORT_PATH",
                "/opt/airflow/data/flagged_observations/label_studio_export.json",
                "data/flagged_observations/label_studio_export.json",
            ),
            raw_manifest_path=resolve_runtime_path(
                "TAXON_RAW_MANIFEST_PATH",
                "/opt/airflow/data/manifests/raw_manifest.parquet",
                "data/manifests/raw_manifest.parquet",
            ),
            catalog_path=resolve_runtime_path(
                "TAXON_CATALOG_PATH",
                "/opt/airflow/config/taxa_catalog.yaml",
                "config/taxa_catalog.yaml",
            ),
        )
        return sync_and_merge_human_feedback(paths)

    @task
    def trigger_jenkins_build(feedback_status: dict[str, Any]) -> dict[str, Any]:
        """Dispatch an authenticated, CSRF-protected Multibranch Jenkins pipeline build.

        Why:
            Training runs on a dedicated NVIDIA RTX 5070 Ti node orchestrated via Jenkins Multibranch pipelines; closed-set annotation merges (Outcome A) must skip iNaturalist scraping (`INGEST_DATA=false`) to preserve freshly labelled samples, whereas novel-species expansions (Outcome B) must trigger full Stage 1 ingestion (`INGEST_DATA=true`) for newly registered classes.

        How:
            1. Resolves Jenkins credentials and opens a session-cookie opener via `build_jenkins_session`.
            2. Fetches `/crumbIssuer/api/json` for a CSRF token (tolerating HTTP 404 if crumb protection is disabled).
            3. Computes `INGEST_DATA` from `feedback_status['catalog_expanded']` and posts to `/job/taxon-vision-mlops/job/{branch}/buildWithParameters` with `SKIP_CODE_GATES=true` so automated drift retrains proceed directly to GPU training.
            4. Extracts the returned `Location` queue URL for deterministic downstream polling.

        Args:
            feedback_status: XCom dictionary emitted by `merge_human_feedback`.

        Returns:
            Build dispatch payload containing `queue_url`, resolved `branch`, `ingest_data`, and HTTP status code.

        Raises:
            RuntimeError: If crumb fetching fails (non-404) or Jenkins omits the `Location` queue header.
        """
        auth = resolve_jenkins_context()
        session = build_jenkins_session(auth)
        headers = fetch_jenkins_crumb_headers(session, auth.base_url)

        catalog_expanded = bool(feedback_status.get("catalog_expanded", False))
        ingest_param = "true" if catalog_expanded else "false"
        trigger_url = (
            f"{auth.base_url}/job/taxon-vision-mlops/job/{auth.branch}/buildWithParameters"
            f"?RETRAIN_MODEL=true&INGEST_DATA={ingest_param}&SKIP_CODE_GATES=true"
        )
        resp = jenkins_json_request(session, trigger_url, method="POST", headers=headers)
        queue_url = resp.headers.get("location", "")
        if not queue_url:
            raise RuntimeError(f"Jenkins trigger HTTP {resp.status_code} succeeded without Location queue header.")

        logger.info(
            "Jenkins Multibranch build queued at %s (INGEST_DATA=%s, SKIP_CODE_GATES=true)",
            queue_url,
            ingest_param,
        )
        return {
            "queue_url": queue_url,
            "base_url": auth.base_url,
            "branch": auth.branch,
            "ingest_data": ingest_param,
            "status": "queued",
            "status_code": resp.status_code,
        }

    @task
    def wait_for_jenkins_build(build_trigger: dict[str, Any]) -> dict[str, Any]:
        """Poll the Jenkins build queue item and active build execution until terminal completion.

        Why:
            Triggering `buildWithParameters` is asynchronous and returns only a queue URL (`HTTP 201`); blocking until Jenkins allocates the GPU worker and finishes Stage 7 guarantees that `report_promotion_outcome` queries post-retrain MLflow metrics rather than stale runs.

        How:
            Phase 1 polls `{queue_url}/api/json` every `TAXON_JENKINS_POLL_SECONDS` until an executable build URL appears or cancellation occurs. Phase 2 polls `{build_url}/api/json` until `building` becomes `False`, verifying `result == "SUCCESS"` within `TAXON_JENKINS_TIMEOUT_SECONDS`.

        Args:
            build_trigger: XCom dispatch dictionary emitted by `trigger_jenkins_build`.

        Returns:
            Terminal Jenkins build summary including `result`, `build_number`, `build_url`, and duration in seconds.

        Raises:
            RuntimeError: If `queue_url` is missing, the queued build is cancelled, or Jenkins completes with non-SUCCESS status.
            TimeoutError: If queue allocation or build execution exceeds `TAXON_JENKINS_TIMEOUT_SECONDS`.
        """
        timeout_seconds = int(os.getenv("TAXON_JENKINS_TIMEOUT_SECONDS", "9000"))
        poll_interval = float(os.getenv("TAXON_JENKINS_POLL_SECONDS", "10"))
        return poll_jenkins_queue_and_build(
            str(build_trigger.get("queue_url", "")),
            timeout_seconds=timeout_seconds,
            poll_interval=poll_interval,
        )

    @task
    def report_promotion_outcome(build_outcome: dict[str, Any]) -> dict[str, Any]:
        """Query the in-cluster MLflow REST API for real Four-Gate promotion verdict and holdout metrics.

        Why:
            Hardcoded promotion verdicts hide quantization degradation or conformal coverage failures from operators inspecting Airflow run logs; querying MLflow provides auditable empirical evidence (`eval.promoted`, `eval.top1_accuracy`, `eval.ece`, `eval.conformal_empirical_coverage`) and the live `@Production` alias version.

        How:
            Delegates REST experiment search and `@Production` alias resolution to `fetch_mlflow_promotion_summary` and logs the audit summary to Airflow task output.

        Args:
            build_outcome: XCom build completion dictionary emitted by `wait_for_jenkins_build`.

        Returns:
            Audit summary mapping Jenkins build metadata, Four-Gate promotion verdict, evaluation metrics, and active `@Production` version.
        """
        mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "https://dagshub.com/foersben/taxon-vision-mlops.mlflow")
        summary = fetch_mlflow_promotion_summary(mlflow_uri, build_outcome)
        logger.info("Model retraining cycle finished. Real MLflow promotion summary: %s", summary)
        return summary

    feedback = merge_human_feedback()
    trigger = trigger_jenkins_build(feedback)
    build = wait_for_jenkins_build(trigger)
    report_promotion_outcome(build)


model_retraining()
