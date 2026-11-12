# SPDX-FileCopyrightText: 2026 Benjamin Förster
# SPDX-License-Identifier: MIT
"""Event-driven model retraining orchestrator DAG.

Summary:
    Coordinates automated retraining workflows triggered by the hourly drift monitor or the operational Grafana alerting relay, interfacing with Jenkins GPU build pipelines and reporting model promotion outcomes.

Why:
    Retraining vision models on raw imagery requires GPU hardware acceleration and signed CI/CD gates managed by Jenkins in the cluster. Airflow acts as the control-plane orchestrator, ensuring strictly serialised execution (`max_active_runs=1`) and transparent traceability in the UI.

How:
    1. `merge_human_feedback`: Appends verified Label Studio annotation exports to the dataset manifest.
    2. `trigger_jenkins_build`: Dispatches a parameterised build request to Jenkins via HTTP API token.
    3. `wait_for_jenkins_build`: Polls the build queue and execution status until terminal success or failure.
    4. `report_promotion_outcome`: Reads the resulting MLflow run metadata and Four-Gate promotion verdict into XCom.
"""

from __future__ import annotations

import base64
import http.cookiejar
import json
import logging
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from airflow.providers.http.hooks.http import HttpHook

try:
    from airflow.sdk import dag, task
except ImportError:
    from airflow.decorators import dag, task  # type: ignore[no-redef]

from taxon_vision.active_learning.feedback import merge_feedback_to_manifest
from taxon_vision.active_learning.label_studio_sync import DagsHubLabelStudioClient

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class JenkinsAuthContext:
    """Resolved Jenkins endpoint base URL and HTTP Basic authentication credentials.

    Why:
        Encapsulates the resolved Jenkins server URL, Basic Auth identity, personal API token, and target Multibranch Git branch into an immutable value object so dispatching and polling tasks never pass raw credential tuples.

    How:
        Constructed once per task invocation by `_resolve_jenkins_context()` after reconciling Airflow Connection metadata with explicit environment overrides.

    Attributes:
        base_url: Normalised Jenkins root URL without trailing slash.
        username: Authenticated Jenkins username.
        api_token: Personal API token used as the HTTP Basic password.
        branch: URL-encoded Multibranch pipeline branch name.
    """

    base_url: str
    username: str
    api_token: str
    branch: str = "main"


def _resolve_runtime_path(env_key: str, container_default: str, local_fallback: str) -> Path:
    """Resolve an execution-environment filesystem path with explicit env and container fallbacks.

    Why:
        The retraining DAG executes identically inside Kubernetes Airflow pods (where PersistentVolumeClaims mount at `/opt/airflow/...`) and during unit tests or local dry runs (where repository-relative paths apply).

    How:
        Checks the environment override `env_key` first, then inspects whether `container_default`'s parent directory exists inside the runtime container before falling back to `local_fallback`.

    Args:
        env_key: Environment variable name checked for an explicit path override.
        container_default: Canonical in-cluster mount path inside Airflow pods.
        local_fallback: Repository-relative fallback path for local runs and unit tests.

    Returns:
        Resolved `Path` pointing to the active runtime location.
    """
    explicit = os.getenv(env_key, "").strip()
    if explicit:
        return Path(explicit)
    container_path = Path(container_default)
    if container_path.parent.exists():
        return container_path
    return Path(local_fallback)


def _resolve_jenkins_context() -> JenkinsAuthContext:
    """Resolve Jenkins endpoint coordinates and API credentials for authenticated dispatch.

    Why:
        Airflow's `HttpHook` can hold unresolved literal `$(VAR)` strings from Helm/Kustomize templates or lack the configured `jenkins` connection during standalone task tests; reconciling both sources prevents silent HTTP 401 failures.

    How:
        Queries the Airflow `jenkins` connection via `HttpHook` when present, strips literal `$(...)` placeholders, prioritises `JENKINS_USER` and `JENKINS_API_TOKEN` pod environment variables, URL-encodes `TAXON_JENKINS_BRANCH`, and raises `RuntimeError` if no token is resolved.

    Returns:
        Immutable `JenkinsAuthContext` configured for authenticated CSRF + REST calls.

    Raises:
        RuntimeError: If neither environment variables nor Airflow connections supply `api_token`.
    """
    host = os.getenv("JENKINS_URL", "http://192.168.178.30:8080")
    conn_login = ""
    conn_password = ""
    try:
        conn = HttpHook(method="POST", http_conn_id="jenkins").get_connection("jenkins")
        if conn.host:
            host = f"{conn.schema or 'http'}://{conn.host}" if "://" not in conn.host else conn.host
            if conn.port and f":{conn.port}" not in host:
                host = f"{host.rstrip('/')}:{conn.port}"
        conn_login = conn.login or ""
        conn_password = conn.password or ""
    except Exception:
        logger.debug("Airflow HTTP connection 'jenkins' unavailable; relying on environment variables.")

    env_user = os.getenv("JENKINS_USER", "").strip()
    env_token = os.getenv("JENKINS_API_TOKEN", "").strip()
    username = env_user or ("" if conn_login.startswith("$(") else conn_login) or "admin"
    api_token = env_token or ("" if conn_password.startswith("$(") else conn_password)
    branch = urllib.parse.quote(os.getenv("TAXON_JENKINS_BRANCH", "main").strip() or "main", safe="")
    if not api_token:
        raise RuntimeError("Missing JENKINS_API_TOKEN for authenticated Jenkins pipeline dispatch.")
    return JenkinsAuthContext(
        base_url=host.rstrip("/"),
        username=username,
        api_token=api_token,
        branch=branch,
    )


def _build_jenkins_opener(auth: JenkinsAuthContext) -> tuple[urllib.request.OpenerDirector, str]:
    """Create a cookie-aware HTTP opener and encoded Basic Auth header for Jenkins CSRF sessions.

    Why:
        Jenkins' strict CSRF protection binds issued crumb tokens to the caller's HTTP session cookie (`JSESSIONID`); stateless single-shot requests without a shared cookie jar fail HTTP 403 `"No valid crumb was included"`.

    How:
        Instantiates a standard-library `CookieJar` bound to `HTTPCookieProcessor` and encodes `username:api_token` as an HTTP Basic `Authorization` header string.

    Args:
        auth: Resolved Jenkins endpoint and credential context.

    Returns:
        Tuple of the session-cookie `OpenerDirector` and the Base64 `Authorization` header value.
    """
    cookie_jar = http.cookiejar.CookieJar()
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cookie_jar))
    raw_creds = f"{auth.username}:{auth.api_token}".encode()
    basic_header = f"Basic {base64.b64encode(raw_creds).decode('ascii')}"
    return opener, basic_header


def _jenkins_json_request(
    opener: urllib.request.OpenerDirector,
    basic_header: str,
    url: str,
    *,
    method: str = "GET",
    headers: dict[str, str] | None = None,
) -> tuple[int, dict[str, str], str]:
    """Execute an authenticated Jenkins HTTP request with optional CSRF headers.

    Why:
        Centralises timeout enforcement, lowercase header normalisation (`Location` queue extraction), and empty POST payload handling across crumb retrieval, build triggering, and queue polling.

    How:
        Builds a `urllib.request.Request` with `Authorization`, executes it through the cookie-preserving `opener` with a 30-second timeout, normalises response headers to lowercase, and decodes the body as UTF-8.

    Args:
        opener: Cookie-preserving `urllib.request.OpenerDirector`.
        basic_header: HTTP Basic `Authorization` header value.
        url: Fully qualified Jenkins REST endpoint URL.
        method: HTTP verb (`GET` or `POST`).
        headers: Optional additional headers such as `Jenkins-Crumb`.

    Returns:
        Tuple containing HTTP status code, lowercase header map, and decoded response body text.
    """
    req_headers = {"Authorization": basic_header, **(headers or {})}
    req = urllib.request.Request(url=url, data=b"" if method == "POST" else None, headers=req_headers, method=method)
    with opener.open(req, timeout=30) as resp:
        resp_headers = {k.lower(): v for k, v in resp.headers.items()}
        body = resp.read().decode("utf-8", errors="replace")
        return int(resp.status), resp_headers, body


def _mlflow_json_request(mlflow_uri: str, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    """Query the in-cluster MLflow REST API using the Python standard library.

    Why:
        Airflow pods do not bundle the full `mlflow` client wheel to keep control-plane container images lean; querying the REST API directly retrieves real Four-Gate promotion metrics without expanding container dependencies.

    How:
        Sends an HTTP `GET` (when `payload` is `None`) or `POST` JSON request with a 15-second socket timeout to `{mlflow_uri}/{path}` and deserialises the response body as a dictionary.

    Args:
        mlflow_uri: Base tracking server URL (typically in-cluster `mlflow-server:5000`).
        path: Relative MLflow REST API v2.0 endpoint path.
        payload: Optional JSON body dict switching the request method to `POST`.

    Returns:
        Parsed JSON dictionary returned by MLflow, or `{}` on an empty body.
    """
    url = f"{mlflow_uri.rstrip('/')}/{path.lstrip('/')}"
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    headers = {"Accept": "application/json"}
    if data is not None:
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url=url, data=data, headers=headers, method="POST" if data is not None else "GET")
    with urllib.request.urlopen(req, timeout=15) as resp:
        raw = resp.read().decode("utf-8")
        return dict(json.loads(raw)) if raw.strip() else {}


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
        flagged_dir = _resolve_runtime_path(
            "TAXON_FLAGGED_DIR",
            "/opt/airflow/data/flagged_observations",
            "data/flagged_observations",
        )
        export_path = _resolve_runtime_path(
            "TAXON_LABEL_STUDIO_EXPORT_PATH",
            "/opt/airflow/data/flagged_observations/label_studio_export.json",
            "data/flagged_observations/label_studio_export.json",
        )
        raw_manifest_path = _resolve_runtime_path(
            "TAXON_RAW_MANIFEST_PATH",
            "/opt/airflow/data/manifests/raw_manifest.parquet",
            "data/manifests/raw_manifest.parquet",
        )
        catalog_path = _resolve_runtime_path(
            "TAXON_CATALOG_PATH",
            "/opt/airflow/config/taxa_catalog.yaml",
            "config/taxa_catalog.yaml",
        )

        ls_client = DagsHubLabelStudioClient()
        synced_pending = 0
        if ls_client.is_configured:
            synced_pending = ls_client.sync_pending_directory(flagged_dir)
            exported_count = ls_client.export_completed_annotations(export_path)
            logger.info(
                "DagsHub sync complete (uploaded_pending=%d, exported_completed=%d).", synced_pending, exported_count
            )
        elif not export_path.is_file():
            logger.info(
                "DAGSHUB_TOKEN unset and no local export at %s; proceeding without feedback merge.", export_path
            )
            return {
                "new_annotations_merged": 0,
                "catalog_expanded": False,
                "new_taxon_ids": [],
                "synced_pending": 0,
                "status": "skipped_no_pending_annotations",
            }

        res = merge_feedback_to_manifest(export_path, manifest_path=raw_manifest_path, catalog_path=catalog_path)
        logger.info(
            "Merged %d human annotations into %s (status=%s, total=%d, catalog_expanded=%s, ood_verified=%d)",
            res.new_records_added,
            raw_manifest_path,
            res.status,
            res.total_manifest_records,
            res.catalog_expanded,
            res.ood_verified_added,
        )
        return {
            "new_annotations_merged": res.new_records_added,
            "ood_records_skipped": res.ood_records_skipped,
            "ood_verified_added": res.ood_verified_added,
            "total_manifest_records": res.total_manifest_records,
            "catalog_expanded": res.catalog_expanded,
            "new_taxon_ids": list(res.new_taxon_ids),
            "synced_pending": synced_pending,
            "status": res.status,
        }

    @task
    def trigger_jenkins_build(feedback_status: dict[str, Any]) -> dict[str, Any]:
        """Dispatch an authenticated, CSRF-protected Multibranch Jenkins pipeline build.

        Why:
            Training runs on a dedicated NVIDIA RTX 5070 Ti node orchestrated via Jenkins Multibranch pipelines; closed-set annotation merges (Outcome A) must skip iNaturalist scraping (`INGEST_DATA=false`) to preserve freshly labelled samples, whereas novel-species expansions (Outcome B) must trigger full Stage 1 ingestion (`INGEST_DATA=true`) for newly registered classes.

        How:
            1. Resolves Jenkins credentials and opens a session-cookie opener via `_build_jenkins_opener`.
            2. Fetches `/crumbIssuer/api/json` for a CSRF token (tolerating HTTP 404 if crumb protection is disabled).
            3. Computes `INGEST_DATA` from `feedback_status['catalog_expanded']` and posts to `/job/taxon-vision-mlops/job/{branch}/buildWithParameters`.
            4. Extracts the returned `Location` queue URL for deterministic downstream polling.

        Args:
            feedback_status: XCom dictionary emitted by `merge_human_feedback`.

        Returns:
            Build dispatch payload containing `queue_url`, resolved `branch`, `ingest_data`, and HTTP status code.

        Raises:
            RuntimeError: If crumb fetching fails (non-404) or Jenkins omits the `Location` queue header.
        """
        auth = _resolve_jenkins_context()
        opener, basic_header = _build_jenkins_opener(auth)

        headers: dict[str, str] = {}
        try:
            _, _, crumb_raw = _jenkins_json_request(opener, basic_header, f"{auth.base_url}/crumbIssuer/api/json")
            crumb_data = json.loads(crumb_raw)
            field = str(crumb_data.get("crumbRequestField", "")).strip()
            crumb = str(crumb_data.get("crumb", "")).strip()
            if field and crumb:
                headers[field] = crumb
        except urllib.error.HTTPError as crumb_err:
            if crumb_err.code != 404:
                raise RuntimeError(f"Jenkins crumbIssuer request failed with HTTP {crumb_err.code}") from crumb_err

        catalog_expanded = bool(feedback_status.get("catalog_expanded", False))
        ingest_param = "true" if catalog_expanded else "false"
        trigger_url = (
            f"{auth.base_url}/job/taxon-vision-mlops/job/{auth.branch}/buildWithParameters"
            f"?RETRAIN_MODEL=true&INGEST_DATA={ingest_param}"
        )
        status_code, resp_headers, _ = _jenkins_json_request(
            opener,
            basic_header,
            trigger_url,
            method="POST",
            headers=headers,
        )
        queue_url = resp_headers.get("location", "")
        if not queue_url:
            raise RuntimeError(f"Jenkins trigger HTTP {status_code} succeeded without Location queue header.")

        logger.info("Jenkins Multibranch build queued at %s (INGEST_DATA=%s)", queue_url, ingest_param)
        return {
            "queue_url": queue_url,
            "base_url": auth.base_url,
            "branch": auth.branch,
            "ingest_data": ingest_param,
            "status": "queued",
            "status_code": status_code,
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
        auth = _resolve_jenkins_context()
        opener, basic_header = _build_jenkins_opener(auth)
        queue_url = str(build_trigger.get("queue_url", "")).rstrip("/")
        if not queue_url:
            raise RuntimeError("Cannot poll Jenkins execution without queue_url.")

        timeout_seconds = int(os.getenv("TAXON_JENKINS_TIMEOUT_SECONDS", "9000"))
        poll_interval = float(os.getenv("TAXON_JENKINS_POLL_SECONDS", "10"))
        deadline = time.monotonic() + timeout_seconds

        build_url = ""
        build_number = 0
        while time.monotonic() < deadline:
            _, _, queue_raw = _jenkins_json_request(opener, basic_header, f"{queue_url}/api/json")
            queue_data = json.loads(queue_raw)
            if queue_data.get("cancelled"):
                raise RuntimeError(f"Queued Jenkins job at {queue_url} was cancelled before execution.")
            executable = queue_data.get("executable")
            if isinstance(executable, dict) and executable.get("url"):
                build_url = str(executable["url"]).rstrip("/")
                build_number = int(executable.get("number", 0))
                break
            time.sleep(poll_interval)

        if not build_url:
            raise TimeoutError(f"Timed out waiting for Jenkins queue item {queue_url} to schedule.")

        while time.monotonic() < deadline:
            _, _, build_raw = _jenkins_json_request(opener, basic_header, f"{build_url}/api/json")
            build_data = json.loads(build_raw)
            if not bool(build_data.get("building", True)) and build_data.get("result"):
                result = str(build_data["result"])
                duration_seconds = float(build_data.get("duration", 0.0)) / 1000.0
                if result != "SUCCESS":
                    raise RuntimeError(f"Jenkins build #{build_number} ({build_url}) finished with status {result}.")
                return {
                    "result": result,
                    "build_number": build_number,
                    "build_url": build_url,
                    "duration_seconds": duration_seconds,
                }
            time.sleep(poll_interval)

        raise TimeoutError(f"Timed out waiting for Jenkins build #{build_number} ({build_url}) to finish.")

    @task
    def report_promotion_outcome(build_outcome: dict[str, Any]) -> dict[str, Any]:
        """Query the in-cluster MLflow REST API for real Four-Gate promotion verdict and holdout metrics.

        Why:
            Hardcoded promotion verdicts hide quantization degradation or conformal coverage failures from operators inspecting Airflow run logs; querying MLflow provides auditable empirical evidence (`eval.promoted`, `eval.top1_accuracy`, `eval.ece`, `eval.conformal_empirical_coverage`) and the live `@Production` alias version.

        How:
            1. Searches the primary `taxon-vision-head-training` experiment (with `taxon_vision_evaluation` fallback) ordered by `start_time DESC` to inspect the resumed parent run logged by Stage 7 `evaluate.py`.
            2. Queries `registered-models/get-model-version-by-alias` for `taxon_vision_onnx@Production`.
            3. Translates `eval.promoted >= 1.0` into `"Production"`, `"Rejected"`, or `"Unavailable"` and returns the structured audit dictionary.

        Args:
            build_outcome: XCom build completion dictionary emitted by `wait_for_jenkins_build`.

        Returns:
            Audit summary mapping Jenkins build metadata, Four-Gate promotion verdict, evaluation metrics, and active `@Production` version.
        """
        mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow-server.taxon-vision.svc.cluster.local:5000")
        eval_metrics: dict[str, float] = {}
        eval_run_id = ""
        production_version = ""

        try:
            for exp_name in ("taxon-vision-head-training", "taxon_vision_evaluation"):
                exp_data = _mlflow_json_request(
                    mlflow_uri,
                    f"api/2.0/mlflow/experiments/get-by-name?experiment_name={exp_name}",
                )
                experiment_id = str(exp_data.get("experiment", {}).get("experiment_id", ""))
                if not experiment_id:
                    continue
                runs_data = _mlflow_json_request(
                    mlflow_uri,
                    "api/2.0/mlflow/runs/search",
                    payload={
                        "experiment_ids": [experiment_id],
                        "max_results": 1,
                        "order_by": ["attributes.start_time DESC"],
                    },
                )
                runs = runs_data.get("runs", [])
                if runs:
                    latest_run = runs[0]
                    eval_run_id = str(latest_run.get("info", {}).get("run_id", ""))
                    for metric in latest_run.get("data", {}).get("metrics", []):
                        eval_metrics[str(metric["key"])] = float(metric["value"])
                    break
        except Exception as err:
            logger.warning("Could not query MLflow evaluation experiment runs at %s: %s", mlflow_uri, err)

        try:
            alias_data = _mlflow_json_request(
                mlflow_uri,
                "api/2.0/mlflow/registered-models/get-model-version-by-alias?name=taxon_vision_onnx&alias=Production",
            )
            production_version = str(alias_data.get("model_version", {}).get("version", ""))
        except Exception as err:
            logger.warning("Could not query taxon_vision_onnx@Production alias at %s: %s", mlflow_uri, err)

        promoted_flag = eval_metrics.get("eval.promoted")
        if promoted_flag is None:
            verdict = "Unavailable"
        elif promoted_flag >= 1.0:
            verdict = "Production"
        else:
            verdict = "Rejected"

        summary = {
            "jenkins_build_result": build_outcome.get("result", "UNKNOWN"),
            "jenkins_build_number": build_outcome.get("build_number"),
            "promotion_verdict": verdict,
            "eval_top1_accuracy": eval_metrics.get("eval.top1_accuracy"),
            "eval_ece": eval_metrics.get("eval.ece"),
            "eval_conformal_empirical_coverage": eval_metrics.get(
                "eval.conformal_empirical_coverage",
                eval_metrics.get("eval.empirical_test_coverage"),
            ),
            "mlflow_eval_run_id": eval_run_id,
            "active_production_version": production_version,
            "active_model_alias": "taxon_vision_onnx@Production",
            "report_timestamp": time.time(),
        }
        logger.info("Model retraining cycle finished. Real MLflow promotion summary: %s", summary)
        return summary

    feedback = merge_human_feedback()
    trigger = trigger_jenkins_build(feedback)
    build = wait_for_jenkins_build(trigger)
    report_promotion_outcome(build)


model_retraining()
