# SPDX-FileCopyrightText: 2026 Benjamin Förster
# SPDX-License-Identifier: MIT
"""Standard-library Jenkins CSRF, REST, and MLflow HTTP client helpers for Airflow DAGs.

Summary:
    Provides cookie-preserving Jenkins HTTP CSRF/build dispatch helpers, runtime path resolution, and stdlib MLflow REST query utilities designed specifically for the stock `apache/airflow:3.0.0-python3.12` container runtime.

Why (Architectural Rationale & Trade-offs):
    Airflow pods run stock upstream Apache Airflow images without `taxon_vision` installed as a wheel, mounting `airflow/dags/` via flat ConfigMap files. Extracting transport and credential helpers into a sibling `_`-prefixed module (ignored by `.airflowignore`) slims `model_retraining.py` below 300 LOC whilst avoiding heavy custom Docker images or `mlflow` wheel installs in the control plane.

How (Algorithmic Mechanism & Invariants):
    Uses standard-library `urllib.request`, `http.cookiejar.CookieJar` (for Jenkins `JSESSIONID` CSRF crumb binding), and `airflow.providers.http.hooks.http.HttpHook` (for Airflow Connection fallback), encapsulating outputs in frozen dataclasses (`JenkinsAuthContext`, `JenkinsHttpSession`, `JenkinsHttpResponse`).
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

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class JenkinsAuthContext:
    """Resolved Jenkins endpoint base URL and HTTP Basic authentication credentials.

    Why:
        Encapsulates the resolved Jenkins server URL, Basic Auth identity, personal API token, and target Multibranch Git branch into an immutable value object so dispatching and polling tasks never pass raw credential tuples.

    How:
        Constructed once per task invocation by `resolve_jenkins_context()` after reconciling Airflow Connection metadata with explicit environment overrides.

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


@dataclass(frozen=True)
class JenkinsHttpSession:
    """Cookie-preserving HTTP opener paired with pre-encoded Basic Auth header."""

    opener: urllib.request.OpenerDirector
    basic_header: str


@dataclass(frozen=True)
class JenkinsHttpResponse:
    """Normalised HTTP response returned by `jenkins_json_request`."""

    status_code: int
    headers: dict[str, str]
    body: str


def resolve_runtime_path(env_key: str, container_default: str, local_fallback: str) -> Path:
    """Resolve an execution-environment filesystem path with explicit env and container fallbacks.

    Why:
        The retraining DAG executes identically inside Kubernetes Airflow pods (where volumes mount at `/opt/airflow/...`) and during unit tests or local dry runs (where repository-relative paths apply).

    How:
        Checks environment override `env_key` first, then inspects whether `container_default`'s parent directory exists inside the runtime container before falling back to `local_fallback`.

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


def resolve_jenkins_context() -> JenkinsAuthContext:
    """Resolve Jenkins endpoint coordinates and API credentials for authenticated dispatch.

    Why:
        Airflow's `HttpHook` can hold unresolved literal `$(VAR)` strings from templates or lack the configured `jenkins` connection during standalone task tests; reconciling both sources prevents silent HTTP 401 failures.

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
    username = env_user or ("" if conn_login.startswith("$(") else conn_login) or "benni"
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


def build_jenkins_session(auth: JenkinsAuthContext) -> JenkinsHttpSession:
    """Create a cookie-aware HTTP opener and encoded Basic Auth header for Jenkins CSRF sessions.

    Why:
        Jenkins' strict CSRF protection binds issued crumb tokens to the caller's HTTP session cookie (`JSESSIONID`); stateless single-shot requests without a shared cookie jar fail HTTP 403 `"No valid crumb was included"`.

    How:
        Instantiates a standard-library `CookieJar` bound to `HTTPCookieProcessor` and encodes `username:api_token` as an HTTP Basic `Authorization` header string inside a frozen `JenkinsHttpSession`.

    Args:
        auth: Resolved Jenkins endpoint and credential context.

    Returns:
        `JenkinsHttpSession` holding the cookie-preserving opener and Basic Auth header.
    """
    cookie_jar = http.cookiejar.CookieJar()
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cookie_jar))
    raw_creds = f"{auth.username}:{auth.api_token}".encode()
    basic_header = f"Basic {base64.b64encode(raw_creds).decode('ascii')}"
    return JenkinsHttpSession(opener=opener, basic_header=basic_header)


def jenkins_json_request(
    session: JenkinsHttpSession,
    url: str,
    *,
    method: str = "GET",
    headers: dict[str, str] | None = None,
) -> JenkinsHttpResponse:
    """Execute an authenticated Jenkins HTTP request with optional CSRF headers.

    Why:
        Centralises timeout enforcement, lowercase header normalisation (`Location` queue extraction), and empty POST payload handling across crumb retrieval, build triggering, and queue polling.

    How:
        Builds a `urllib.request.Request` with `Authorization`, executes it through the cookie-preserving opener with a 30-second timeout, normalises response headers to lowercase, and returns `JenkinsHttpResponse`.

    Args:
        session: Cookie-preserving `JenkinsHttpSession`.
        url: Fully qualified Jenkins REST endpoint URL.
        method: HTTP verb (`GET` or `POST`).
        headers: Optional additional headers such as `Jenkins-Crumb`.

    Returns:
        `JenkinsHttpResponse` containing status code, lowercase headers, and decoded body text.
    """
    req_headers = {"Authorization": session.basic_header, **(headers or {})}
    req = urllib.request.Request(url=url, data=b"" if method == "POST" else None, headers=req_headers, method=method)
    with session.opener.open(req, timeout=30) as resp:
        resp_headers = {k.lower(): v for k, v in resp.headers.items()}
        body = resp.read().decode("utf-8", errors="replace")
        return JenkinsHttpResponse(status_code=int(resp.status), headers=resp_headers, body=body)


def fetch_jenkins_crumb_headers(session: JenkinsHttpSession, base_url: str) -> dict[str, str]:
    """Fetch a Jenkins CSRF crumb header mapping, tolerating disabled crumb protection (HTTP 404)."""
    try:
        resp = jenkins_json_request(session, f"{base_url}/crumbIssuer/api/json")
        crumb_data = json.loads(resp.body)
        field = str(crumb_data.get("crumbRequestField", "")).strip()
        crumb = str(crumb_data.get("crumb", "")).strip()
        return {field: crumb} if field and crumb else {}
    except urllib.error.HTTPError as crumb_err:
        if crumb_err.code != 404:
            raise RuntimeError(f"Jenkins crumbIssuer request failed with HTTP {crumb_err.code}") from crumb_err
        return {}


def mlflow_json_request(mlflow_uri: str, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    """Query the configured MLflow REST API using the Python standard library.

    Why:
        Airflow pods do not bundle the full `mlflow` client wheel to keep control-plane container images lean; querying the REST API directly retrieves real Four-Gate promotion metrics without expanding container dependencies.

    How:
        Sends an HTTP `GET` (when `payload` is `None`) or `POST` JSON request with a 15-second socket timeout to `{mlflow_uri}/{path}`, injects HTTP Basic Auth headers when targeting DagsHub (`DAGSHUB_TOKEN` or `MLFLOW_TRACKING_PASSWORD`), and deserialises the response body as a dictionary.

    Args:
        mlflow_uri: Base tracking server URL (defaulting to DagsHub MLflow).
        path: Relative MLflow REST API v2.0 endpoint path.
        payload: Optional JSON body dict switching the request method to `POST`.

    Returns:
        Parsed JSON dictionary returned by MLflow, or `{}` on an empty body.
    """
    url = f"{mlflow_uri.rstrip('/')}/{path.lstrip('/')}"
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    headers = {"Accept": "application/json"}
    token = os.getenv("DAGSHUB_TOKEN") or os.getenv("MLFLOW_TRACKING_PASSWORD") or ""
    if "dagshub.com" in mlflow_uri and token:
        username = os.getenv("MLFLOW_TRACKING_USERNAME") or "foersben"
        encoded = base64.b64encode(f"{username}:{token}".encode()).decode("ascii")
        headers["Authorization"] = f"Basic {encoded}"
    if data is not None:
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url=url, data=data, headers=headers, method="POST" if data is not None else "GET")
    with urllib.request.urlopen(req, timeout=15) as resp:
        raw = resp.read().decode("utf-8")
        return dict(json.loads(raw)) if raw.strip() else {}


@dataclass(frozen=True)
class _QueuedBuildInfo:
    """Resolved Jenkins executable build URL and numeric build identifier."""

    build_url: str
    build_number: int


def _wait_for_executable_build(
    session: JenkinsHttpSession,
    queue_url: str,
    deadline: float,
    poll_interval: float,
) -> _QueuedBuildInfo:
    """Poll a queued Jenkins item until an executable build URL is assigned."""
    while time.monotonic() < deadline:
        queue_resp = jenkins_json_request(session, f"{queue_url}/api/json")
        queue_data = json.loads(queue_resp.body)
        if queue_data.get("cancelled"):
            raise RuntimeError(f"Queued Jenkins job at {queue_url} was cancelled before execution.")
        executable = queue_data.get("executable")
        if isinstance(executable, dict) and executable.get("url"):
            return _QueuedBuildInfo(
                build_url=str(executable["url"]).rstrip("/"),
                build_number=int(executable.get("number", 0)),
            )
        time.sleep(poll_interval)
    raise TimeoutError(f"Timed out waiting for Jenkins queue item {queue_url} to schedule.")


def poll_jenkins_queue_and_build(queue_url: str, *, timeout_seconds: int, poll_interval: float) -> dict[str, Any]:
    """Block until a queued Jenkins build finishes execution and verify SUCCESS status."""
    cleaned_queue = queue_url.rstrip("/")
    if not cleaned_queue:
        raise RuntimeError("Cannot poll Jenkins execution without queue_url.")

    auth = resolve_jenkins_context()
    session = build_jenkins_session(auth)
    deadline = time.monotonic() + timeout_seconds
    queued = _wait_for_executable_build(session, cleaned_queue, deadline, poll_interval)
    build_url = queued.build_url
    build_number = queued.build_number

    while time.monotonic() < deadline:
        build_resp = jenkins_json_request(session, f"{build_url}/api/json")
        build_data = json.loads(build_resp.body)
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


def fetch_mlflow_promotion_summary(mlflow_uri: str, build_outcome: dict[str, Any]) -> dict[str, Any]:
    """Query DagsHub MLflow REST API for latest holdout evaluation metrics and @Production alias."""
    eval_metrics: dict[str, float] = {}
    eval_run_id = ""
    production_version = ""

    try:
        for exp_name in ("taxon-vision-head-training", "taxon_vision_evaluation"):
            exp_data = mlflow_json_request(
                mlflow_uri,
                f"api/2.0/mlflow/experiments/get-by-name?experiment_name={exp_name}",
            )
            experiment_id = str(exp_data.get("experiment", {}).get("experiment_id", ""))
            if not experiment_id:
                continue
            runs_data = mlflow_json_request(
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
        alias_data = mlflow_json_request(
            mlflow_uri,
            "api/2.0/mlflow/registered-models/get-model-version-by-alias?name=taxon_vision_onnx&alias=Production",
        )
        production_version = str(alias_data.get("model_version", {}).get("version", ""))
    except Exception as err:
        logger.warning("Could not query taxon_vision_onnx@Production alias at %s: %s", mlflow_uri, err)

    promoted_flag = eval_metrics.get("eval.promoted")
    verdict = "Unavailable" if promoted_flag is None else ("Production" if promoted_flag >= 1.0 else "Rejected")
    return {
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
