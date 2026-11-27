# SPDX-FileCopyrightText: 2026 Benjamin Förster
# SPDX-License-Identifier: MIT
"""DagsHub repository configuration and pure-stdlib AWS SigV4 S3 upload client.

Summary:
    Encapsulates DagsHub tenant coordinates (`DagsHubRepoConfig`), environment-based credential resolution (`build_dagshub_repo_config`), and stdlib AWS SigV4 HTTP PUT uploads (`_s3_put_object`) for active-learning image crops.

Why (Architectural Rationale & Trade-offs):
    Trimmed runtime serving containers (`pixi -e ci`) and lightweight Airflow worker images omit `boto3`. Isolating DagsHub coordinate resolution and zero-dependency AWS SigV4 signing into a leaf sibling module keeps `label_studio_sync.py` below the 500-LOC Rule 08 threshold and eliminates external S3 SDK bloat.

How (Algorithmic Mechanism & Invariants):
    Resolves `TAXON_DAGSHUB_OWNER`, `TAXON_DAGSHUB_REPO`, `DAGSHUB_TOKEN`, and optional numeric `TAXON_LABEL_STUDIO_PROJECT_ID` into an immutable `DagsHubRepoConfig`. Signs path-style S3 HTTP `PUT` requests via `hmac` + `hashlib` (`AWS4-HMAC-SHA256`) and uploads via `urllib.request.urlopen`.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any


@dataclass(frozen=True)
class DagsHubRepoConfig:
    """Configuration coordinates for DagsHub Label Studio and S3 repository storage.

    Why:
        Bundles DagsHub tenant coordinates, Personal Access Token credentials, project identity, and network timeouts into an immutable typed container to avoid scattered string formatting across HTTP and S3 clients.

    How:
        Constructed by `build_dagshub_repo_config()` from environment defaults and exposes derived REST (`label_studio_api_base`) and SigV4 (`s3_endpoint_url`) URL properties.

    Attributes:
        owner: Repository owner username on DagsHub.
        repo: Repository slug name on DagsHub.
        token: DagsHub Personal Access Token for Bearer and S3 SigV4 authentication.
        project_id: Optional explicit Label Studio numeric project ID.
        project_title: Default project title created when none exists.
        bucket_prefix: S3 prefix inside the repository bucket for staged triage crops.
        timeout_seconds: HTTP request timeout ceiling in seconds.
    """

    owner: str = "foersben"
    repo: str = "taxon-vision-mlops"
    token: str = ""
    project_id: int | None = None
    project_title: str = "TaxonVision Active Learning Triage"
    bucket_prefix: str = "triage-queue"
    timeout_seconds: float = 20.0

    @property
    def label_studio_api_base(self) -> str:
        """Return the base REST URL for the managed DagsHub Label Studio instance.

        Why:
            DagsHub namespaces hosted Label Studio endpoints per owner and repository slug under `/annotations/de/api`.

        How:
            Interpolates `self.owner` and `self.repo` into the canonical HTTPS API root.

        Returns:
            Base URL string without a trailing slash.
        """
        return f"https://dagshub.com/{self.owner}/{self.repo}/annotations/de/api"

    @property
    def s3_endpoint_url(self) -> str:
        """Return the S3-compatible endpoint URL for the repository bucket.

        Why:
            DagsHub exposes each user's repository object store as an S3-compatible endpoint scoped under `/api/v1/repo-buckets/s3/<owner>`.

        How:
            Interpolates `self.owner` into DagsHub's repository bucket gateway URL.

        Returns:
            Endpoint URL used for SigV4 path-style S3 requests.
        """
        return f"https://dagshub.com/api/v1/repo-buckets/s3/{self.owner}"


def build_dagshub_repo_config() -> DagsHubRepoConfig:
    """Construct DagsHubRepoConfig from runtime environment variables.

    Why:
        Decouples repository coordinates and personal access token resolution from caller invocation sites so serving pods, CLI tools, and Airflow tasks share identical environment fallback rules.

    How:
        Reads TAXON_DAGSHUB_OWNER, TAXON_DAGSHUB_REPO, DAGSHUB_TOKEN (falling back to MLFLOW_TRACKING_PASSWORD), and optional numeric TAXON_LABEL_STUDIO_PROJECT_ID into an immutable DagsHubRepoConfig instance.

    Returns:
        Populated DagsHubRepoConfig container.
    """
    owner = os.getenv("TAXON_DAGSHUB_OWNER", "foersben").strip()
    repo = os.getenv("TAXON_DAGSHUB_REPO", "taxon-vision-mlops").strip()
    token = (os.getenv("DAGSHUB_TOKEN") or os.getenv("MLFLOW_TRACKING_PASSWORD") or "").strip()
    raw_pid = os.getenv("TAXON_LABEL_STUDIO_PROJECT_ID", "").strip()
    project_id = int(raw_pid) if raw_pid.isdigit() else None
    return DagsHubRepoConfig(owner=owner, repo=repo, token=token, project_id=project_id)


@dataclass(frozen=True)
class _S3PutRequest:
    """Typed parameter container for a pure-stdlib AWS SigV4 S3 PUT upload."""

    endpoint_url: str
    bucket: str
    object_key: str
    payload: bytes
    access_key: str
    secret_key: str
    content_type: str = "image/jpeg"
    region: str = "us-east-1"
    timeout_seconds: float = 20.0


def _hmac_sha256(key: bytes, message: str) -> bytes:
    """Compute raw HMAC-SHA256 digest for SigV4 key derivation."""
    return hmac.new(key, message.encode("utf-8"), hashlib.sha256).digest()


def _derive_sigv4_signing_key(secret_key: str, date_stamp: str, region: str, service: str = "s3") -> bytes:
    """Derive an AWS Signature Version 4 signing key from secret credentials and scope."""
    k_date = _hmac_sha256(f"AWS4{secret_key}".encode(), date_stamp)
    k_region = _hmac_sha256(k_date, region)
    k_service = _hmac_sha256(k_region, service)
    return _hmac_sha256(k_service, "aws4_request")


def _s3_put_object(req: _S3PutRequest) -> None:
    """Upload binary bytes to an S3-compatible path-style endpoint using stdlib AWS SigV4.

    Why:
        Trimmed runtime serving containers (`pixi -e ci`) and lightweight Airflow workers omit `boto3`. Signing path-style S3 HTTP `PUT` requests directly with `hmac` and `hashlib` keeps active-learning crop uploads self-contained with zero third-party SDK overhead.

    How:
        Normalises the canonical URI `{endpoint_path}/{bucket}/{object_key}`, computes SHA-256 payload digest and canonical `content-type;host;x-amz-content-sha256;x-amz-date` headers, signs the AWS4-HMAC-SHA256 string-to-sign, and issues an HTTP `PUT` via `urllib.request.urlopen`.

    Args:
        req: Typed `_S3PutRequest` containing endpoint URL, bucket, key, payload bytes, and credentials.

    Raises:
        RuntimeError: If the S3 gateway returns an HTTP 4xx/5xx status code.
    """
    parsed = urllib.parse.urlsplit(req.endpoint_url.rstrip("/"))
    bucket_seg = urllib.parse.quote(req.bucket.strip("/"), safe="~")
    key_seg = urllib.parse.quote(req.object_key.lstrip("/"), safe="/~")
    canonical_uri = f"{parsed.path.rstrip('/')}/{bucket_seg}/{key_seg}"
    target_url = f"{parsed.scheme}://{parsed.netloc}{canonical_uri}"

    now = datetime.now(UTC)
    amz_date = now.strftime("%Y%m%dT%H%M%SZ")
    date_stamp = now.strftime("%Y%m%d")
    payload_hash = hashlib.sha256(req.payload).hexdigest()

    canonical_headers = (
        f"content-type:{req.content_type}\n"
        f"host:{parsed.netloc}\n"
        f"x-amz-content-sha256:{payload_hash}\n"
        f"x-amz-date:{amz_date}\n"
    )
    signed_headers = "content-type;host;x-amz-content-sha256;x-amz-date"
    canonical_request = f"PUT\n{canonical_uri}\n\n{canonical_headers}\n{signed_headers}\n{payload_hash}"

    scope = f"{date_stamp}/{req.region}/s3/aws4_request"
    request_digest = hashlib.sha256(canonical_request.encode("utf-8")).hexdigest()
    string_to_sign = f"AWS4-HMAC-SHA256\n{amz_date}\n{scope}\n{request_digest}"
    signing_key = _derive_sigv4_signing_key(req.secret_key, date_stamp, req.region)
    signature = hmac.new(signing_key, string_to_sign.encode("utf-8"), hashlib.sha256).hexdigest()

    headers = {
        "Content-Type": req.content_type,
        "Host": parsed.netloc,
        "x-amz-content-sha256": payload_hash,
        "x-amz-date": amz_date,
        "Authorization": (
            f"AWS4-HMAC-SHA256 Credential={req.access_key}/{scope}, "
            f"SignedHeaders={signed_headers}, Signature={signature}"
        ),
    }
    http_req = urllib.request.Request(url=target_url, data=req.payload, headers=headers, method="PUT")
    try:
        with urllib.request.urlopen(http_req, timeout=req.timeout_seconds):
            return
    except urllib.error.HTTPError as http_err:
        body = http_err.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"DagsHub S3 PUT {target_url} failed with HTTP {http_err.code}: {body[:300]}") from http_err


def _parse_json_response(resp: Any, method: str, url: str) -> Any:
    """Decode an HTTP response body into JSON, rejecting HTML sleeping/redirect pages cleanly.

    Why:
        DagsHub's hosted Label Studio workspace returns HTTP 200 HTML pages when asleep or redirecting, causing raw `json.loads` to fail with an opaque `JSONDecodeError`. Validating `Content-Type` and wrapping decode failures yields an actionable `RuntimeError`.

    How:
        Decodes UTF-8 response bytes, checks `Content-Type` for `text/html`, parses non-empty JSON payloads, and wraps `JSONDecodeError` with endpoint context.

    Args:
        resp: Active HTTP response context returned by `urlopen`.
        method: HTTP verb used for the request.
        url: Target request URL for error context.

    Returns:
        Parsed JSON object/list, or `{}` on an empty response body.

    Raises:
        RuntimeError: If the server returns HTML or malformed JSON.
    """
    raw = resp.read().decode("utf-8", errors="replace")
    content_type = resp.headers.get("Content-Type", "") if resp.headers is not None else ""
    if not raw.strip():
        return {}
    if "text/html" in content_type.lower():
        raise RuntimeError(
            f"DagsHub Label Studio API {method} {url} returned HTML instead of JSON "
            f"(Content-Type: {content_type!r}; workspace may be asleep or redirecting): {raw[:120]!r}"
        )
    try:
        return json.loads(raw)
    except json.JSONDecodeError as json_err:
        raise RuntimeError(
            f"DagsHub Label Studio API {method} {url} returned invalid JSON "
            f"(Content-Type: {content_type!r}): {raw[:120]!r}"
        ) from json_err
