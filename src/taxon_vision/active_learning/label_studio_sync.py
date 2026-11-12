# SPDX-FileCopyrightText: 2026 Benjamin Förster
# SPDX-License-Identifier: MIT
"""DagsHub Label Studio and S3 repository bucket synchronisation client.

Summary:
    Synchronises open-license-consented triage observations (image crops, uncertainty scores, and attribution metadata) to DagsHub S3 and DagsHub Label Studio, and exports completed expert taxonomic annotations back to the MLOps retraining pipeline.

Why (Architectural Rationale & Trade-offs):
    Hosting a separate Label Studio instance inside K3s duplicates PostgreSQL, authentication, and object-storage administration. Integrating directly with DagsHub's managed Label Studio REST API (`/annotations/de/api`) and repository S3 bucket (`https://dagshub.com/api/v1/repo-buckets/s3/<owner>`) eliminates stateful self-hosted maintenance whilst keeping active-learning tasks accessible over HTTPS. Using Python standard library `urllib.request` for REST operations guarantees compatibility across both serving pods and lightweight Airflow worker images without extra dependencies.

How (Algorithmic Mechanism & Invariants):
    1. Resolves repository coordinates (`foersben/taxon-vision-mlops`) and authentication credential `DAGSHUB_TOKEN` from config/environment.
    2. Uploads consented `{obs_id}_crop.jpg` bytes into the repository S3 bucket prefix `triage-queue/` via AWS SigV4-signed `urllib.request` HTTP `PUT` (zero external S3 client dependencies).
    3. Ensures the target DagsHub Label Studio project exists and imports priority-ranked tasks carrying `observation_uuid`, `photographer_name`, `license_code`, `license_url`, `uncertainty_score`, and `ood_energy_score`.
    4. Exports completed annotations into `label_studio_export.json` and synchronises updated XML label configurations when new taxa expand `config/taxa_catalog.yaml`.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from taxon_vision.active_learning.feedback import PERMITTED_LICENSES
from taxon_vision.active_learning.triage_queue import TriageItem, TriageQueue

logger = logging.getLogger(__name__)


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
            Endpoint URL passed to `boto3.client("s3", endpoint_url=...)`.
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


class DagsHubLabelStudioClient:
    """REST and S3 client for DagsHub-hosted Label Studio active learning workflows.

    Why:
        Centralises authentication, idempotent project resolution, S3 crop uploads, and JSON annotation exports so inference workers and Airflow DAGs share one verified contract without duplicated HTTP boilerplate.

    How:
        Wraps DagsHub REST endpoints (`urllib.request`) with Bearer token authentication and lazy `boto3` S3 client instantiation for uploading image crops under `triage-queue/`.
    """

    def __init__(self, config: DagsHubRepoConfig | None = None) -> None:
        """Initialise the DagsHub Label Studio client with repository credentials.

        Why:
            Allows deterministic credential injection during unit tests whilst automatically resolving pod environment secrets (`DAGSHUB_TOKEN`) in production services and Airflow workers.

        How:
            Stores the provided `config` or constructs one via `build_dagshub_repo_config()`.

        Args:
            config: Optional explicit `DagsHubRepoConfig`; resolves from environment when omitted.
        """
        self.config = config or build_dagshub_repo_config()

    @property
    def is_configured(self) -> bool:
        """Return True when a non-empty DagsHub authentication token is available.

        Why:
            Provides an explicit guard so air-gapped environments, unit tests, and local cold starts degrade cleanly to disk-only triage retention without issuing failing network calls.

        How:
            Evaluates boolean truthiness of `self.config.token`.

        Returns:
            True when DagsHub REST and S3 calls can authenticate, False otherwise.
        """
        return bool(self.config.token)

    def _api_request(
        self,
        method: str,
        path: str,
        payload: Any | None = None,
    ) -> Any:
        """Execute an authenticated JSON HTTP request against the DagsHub Label Studio API.

        Why:
            Using standard-library `urllib.request` with uniform Bearer token injection, timeout enforcement, and HTTP error context wrapping avoids adding third-party SDK dependencies inside slim Airflow containers.

        How:
            Serialises `payload` as UTF-8 JSON when present, attaches Bearer and `User-Agent` headers, executes the request with `self.config.timeout_seconds`, and parses non-empty JSON responses.

        Args:
            method: HTTP verb (`GET`, `POST`, or `PATCH`).
            path: Relative path appended to `self.config.label_studio_api_base`.
            payload: Optional JSON-serialisable request body.

        Returns:
            Parsed JSON object or list returned by the API, or `{}` on an empty response.

        Raises:
            RuntimeError: If `DAGSHUB_TOKEN` is unset or the server returns HTTP 4xx/5xx.
        """
        if not self.is_configured:
            raise RuntimeError("DAGSHUB_TOKEN is not configured for DagsHub Label Studio API access.")

        url = f"{self.config.label_studio_api_base}/{path.lstrip('/')}"
        headers = {
            "Authorization": f"Bearer {self.config.token}",
            "Accept": "application/json",
            "User-Agent": "TaxonVision-ActiveLearning/1.0",
        }
        data_bytes: bytes | None = None
        if payload is not None:
            headers["Content-Type"] = "application/json"
            data_bytes = json.dumps(payload).encode("utf-8")

        req = urllib.request.Request(url=url, data=data_bytes, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=self.config.timeout_seconds) as resp:
                raw = resp.read().decode("utf-8")
                return json.loads(raw) if raw.strip() else {}
        except urllib.error.HTTPError as http_err:
            body = http_err.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"DagsHub Label Studio API {method} {url} failed with HTTP {http_err.code}: {body[:300]}"
            ) from http_err

    def resolve_or_create_project_id(
        self, label_config_path: Path | str = "deploy/label_studio/project_config.xml"
    ) -> int:
        """Resolve an active DagsHub Label Studio project ID, bootstrapping project and S3 storage if absent.

        Why:
            Eliminates fragile manual ID discovery on fresh environments by idempotently locating the configured TaxonVision annotation project or provisioning it from `project_config.xml`.

        How:
            Returns `self.config.project_id` if explicitly set; otherwise searches `GET projects/` by `project_title` (or selects the first existing project), and creates a new project followed by `_ensure_s3_source_storage()` if none exist.

        Args:
            label_config_path: Path to the Label Studio XML interface template.

        Returns:
            Numeric DagsHub Label Studio project identifier.
        """
        if self.config.project_id is not None:
            return self.config.project_id

        projects_payload = self._api_request("GET", "projects/")
        results = (
            projects_payload.get("results", projects_payload)
            if isinstance(projects_payload, dict)
            else projects_payload
        )
        if isinstance(results, list) and results:
            for proj in results:
                if str(proj.get("title", "")).strip() == self.config.project_title:
                    return int(proj["id"])
            return int(results[0]["id"])

        xml_file = Path(label_config_path)
        label_xml = (
            xml_file.read_text(encoding="utf-8")
            if xml_file.is_file()
            else '<View><Image name="image" value="$image_url"/></View>'
        )
        created = self._api_request(
            "POST",
            "projects/",
            payload={"title": self.config.project_title, "label_config": label_xml},
        )
        pid = int(created["id"])
        self._ensure_s3_source_storage(pid)
        return pid

    def _ensure_s3_source_storage(self, project_id: int) -> None:
        """Register the DagsHub S3 repository bucket as a backend-proxied source storage for Label Studio.

        Why:
            DagsHub's `/api/v1/repo-buckets/s3/` gateway rejects AWS SigV4 query-string presigned URLs with HTTP 401 Unauthorized, whereas setting `presign=False` causes Label Studio's backend to stream `s3://taxon-vision-mlops/triage-queue/...` objects using SigV4 `Authorization` headers through `/tasks/{id}/resolve/` without making private buckets public.

        How:
            Issues `POST storages/s3` configured with DagsHub's S3 endpoint, repository name, `triage-queue/` prefix, and `presign=False`, logging a warning if the endpoint rejects duplicate registration.

        Args:
            project_id: Target DagsHub Label Studio numeric project identifier.
        """
        try:
            self._api_request(
                "POST",
                "storages/s3",
                payload={
                    "project": project_id,
                    "title": "DagsHub Triage S3 Storage",
                    "bucket": self.config.repo,
                    "prefix": f"{self.config.bucket_prefix}/",
                    "region_name": "us-east-1",
                    "s3_endpoint": self.config.s3_endpoint_url,
                    "aws_access_key_id": self.config.token,
                    "aws_secret_access_key": self.config.token,
                    "presign": False,
                    "recursive_scan": True,
                },
            )
        except Exception as err:
            logger.warning(
                "Could not auto-configure Label Studio S3 source storage for project %d: %s", project_id, err
            )

    def _data_engine_graphql(self, query: str) -> dict[str, Any]:
        """Execute an authenticated GraphQL request against DagsHub Data Engine.

        Why:
            Keeps Data Engine datasource indexing coupled cleanly inside standard-library HTTPS calls so newly uploaded S3 triage crops appear in DagsHub Datasources without requiring manual UI scans or extra SDK packages.

        How:
            Sends a UTF-8 JSON GraphQL payload to `/api/v1/repos/{owner}/{repo}/data-engine/graphql` with `Authorization: Bearer` headers and returns the parsed JSON body.

        Args:
            query: GraphQL query or mutation string.

        Returns:
            Parsed GraphQL response dictionary.
        """
        endpoint = f"https://dagshub.com/api/v1/repos/{self.config.owner}/{self.config.repo}/data-engine/graphql"
        req = urllib.request.Request(
            url=endpoint,
            data=json.dumps({"query": query}).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.config.token}",
                "Content-Type": "application/json",
                "User-Agent": "TaxonVision-ActiveLearning/1.0",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.config.timeout_seconds) as resp:
            raw = resp.read().decode("utf-8")
            parsed: dict[str, Any] = json.loads(raw) if raw.strip() else {}
            return parsed

    def _refresh_matching_datasource(self) -> None:
        """Trigger an asynchronous Data Engine bucket scan for the matching triage datasource.

        Why:
            Objects written directly via DagsHub's S3 gateway do not appear inside Data Engine datasource views until a bucket rescan runs. Triggering `scanDatasource` best-effort keeps DagsHub's Datasets UI synchronised with Label Studio tasks.

        How:
            Queries matching datasources named after `self.config.bucket_prefix` and invokes `scanDatasource` with `FORCE_REGENERATE_AUTO_SCAN_VALUES`, degrading softly with a warning if GraphQL is unreachable.
        """
        target_name = self.config.bucket_prefix.strip("/")
        try:
            listing = self._data_engine_graphql("{ datasource { id name } }")
            sources = listing.get("data", {}).get("datasource") or []
            for src in sources:
                if isinstance(src, dict) and str(src.get("name", "")).strip() == target_name:
                    ds_id = int(src["id"])
                    self._data_engine_graphql(
                        f"mutation {{ scanDatasource(id: {ds_id}, options: [FORCE_REGENERATE_AUTO_SCAN_VALUES]) {{ id preprocessingStatus }} }}"
                    )
        except Exception as err:
            logger.debug("Skipping Data Engine datasource refresh for %s: %s", target_name, err)

    def upload_crop_to_bucket(self, observation_id: str, crop_path: Path) -> str:
        """Upload a consented flagged observation crop to the DagsHub S3 repository bucket.

        Why:
            Presigned `s3://` URIs hosted in the DagsHub repository bucket remain accessible to both browser-based Label Studio annotators and headless Jenkins Stage 3 feature extraction workers without persisting raw imagery in Git. Signing HTTP `PUT` requests directly with AWS SigV4 in Python standard library eliminates runtime `boto3` dependencies inside trimmed serving containers.

        How:
            Constructs an `_S3PutRequest` targeting `self.config.s3_endpoint_url/{repo}/{bucket_prefix}/{observation_id}_crop.jpg`, signs canonical headers using HMAC-SHA256 (`AWS4-HMAC-SHA256`), and issues the `PUT` via `urllib.request.urlopen`.

        Args:
            observation_id: Unique UUID identifier of the observation.
            crop_path: Local filesystem path to the JPEG image crop.

        Returns:
            Canonical S3 URI targeting the uploaded object.

        Raises:
            RuntimeError: If `self.is_configured` is False or the HTTP upload returns an error response.
        """
        if not self.is_configured:
            raise RuntimeError("DAGSHUB_TOKEN is required to upload triage crops to DagsHub S3.")

        object_key = f"{self.config.bucket_prefix.strip('/')}/{observation_id}_crop.jpg"
        _s3_put_object(
            _S3PutRequest(
                endpoint_url=self.config.s3_endpoint_url,
                bucket=self.config.repo,
                object_key=object_key,
                payload=crop_path.read_bytes(),
                access_key=self.config.token,
                secret_key=self.config.token,
                timeout_seconds=self.config.timeout_seconds,
            )
        )
        return f"s3://{self.config.repo}/{object_key}"

    def sync_flagged_observation(self, meta_path: Path, crop_path: Path) -> bool:
        """Upload one open-licensed flagged observation crop and import its triage task into DagsHub Label Studio.

        Why:
            Enforces Three-Tier Legal Compliance prior to remote synchronisation: observations lacking explicit open-license consent (`CC0`, `CC-BY`, `CC-BY-NC`) or valid photographer attribution are strictly blocked from external annotation upload.

        How:
            1. Reads `{obs_id}_meta.json` and exits idempotently if `synced_to_label_studio` is already `True`.
            2. Validates normalised `license_code` against `PERMITTED_LICENSES` and `len(photographer_name) >= 3`.
            3. Uploads the image crop to DagsHub S3 via `upload_crop_to_bucket`, imports a pre-annotated Label Studio task carrying conformal sets and energy scores, and stamps `synced_to_label_studio: True` plus `image_url` atomically back into `meta_path`.

        Args:
            meta_path: Path to `{obs_id}_meta.json` containing inference and licensing metadata.
            crop_path: Path to `{obs_id}_crop.jpg` binary image file.

        Returns:
            True when successfully uploaded and imported into DagsHub Label Studio, False if skipped or unconfigured.
        """
        if not self.is_configured:
            logger.debug("Skipping DagsHub Label Studio sync for %s: DAGSHUB_TOKEN unset.", meta_path.name)
            return False
        if not meta_path.is_file() or not crop_path.is_file():
            return False

        record: dict[str, Any] = json.loads(meta_path.read_text(encoding="utf-8"))
        if record.get("synced_to_label_studio"):
            return True

        license_norm = str(record.get("license_code", "")).strip().lower().replace("_", "-")
        photographer = str(record.get("photographer_name", "")).strip()
        if license_norm not in PERMITTED_LICENSES or len(photographer) < 3:
            logger.info(
                "Observation %s retained locally only (no open-license opt-in/attribution for DagsHub export).",
                record.get("observation_id"),
            )
            return False

        obs_id = str(record["observation_id"])
        s3_uri = self.upload_crop_to_bucket(obs_id, crop_path)
        project_id = self.resolve_or_create_project_id()

        task_payload = [
            {
                "data": {
                    "observation_uuid": f"label_studio_{obs_id}",
                    "image_url": s3_uri,
                    "photographer_name": photographer,
                    "license_code": license_norm.upper(),
                    "license_url": str(record.get("license_url") or "https://creativecommons.org/licenses/"),
                    "top_prediction": record.get("top_prediction", {}),
                    "conformal_set": record.get("conformal_set", []),
                    "ood_energy_score": float(record.get("ood_energy_score", 0.0)),
                    "uncertainty_score": float(record.get("uncertainty_score", 0.0)),
                }
            }
        ]
        self._api_request("POST", f"projects/{project_id}/import", payload=task_payload)
        self._refresh_matching_datasource()

        record["synced_to_label_studio"] = True
        record["image_url"] = s3_uri
        meta_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
        logger.info("Synchronised observation %s to DagsHub Label Studio project %d", obs_id, project_id)
        return True

    def sync_pending_directory(self, flagged_dir: Path | str) -> int:
        """Flush all pending local consented observations in flagged_dir ordered by uncertainty descending.

        Why:
            During transient network partitions or cold starts before DAGSHUB_TOKEN is provisioned, consented observations accumulate on the local persistent volume; flushing in TriageQueue priority order ensures highest-uncertainty specimens reach human annotators first.

        How:
            Enqueues unsynchronised local captures into TriageQueue via _collect_pending_triage_items and sequentially dispatches sync_flagged_observation in descending margin uncertainty order.

        Args:
            flagged_dir: Directory path containing local `{obs_id}_meta.json` and `{obs_id}_crop.jpg` pairs.

        Returns:
            Count of observations successfully uploaded and imported into DagsHub Label Studio.
        """
        target_dir = Path(flagged_dir)
        if not target_dir.is_dir() or not self.is_configured:
            return 0

        synced = 0
        for candidate in _collect_pending_triage_items(target_dir):
            try:
                if self.sync_flagged_observation(candidate.meta_path, candidate.crop_path):
                    synced += 1
            except Exception:
                logger.exception("Failed syncing pending triage item %s", candidate.meta_path.name)
        return synced

    def export_completed_annotations(self, export_path: Path | str) -> int:
        """Pull completed taxonomic annotations from DagsHub Label Studio to a local JSON export file.

        Why:
            Bridges remote expert annotations back into the local Airflow/DVC retraining pipeline without manual file downloads from the DagsHub web interface.

        How:
            Resolves the active project ID, queries `projects/{id}/export?exportType=JSON`, filters tasks containing non-empty `annotations`, and writes the JSON array to export_path.

        Args:
            export_path: Destination JSON file path consumed by feedback merger.

        Returns:
            Number of annotated tasks exported from DagsHub Label Studio.
        """
        if not self.is_configured:
            logger.warning("DAGSHUB_TOKEN unset; skipping remote DagsHub Label Studio annotation export.")
            return 0

        project_id = self.resolve_or_create_project_id()
        exported = self._api_request("GET", f"projects/{project_id}/export?exportType=JSON")
        tasks = exported if isinstance(exported, list) else []
        completed = [t for t in tasks if isinstance(t, dict) and t.get("annotations")]

        dest = Path(export_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(completed, indent=2), encoding="utf-8")
        logger.info(
            "Exported %d completed Label Studio annotations from project %d to %s", len(completed), project_id, dest
        )
        return len(completed)

    def push_project_label_config(self, xml_path: Path | str = "deploy/label_studio/project_config.xml") -> bool:
        """Push an updated XML label configuration to the DagsHub Label Studio project after taxonomy expansion.

        Why:
            When Outcome B admits a new species into `config/taxa_catalog.yaml` and `deploy/label_studio/project_config.xml`, synchronising the updated XML ensures subsequent triage tasks display the newly registered species as a selectable choice.

        How:
            Reads the updated XML file from disk, resolves the DagsHub Label Studio project ID, and issues an authenticated PATCH request updating `label_config`.

        Args:
            xml_path: Filesystem path to the updated Label Studio XML template.

        Returns:
            True when the remote project configuration was updated, False otherwise.
        """
        if not self.is_configured:
            return False
        xml_file = Path(xml_path)
        if not xml_file.is_file():
            return False
        project_id = self.resolve_or_create_project_id(xml_file)
        self._api_request(
            "PATCH",
            f"projects/{project_id}/",
            payload={"label_config": xml_file.read_text(encoding="utf-8")},
        )
        logger.info("Updated DagsHub Label Studio project %d label configuration from %s", project_id, xml_file)
        return True


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


@dataclass(frozen=True)
class _PendingTriageItem:
    """Unsynchronised local capture candidate ranked by margin uncertainty.

    Why:
        Pairs the priority scalar with both filesystem paths so items popped from `TriageQueue` resolve directly to their `(meta_path, crop_path)` files without directory re-scans.

    How:
        Constructed by `_collect_pending_triage_items` for every verified unsynchronised capture pair found on the PVC.

    Attributes:
        score: Margin uncertainty scalar (`1.0 - (p1 - p2)`).
        meta_path: Filesystem path to `{obs_id}_meta.json`.
        crop_path: Filesystem path to `{obs_id}_crop.jpg`.
    """

    score: float
    meta_path: Path
    crop_path: Path


def _collect_pending_triage_items(target_dir: Path) -> list[_PendingTriageItem]:
    """Scan local flagged captures and order unsynchronised open-licensed crop pairs via TriageQueue.

    Why:
        When backlogged observations accumulate offline on `/var/lib/taxon-vision/flagged_observations`, uploading them in strict priority order guarantees that the highest-uncertainty specimens enter the human review queue first under finite expert annotation bandwidth.

    How:
        Globs `*_meta.json`, skips malformed or already synchronised records, verifies crop existence, pushes `TriageItem` entries into a max-heap `TriageQueue` ordered by uncertainty score, and drains items in descending priority order.

    Args:
        target_dir: Local directory holding `{obs_id}_meta.json` and `{obs_id}_crop.jpg` pairs.

    Returns:
        List of `_PendingTriageItem` entries sorted from highest to lowest margin uncertainty.
    """
    queue = TriageQueue()
    by_id: dict[str, _PendingTriageItem] = {}
    for meta_path in target_dir.glob("*_meta.json"):
        try:
            data = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if data.get("synced_to_label_studio"):
            continue
        obs_id = str(data.get("observation_id") or meta_path.stem)
        crop_path = target_dir / str(data.get("image_path") or f"{obs_id}_crop.jpg")
        if crop_path.is_file():
            score = float(data.get("uncertainty_score", 0.0))
            set_size = len(data.get("conformal_set", []))
            by_id[obs_id] = _PendingTriageItem(score, meta_path, crop_path)
            queue.push(TriageItem(obs_id, score, set_size, str(crop_path)))

    ordered: list[_PendingTriageItem] = []
    while (item := queue.pop()) is not None:
        if item.observation_id in by_id:
            ordered.append(by_id[item.observation_id])
    return ordered


def background_sync_flagged_observation(meta_path: Path, crop_path: Path) -> None:
    """FastAPI BackgroundTasks entrypoint to upload a consented observation crop to DagsHub Label Studio cleanly.

    Why:
        Executes network I/O to DagsHub S3 and Label Studio outside the critical `/predict` HTTP response path, preserving the sub-25ms p95 microservice latency budget under Rule 02 and Rule 08.

    How:
        Instantiates DagsHubLabelStudioClient, checks `is_configured`, and invokes `sync_flagged_observation` inside a fail-safe exception boundary with structured traceback logging.

    Args:
        meta_path: Path to `{obs_id}_meta.json` on disk.
        crop_path: Path to `{obs_id}_crop.jpg` on disk.
    """
    try:
        client = DagsHubLabelStudioClient()
        if client.is_configured:
            client.sync_flagged_observation(meta_path, crop_path)
    except Exception:
        logger.exception("Background DagsHub Label Studio sync failed for %s", meta_path)
