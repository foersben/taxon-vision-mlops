# SPDX-FileCopyrightText: 2026 Benjamin Förster
# SPDX-License-Identifier: MIT
"""Background observation sync and Airflow human-feedback merge orchestration.

Summary:
    Provides the FastAPI `BackgroundTasks` upload hook (`background_sync_flagged_observation`) and the Airflow `model_retraining` orchestration entrypoint (`sync_and_merge_human_feedback` with `FeedbackSyncPaths`).

Why (Architectural Rationale & Trade-offs):
    Decoupling workflow-level orchestration from the low-level DagsHub Label Studio REST client keeps each sibling module below 500 LOC (Rule 08) and allows Airflow DAG tasks and unit tests to exercise remote degradation and local Parquet merging cleanly without requiring Apache Airflow installed.

How (Algorithmic Mechanism & Invariants):
    `background_sync_flagged_observation` uploads consented captures off the `/predict` hot path inside a logged exception boundary. `sync_and_merge_human_feedback` flushes pending captures and exports completed annotations, gracefully degrading on remote `RuntimeError` failures whilst propagating local filesystem or schema errors.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from taxon_vision.active_learning.label_studio_sync import DagsHubLabelStudioClient

logger = logging.getLogger(__name__)


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
    from taxon_vision.active_learning.label_studio_sync import DagsHubLabelStudioClient

    try:
        client = DagsHubLabelStudioClient()
        if client.is_configured:
            client.sync_flagged_observation(meta_path, crop_path)
    except Exception:
        logger.exception("Background DagsHub Label Studio sync failed for %s", meta_path)


@dataclass(frozen=True)
class FeedbackSyncPaths:
    """Filesystem path bundle for active-learning Label Studio synchronisation and manifest merging.

    Why:
        Encapsulates the four runtime PVC or local repository paths consumed by `sync_and_merge_human_feedback` into a single immutable value object, complying with Rule 08 parameter arity limits (<= 4 positional arguments).

    How:
        Constructed by Airflow's `model_retraining` DAG or unit tests and passed directly to `sync_and_merge_human_feedback`.

    Attributes:
        flagged_dir: Directory holding pending locally captured flagged observations.
        export_path: Target JSON path for completed Label Studio annotations.
        raw_manifest_path: Target Parquet path for `raw_manifest.parquet`.
        catalog_path: Path to `config/taxa_catalog.yaml`.
    """

    flagged_dir: Path
    export_path: Path
    raw_manifest_path: Path
    catalog_path: Path


def _empty_feedback_summary(status: str, synced_pending: int = 0) -> dict[str, Any]:
    """Build a uniform zero-merge XCom summary payload when no feedback export can be merged.

    Why:
        Guarantees downstream Airflow tasks (`trigger_jenkins_build`) always receive the expected schema keys (`catalog_expanded`, `new_annotations_merged`, `status`) even when remote Label Studio is unavailable or unconfigured.

    How:
        Returns a deterministic dictionary with zero counts and `catalog_expanded=False`.

    Args:
        status: Machine-readable skip status string.
        synced_pending: Count of pending captures uploaded prior to any export error.

    Returns:
        Dictionary suitable for Airflow XCom serialisation.
    """
    return {
        "new_annotations_merged": 0,
        "catalog_expanded": False,
        "new_taxon_ids": [],
        "synced_pending": synced_pending,
        "status": status,
    }


def sync_and_merge_human_feedback(
    paths: FeedbackSyncPaths,
    client: DagsHubLabelStudioClient | None = None,
) -> dict[str, Any]:
    """Synchronise pending captures with DagsHub Label Studio and merge completed annotations into `raw_manifest.parquet`.

    Why:
        Extracting this orchestration out of Airflow's `@task` closure makes the exact production control flow directly testable in unit tests without requiring Apache Airflow installed. Furthermore, catching remote `RuntimeError` failures prevents a sleeping DagsHub Label Studio workspace from blocking drift-triggered GPU retraining whilst preserving fail-fast propagation for local filesystem permission or schema errors.

    How:
        1. If the DagsHub client is configured, flushes consented captures from `paths.flagged_dir` and exports completed annotations to `paths.export_path`, degrading gracefully with a warning on remote `RuntimeError` failures.
        2. If `paths.export_path` does not exist locally (either unconfigured or remote unavailable), returns a typed skip summary so Jenkins retraining still proceeds.
        3. Invokes `merge_feedback_to_manifest` against `paths.raw_manifest_path` and returns merged counts and `catalog_expanded` metadata.

    Args:
        paths: `FeedbackSyncPaths` specifying flagged captures, export JSON, raw manifest, and taxa catalogue.
        client: Optional explicit `DagsHubLabelStudioClient` instance (defaults to environment-configured client).

    Returns:
        Dictionary summarising merged record counts, OOD audit counts, newly registered taxon IDs, `catalog_expanded`, and status.
    """
    from taxon_vision.active_learning.feedback import merge_feedback_to_manifest
    from taxon_vision.active_learning.label_studio_sync import DagsHubLabelStudioClient

    ls_client = client or DagsHubLabelStudioClient()
    synced_pending = 0
    remote_unavailable = False

    if ls_client.is_configured:
        try:
            synced_pending = ls_client.sync_pending_directory(paths.flagged_dir)
            exported_count = ls_client.export_completed_annotations(paths.export_path)
            logger.info(
                "DagsHub sync complete (uploaded_pending=%d, exported_completed=%d).", synced_pending, exported_count
            )
        except RuntimeError as err:
            remote_unavailable = True
            logger.warning("DagsHub Label Studio unavailable; continuing with local fallback if present: %s", err)

    if not paths.export_path.is_file():
        skip_status = "skipped_dagshub_unavailable" if remote_unavailable else "skipped_no_pending_annotations"
        logger.info("No local export at %s (%s); proceeding without feedback merge.", paths.export_path, skip_status)
        return _empty_feedback_summary(skip_status, synced_pending=synced_pending)

    res = merge_feedback_to_manifest(
        paths.export_path, manifest_path=paths.raw_manifest_path, catalog_path=paths.catalog_path
    )
    logger.info(
        "Merged %d human annotations into %s (status=%s, total=%d, catalog_expanded=%s, ood_verified=%d)",
        res.new_records_added,
        paths.raw_manifest_path,
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
