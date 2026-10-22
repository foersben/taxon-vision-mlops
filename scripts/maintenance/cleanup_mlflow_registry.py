#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Benjamin Förster
# SPDX-License-Identifier: MIT
"""MLflow Model Registry Maintenance & Version Pruning Utility.

Purges unaliased, stale model versions accumulated from automated CI/CD runs while preserving active alias pointers (@Production, @Challenger), top-performing versions by validation metric, and recent deployment candidates.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time

import mlflow
from mlflow.entities.model_registry import ModelVersion
from mlflow.tracking import MlflowClient

from taxon_vision.config import get_settings

logger = logging.getLogger("mlflow_registry_cleaner")


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments with dry-run enabled by default.

    Args:
        argv: Optional list of argument strings.

    Returns:
        Parsed Namespace container.
    """
    settings = get_settings()
    default_uri = (
        os.environ.get("MLFLOW_TRACKING_URI", settings.mlflow.tracking_uri)
        or "https://dagshub.com/foersben/taxon-vision-mlops.mlflow"
    )

    parser = argparse.ArgumentParser(description="Clean up unaliased and low-performing model versions in MLflow.")
    parser.add_argument(
        "--model-name",
        type=str,
        default="taxon_vision_classifier",
        help="Name of the registered model in MLflow (default: taxon_vision_classifier).",
    )
    parser.add_argument(
        "--tracking-uri",
        type=str,
        default=default_uri,
        help=f"Remote MLflow tracking URI (default: {default_uri}).",
    )
    parser.add_argument(
        "--keep-last",
        type=int,
        default=10,
        help="Number of most recent versions to retain unconditionally (default: 10).",
    )
    parser.add_argument(
        "--keep-top-k",
        type=int,
        default=5,
        help="Number of highest-scoring versions to preserve based on validation metric (default: 5).",
    )
    parser.add_argument(
        "--metric-key",
        type=str,
        default="val_pr_auc",
        help="Tag or metric key to rank model performance (default: val_pr_auc).",
    )
    parser.add_argument(
        "--request-delay",
        type=float,
        default=0.15,
        help="Delay in seconds between deletion requests to avoid API rate limits (default: 0.15s).",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        default=False,
        help="Execute deletions. If omitted, runs in safe dry-run mode.",
    )
    return parser.parse_args(argv)


def fetch_protected_aliases(client: MlflowClient, model_name: str) -> set[str]:
    """Retrieve all model versions currently assigned an active alias.

    Why:
        Production stability requires that models referenced by deployment aliases
        (@Production, @Challenger, or custom staging aliases) are never deleted.

    How:
        Queries the registered model entity, iterates its alias mappings, and extracts
        the assigned version numbers into a set.

    Args:
        client: Active MlflowClient instance.
        model_name: Name of the registered model.

    Returns:
        Set of protected model version strings.
    """
    protected: set[str] = set()
    try:
        registered_model = client.get_registered_model(model_name)
        if registered_model.aliases:
            for alias_name, version_str in registered_model.aliases.items():
                logger.info("Preserving alias @%s -> Version %s", alias_name, version_str)
                protected.add(str(version_str))
    except Exception as exc:
        logger.warning("Could not fetch registered model aliases for '%s': %s", model_name, exc)
    return protected


def fetch_all_model_versions(client: MlflowClient, model_name: str) -> list[ModelVersion]:
    """Fetch all model versions across paginated MLflow API responses.

    Why:
        Repositories accumulating hundreds of automated trial and canary runs exceed
        the default single-page API limit (typically 100-500 versions). Exhaustive
        retrieval requires iterating through pagination tokens.

    How:
        Invokes search_model_versions in a loop with max_results=500, appending
        paged results until the pagination token is empty.

    Args:
        client: Active MlflowClient instance.
        model_name: Registered model identifier string.

    Returns:
        Complete list of ModelVersion entities.
    """
    versions: list[ModelVersion] = []
    page_token: str | None = None

    while True:
        paged_list = client.search_model_versions(
            filter_string=f"name = '{model_name}'",
            max_results=500,
            page_token=page_token,
        )
        versions.extend(paged_list)
        page_token = getattr(paged_list, "token", None)
        if not page_token:
            break

    logger.info("Retrieved %d total versions for model '%s'", len(versions), model_name)
    return versions


def extract_metric_score(version: ModelVersion, metric_key: str) -> float:
    """Extract metric float from version tags or default to 0.0.

    Args:
        version: ModelVersion entity.
        metric_key: Tag key containing scalar performance score.

    Returns:
        Parsed float score, or 0.0 if missing or invalid.
    """
    tags = version.tags or {}
    val = tags.get(metric_key)
    if val is None:
        return 0.0
    try:
        return float(val)
    except ValueError:
        return 0.0


def identify_protected_versions(
    versions: list[ModelVersion],
    protected_aliases: set[str],
    keep_last: int,
    keep_top_k: int,
    metric_key: str,
) -> set[str]:
    """Calculate the union of alias-protected, top-performing, and recent versions.

    Why:
        Multi-tier protection guarantees zero accidental service outages by guarding:
        1. All aliased versions (@Production, @Challenger).
        2. Legacy staged versions (Production, Staging).
        3. The N newest versions to prevent deleting in-flight deployments.
        4. The K highest-scoring historical models for offline benchmarking.

    How:
        Constructs a protection set by taking the union across aliases, legacy stages, recency sort, and metric ranking sort.

    Args:
        versions: List of all ModelVersion instances.
        protected_aliases: Set of version strings referenced by aliases.
        keep_last: Number of most recent versions to retain unconditionally.
        keep_top_k: Number of highest-scoring versions to preserve.
        metric_key: Tag key identifying the sorting metric.

    Returns:
        Set of protected model version strings.
    """
    protected: set[str] = set(protected_aliases)

    # 1. Protect any model currently in Production or Staging legacy stages
    for v in versions:
        if v.current_stage in ("Production", "Staging"):
            protected.add(str(v.version))

    # 2. Protect the N most recent versions (sorted by version integer descending)
    sorted_by_recency = sorted(
        versions,
        key=lambda x: int(x.version) if str(x.version).isdigit() else 0,
        reverse=True,
    )
    for v in sorted_by_recency[:keep_last]:
        protected.add(str(v.version))

    # 3. Protect the top K versions by validation metric
    sorted_by_metric = sorted(
        versions,
        key=lambda x: extract_metric_score(x, metric_key),
        reverse=True,
    )
    for v in sorted_by_metric[:keep_top_k]:
        score = extract_metric_score(v, metric_key)
        logger.info(
            "Preserving high-score Version %s (%s: %.4f)",
            v.version,
            metric_key,
            score,
        )
        protected.add(str(v.version))

    return protected


def delete_version_safely(
    client: MlflowClient,
    model_name: str,
    version: str,
    request_delay: float,
) -> bool:
    """Delete a single model version with error handling and rate-limit delay.

    Why:
        Remote managed MLflow servers (such as DagsHub) strictly enforce API rate limits. Rapid back-to-back DELETE calls trigger HTTP 429 errors. Adding a micro-delay ensures safe, reliable batch cleanup.

    Args:
        client: Active MlflowClient instance.
        model_name: Name of the registered model.
        version: Version identifier string.
        request_delay: Pause duration in seconds between calls.

    Returns:
        True if deletion succeeded; False otherwise.
    """
    try:
        client.delete_model_version(name=model_name, version=version)
        if request_delay > 0:
            time.sleep(request_delay)
        return True
    except Exception as exc:
        logger.error("Failed to delete version %s: %s", version, exc)
        return False


def execute_pruning(
    client: MlflowClient,
    model_name: str,
    candidates: list[ModelVersion],
    dry_run: bool,
    request_delay: float,
) -> int:
    """Execute deletion loop or report planned deletions in dry-run mode.

    Args:
        client: Active MlflowClient instance.
        model_name: Name of the registered model.
        candidates: List of ModelVersion entities designated for deletion.
        dry_run: If True, previews actions without calling the deletion API.
        request_delay: Delay in seconds between deletion requests.

    Returns:
        Count of deleted (or previewed) model versions.
    """
    total = len(candidates)
    if dry_run:
        logger.info("[DRY-RUN] Would delete %d model versions.", total)
        for v in candidates[:15]:
            logger.info("  [DRY-RUN] Pending deletion: Version %s (tags: %s)", v.version, v.tags)
        if total > 15:
            logger.info("  ... and %d more versions.", total - 15)
        logger.info("Rerun with --execute to commit deletions.")
        return total

    logger.warning("Commencing deletion of %d model versions...", total)
    deleted_count = 0
    for idx, v in enumerate(candidates, start=1):
        success = delete_version_safely(client, model_name, str(v.version), request_delay)
        if success:
            deleted_count += 1
        if idx % 50 == 0 or idx == total:
            logger.info("Progress: %d/%d versions deleted.", deleted_count, total)

    logger.info("Successfully pruned %d versions from '%s'.", deleted_count, model_name)
    return deleted_count


def cleanup_registry(
    model_name: str = "taxon_vision_classifier",
    tracking_uri: str | None = None,
    keep_last: int = 10,
    keep_top_k: int = 5,
    metric_key: str = "val_pr_auc",
    request_delay: float = 0.15,
    execute: bool = False,
) -> int:
    """Programmatic entrypoint to clean up unaliased and stale model versions.

    Args:
        model_name: Name of the registered model.
        tracking_uri: Optional custom tracking URI.
        keep_last: Number of most recent versions to retain unconditionally.
        keep_top_k: Number of top-performing versions to retain by metric.
        metric_key: Tag key for metric score comparison.
        request_delay: Delay in seconds between deletion requests.
        execute: True to apply deletions; False for safe dry-run.

    Returns:
        Number of versions deleted or identified for deletion.
    """
    resolved_uri = tracking_uri or os.environ.get("MLFLOW_TRACKING_URI", get_settings().mlflow.tracking_uri)
    if resolved_uri:
        mlflow.set_tracking_uri(resolved_uri)
    client = MlflowClient(tracking_uri=resolved_uri)

    logger.info("Target Model: %s", model_name)
    logger.info("Tracking URI: %s", resolved_uri)
    logger.info("Execution Mode: %s", "LIVE EXECUTION" if execute else "DRY-RUN")

    try:
        all_versions = fetch_all_model_versions(client, model_name)
    except Exception as exc:
        logger.error("Failed to connect to MLflow or model does not exist: %s", exc)
        return 0

    if not all_versions:
        logger.info("No versions found for model '%s'. Exiting.", model_name)
        return 0

    protected_aliases = fetch_protected_aliases(client, model_name)
    all_protected = identify_protected_versions(
        versions=all_versions,
        protected_aliases=protected_aliases,
        keep_last=keep_last,
        keep_top_k=keep_top_k,
        metric_key=metric_key,
    )

    deletable_versions = [v for v in all_versions if str(v.version) not in all_protected]
    deletable_versions.sort(key=lambda x: int(x.version) if str(x.version).isdigit() else 0)

    logger.info("Protected versions to retain: %d", len(all_protected))
    logger.info("Candidate versions to prune:  %d", len(deletable_versions))

    return execute_pruning(
        client=client,
        model_name=model_name,
        candidates=deletable_versions,
        dry_run=not execute,
        request_delay=request_delay,
    )


def main(argv: list[str] | None = None) -> int:
    """CLI script entrypoint."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    args = parse_arguments(argv)
    cleanup_registry(
        model_name=args.model_name,
        tracking_uri=args.tracking_uri,
        keep_last=args.keep_last,
        keep_top_k=args.keep_top_k,
        metric_key=args.metric_key,
        request_delay=args.request_delay,
        execute=args.execute,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
