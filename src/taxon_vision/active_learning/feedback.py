# SPDX-FileCopyrightText: 2026 Benjamin Förster
# SPDX-License-Identifier: MIT
"""Active learning feedback merger for Label Studio human annotation exports.

Summary:
    Parses completed expert taxonomic annotations from Label Studio JSON exports, enforces open-licensing and attribution invariants, and idempotently appends verified records into the canonical training manifest.

Why:
    Observations deferred to human experts during production inference (due to high conformal ambiguity or OOD energy) yield maximal information gain when reintroduced into model training. Merging must remain strictly idempotent and gatekeeper-compliant: any record with non-compliant licensing or missing photographer attribution must be discarded before touching the training dataset.

How:
    1. Delegates annotation parsing and CC0/CC-BY/CC-BY-NC validation to `feedback_parser`.
    2. Delegates Outcome B catalogue expansion registration/staging to `catalog_expansion`.
    3. Deduplicates against existing `observation_uuid` values and atomically rewrites the Parquet tables via sibling temp files.
"""

from __future__ import annotations

import argparse
import json
import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple

import polars as pl

from taxon_vision.active_learning.catalog_expansion import (
    FeedbackMergeOptions,
    _persist_catalog_expansions,
    _resolve_expansion_paths,
    apply_pending_catalog_expansions,
)
from taxon_vision.active_learning.feedback_parser import (
    PERMITTED_LICENSES,
    NewTaxonSpec,
    ParsedTaskAnnotation,
    TaxaMappings,
    _collect_feedback_items,
    load_taxa_mappings,
    parse_single_task_annotation,
)

__all__ = [
    "OOD_MANIFEST_COLUMNS",
    "PERMITTED_LICENSES",
    "RAW_MANIFEST_COLUMNS",
    "FeedbackMergeOptions",
    "FeedbackMergeResult",
    "NewTaxonSpec",
    "ParquetAppendStats",
    "ParsedTaskAnnotation",
    "TaxaMappings",
    "apply_pending_catalog_expansions",
    "load_taxa_mappings",
    "merge_feedback_to_manifest",
    "parse_single_task_annotation",
]

logger = logging.getLogger(__name__)

RAW_MANIFEST_COLUMNS: list[str] = [
    "observation_uuid",
    "taxon_id",
    "image_url",
    "photographer_name",
    "attribution",
    "license_code",
    "license_url",
    "is_open_dataset_verified",
]

OOD_MANIFEST_COLUMNS: list[str] = [
    "observation_uuid",
    "image_url",
    "ood_energy_score",
    "photographer_name",
    "license_code",
    "license_url",
    "verified_status",
]


class FeedbackMergeResult(NamedTuple):
    """Structured outcome of an active learning annotation ingestion cycle.

    Attributes:
        new_records_added: Number of newly verified records appended to the dataset manifest.
        ood_records_skipped: Number of Out-of-Distribution annotations safely excluded from closed-set training.
        total_manifest_records: Total count of verified observations in the updated Parquet manifest.
        status: High-level descriptive status ('merged', 'skipped_empty', or 'skipped_no_new_records').
        catalog_expanded: True when at least one novel target taxon was detected and queued or registered.
        new_taxon_ids: Tuple of newly registered iNaturalist taxon identifiers added in this cycle.
        ood_verified_added: Count of newly persisted confirmed-OOD specimens in ``ood_verified.parquet``.
    """

    new_records_added: int
    ood_records_skipped: int
    total_manifest_records: int
    status: str
    catalog_expanded: bool = False
    new_taxon_ids: tuple[int, ...] = ()
    ood_verified_added: int = 0


@dataclass(frozen=True)
class ParquetAppendStats:
    """Counts resulting from deduplicating and appending rows to a Parquet table."""

    added_count: int
    total_count: int


def _atomic_write_manifest(df: pl.DataFrame, manifest_path: Path) -> None:
    """Write a Polars DataFrame atomically to a target Parquet path via a temporary sibling file.

    Why:
        Partial Parquet writes break footer parsing across downstream DVC stages and dataset audits; atomic `replace()` prevents concurrent readers from seeing half-written files.

    How:
        Serialises `df` into a sibling temporary `.parquet` file via Polars and renames it atomically onto `manifest_path`, cleaning up on error.

    Args:
        df: Updated manifest Polars DataFrame.
        manifest_path: Destination Parquet path on disk.
    """
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=manifest_path.parent, delete=False, suffix=".parquet") as tmp:
        tmp_path = Path(tmp.name)
    try:
        df.write_parquet(tmp_path)
        tmp_path.replace(manifest_path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise


def _append_deduplicated_parquet(
    records: list[dict[str, Any]],
    target_path: Path,
    columns: list[str],
) -> ParquetAppendStats:
    """Deduplicate records by `observation_uuid` and atomically append new rows to Parquet.

    Why:
        Repeated exports from DagsHub Label Studio contain already-ingested tasks; O(N) set-based deduplication guarantees idempotency across scheduled/retried DAG runs for both `raw_manifest.parquet` and `ood_verified.parquet`.

    How:
        Reads `target_path` via Polars if present, builds a set of existing `observation_uuid` strings, filters `records`, concatenates new rows with relaxed diagonal schema alignment, and writes via `_atomic_write_manifest`.

    Args:
        records: Candidate record dicts to append.
        target_path: Destination Parquet file path.
        columns: Canonical column list preserved when initialising or aligning rows.

    Returns:
        `ParquetAppendStats` holding counts of newly added rows and total rows.
    """
    existing_df = pl.read_parquet(target_path) if target_path.is_file() else None
    existing_uuids = (
        set(existing_df["observation_uuid"].cast(pl.String).to_list())
        if existing_df is not None and "observation_uuid" in existing_df.columns
        else set()
    )
    new_records = [rec for rec in records if str(rec["observation_uuid"]) not in existing_uuids]
    if not new_records:
        return ParquetAppendStats(added_count=0, total_count=existing_df.height if existing_df is not None else 0)

    new_df = pl.DataFrame(new_records)
    updated_df = (
        pl.concat([existing_df, new_df], how="diagonal_relaxed")
        if existing_df is not None
        else new_df.select([col for col in columns if col in new_df.columns])
    )
    _atomic_write_manifest(updated_df, target_path)
    return ParquetAppendStats(added_count=len(new_records), total_count=updated_df.height)


def merge_feedback_to_manifest(
    export_path: Path | str,
    manifest_path: Path | str = "data/manifests/raw_manifest.parquet",
    catalog_path: Path | str = "config/taxa_catalog.yaml",
    options: FeedbackMergeOptions | None = None,
) -> FeedbackMergeResult:
    """Ingest Label Studio annotations and merge verified records into the canonical raw manifest.

    Why:
        Closes the active learning loop by persisting expert-verified observations into raw_manifest.parquet upstream of DVC Stage 2 (audit_license) so subsequent dvc repro runs retain human annotations alongside API-ingested observations, whilst supporting novel taxon catalogue expansion and isolated confirmed-OOD tracking.

    How:
        Parses Label Studio export JSON, routes Outcome C OOD annotations to ood_verified.parquet, resolves or registers Outcome B novel target taxa (staging to catalog_expansions.yaml when configs are mounted read-only), deduplicates against existing raw_manifest.parquet observation UUIDs, and atomically writes the merged Parquet table.

    Args:
        export_path: Path to Label Studio exported JSON file.
        manifest_path: Path to canonical raw ingestion manifest Parquet table.
        catalog_path: Path to taxa catalog YAML configuration.
        options: Optional FeedbackMergeOptions overriding expansion and OOD manifest paths.

    Returns:
        Structured FeedbackMergeResult with counts, catalogue expansion flags, and status summary.
    """
    exp_file = Path(export_path)
    if not exp_file.is_file():
        logger.info("Feedback export file not found at %s; skipping merge", exp_file)
        return FeedbackMergeResult(0, 0, 0, "skipped_empty")

    try:
        with exp_file.open("r", encoding="utf-8") as f:
            tasks = json.load(f)
    except (json.JSONDecodeError, OSError):
        logger.error("Failed to parse JSON export at %s", exp_file)
        return FeedbackMergeResult(0, 0, 0, "skipped_empty")

    task_list = [tasks] if isinstance(tasks, dict) else tasks
    cat_path = Path(catalog_path)
    man_path = Path(manifest_path)
    mappings = load_taxa_mappings(cat_path)
    collected = _collect_feedback_items(task_list, mappings)

    resolved_options = _resolve_expansion_paths(cat_path, man_path, options)
    expansion = _persist_catalog_expansions(collected.novel_specs, cat_path, resolved_options)

    ood_stats = ParquetAppendStats(added_count=0, total_count=0)
    if collected.ood_records and resolved_options.ood_manifest_path is not None:
        ood_stats = _append_deduplicated_parquet(
            collected.ood_records,
            resolved_options.ood_manifest_path,
            OOD_MANIFEST_COLUMNS,
        )

    manifest_stats = _append_deduplicated_parquet(
        collected.verified_records,
        man_path,
        RAW_MANIFEST_COLUMNS,
    )
    if manifest_stats.added_count == 0:
        return FeedbackMergeResult(
            new_records_added=0,
            ood_records_skipped=collected.ood_count,
            total_manifest_records=manifest_stats.total_count,
            status="skipped_no_new_records",
            catalog_expanded=expansion.catalog_expanded,
            new_taxon_ids=expansion.new_taxon_ids,
            ood_verified_added=ood_stats.added_count,
        )

    logger.info("Appended %d newly verified human annotations to %s", manifest_stats.added_count, man_path)
    return FeedbackMergeResult(
        new_records_added=manifest_stats.added_count,
        ood_records_skipped=collected.ood_count,
        total_manifest_records=manifest_stats.total_count,
        status="merged",
        catalog_expanded=expansion.catalog_expanded,
        new_taxon_ids=expansion.new_taxon_ids,
        ood_verified_added=ood_stats.added_count,
    )


def _main() -> None:
    """CLI entrypoint for merging Label Studio exports or applying pending catalogue expansions."""
    parser = argparse.ArgumentParser(description="Active learning Label Studio feedback merger and catalogue expander")
    parser.add_argument("--export-path", type=Path, default=None, help="Path to Label Studio export JSON file")
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=Path("data/manifests/raw_manifest.parquet"),
        help="Target raw manifest Parquet file",
    )
    parser.add_argument(
        "--catalog-path", type=Path, default=Path("config/taxa_catalog.yaml"), help="Path to taxa_catalog.yaml"
    )
    parser.add_argument(
        "--apply-expansions",
        type=Path,
        default=None,
        help="Commit pending catalog_expansions.yaml into repository manifests",
    )
    parser.add_argument("--params-path", type=Path, default=Path("params.yaml"), help="Path to params.yaml")
    parser.add_argument(
        "--label-studio-config-path",
        type=Path,
        default=Path("deploy/label_studio/project_config.xml"),
        help="Path to Label Studio XML",
    )
    args = parser.parse_args()

    if args.apply_expansions is not None:
        apply_pending_catalog_expansions(
            expansions_path=args.apply_expansions,
            catalog_path=args.catalog_path,
            params_path=args.params_path,
            label_studio_config_path=args.label_studio_config_path,
        )
        return

    if args.export_path is None:
        parser.error("Specify either --export-path or --apply-expansions")

    merge_feedback_to_manifest(
        export_path=args.export_path,
        manifest_path=args.manifest_path,
        catalog_path=args.catalog_path,
    )


if __name__ == "__main__":
    _main()
