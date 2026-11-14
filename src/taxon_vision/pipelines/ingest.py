# SPDX-FileCopyrightText: 2026 Benjamin Förster
# SPDX-License-Identifier: MIT
"""Taxon metadata ingestion pipeline stage.

Fetches observation metadata for catalog taxa from the open iNaturalist API
and serializes them into an un-filtered raw Arrow Parquet manifest.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path
from typing import Any

import polars as pl

from taxon_vision.config import get_settings
from taxon_vision.data.inat_client import INatAPIClient
from taxon_vision.service.inference import load_taxa_catalog

logger = logging.getLogger(__name__)

DEFAULT_MANIFEST_DIR = Path("data/manifests")


async def ingest_taxon_metadata(
    taxon_id: int,
    inat_client: INatAPIClient,
    per_page: int,
) -> list[dict[str, Any]]:
    """Fetch raw observation metadata for a specific taxon.

    Args:
        taxon_id: The taxon ID to fetch metadata for.
        inat_client: The iNaturalist API client to use for fetching metadata.
        per_page: Number of items per page to fetch.

    Returns:
        List of dictionaries containing raw observation records.
    """
    logger.info("Fetching metadata for taxon %d...", taxon_id)
    observations = await inat_client.fetch_observations(taxon_id=taxon_id, per_page=per_page)

    manifest_records: list[dict[str, Any]] = []
    for obs in observations:
        obs_id = obs.get("observation_id") or obs.get("id") or ""
        manifest_records.append(
            {
                "observation_uuid": str(obs_id),
                "taxon_id": taxon_id,
                "image_url": str(obs.get("image_url", "")),
                "photographer_name": str(obs.get("attribution", "")),
                "attribution": str(obs.get("attribution", "")),
                "license_code": str(obs.get("license_code", "")),
                "license_url": str(obs.get("license_url", "")),
                "is_open_dataset_verified": bool(obs.get("is_open_dataset_verified", False)),
            }
        )
    return manifest_records


def _load_preserved_human_records(manifest_path: Path, fetched_uuids: set[str]) -> list[dict[str, Any]]:
    """Retain human-verified active learning records when regenerating `raw_manifest.parquet`.

    Why:
        Outcome B taxonomy expansions unfreeze DVC Stage 1 (`ingestion`) to fetch iNaturalist samples for newly added taxa. Rewriting `raw_manifest.parquet` purely from iNaturalist API results would discard previously merged Label Studio citizen-science observations and their S3 crop links.

    How:
        Inspects `manifest_path` if present on disk, identifies human-labelled rows (`label_studio_*` UUID prefix, `/repo-buckets/s3/`, or `s3://` URIs) absent from `fetched_uuids`, normalises schema attribution columns, and returns them for union preservation.

    Args:
        manifest_path: Existing `raw_manifest.parquet` path prior to API overwrite.
        fetched_uuids: Set of iNaturalist `observation_uuid` strings fetched in the current run.

    Returns:
        List of verified human-annotated record dictionaries to append alongside newly scraped API records.
    """
    if not manifest_path.is_file():
        return []
    existing_df = pl.read_parquet(manifest_path)
    preserved: list[dict[str, Any]] = []
    for row in existing_df.to_dicts():
        obs_uuid = str(row.get("observation_uuid", ""))
        image_url = str(row.get("image_url", ""))
        is_human = (
            obs_uuid.startswith("label_studio_") or "/repo-buckets/s3/" in image_url or image_url.startswith("s3://")
        )
        if is_human and obs_uuid not in fetched_uuids:
            photographer = str(row.get("photographer_name") or row.get("attribution") or "").strip()
            preserved.append(
                {
                    "observation_uuid": obs_uuid,
                    "taxon_id": int(row["taxon_id"]),
                    "image_url": image_url,
                    "photographer_name": photographer,
                    "attribution": str(row.get("attribution") or photographer),
                    "license_code": str(row.get("license_code", "")),
                    "license_url": str(row.get("license_url", "")),
                    "is_open_dataset_verified": bool(row.get("is_open_dataset_verified", True)),
                }
            )
    return preserved


async def run_ingestion(
    taxa: list[int] | None = None,
    per_page: int | None = None,
    out_path: Path | None = None,
) -> Path:
    """Execute asynchronous metadata ingestion and write to Parquet.

    Args:
        taxa: Optional list of taxon IDs. Defaults to configured taxa in settings.
        per_page: Optional observation count per taxon. Defaults to settings.
        out_path: Target Parquet output path.

    Returns:
        Path to the serialized Parquet manifest.
    """
    settings = get_settings()
    taxa_to_fetch = taxa if taxa is not None else settings.ingestion.taxa_to_fetch
    if not taxa_to_fetch:
        catalog = load_taxa_catalog()
        taxa_to_fetch = [int(t["inat_taxon_id"]) for t in catalog if "inat_taxon_id" in t]

    fetch_per_page = per_page if per_page is not None else settings.ingestion.per_page
    target_out = out_path if out_path is not None else DEFAULT_MANIFEST_DIR / "raw_manifest.parquet"

    inat = INatAPIClient()
    all_records: list[dict[str, Any]] = []
    for taxon_id in taxa_to_fetch:
        records = await ingest_taxon_metadata(taxon_id, inat, fetch_per_page)
        all_records.extend(records)
        logger.info("Ingested %d raw metadata records for taxon %d.", len(records), taxon_id)

    fetched_uuids = {str(rec.get("observation_uuid", "")) for rec in all_records}
    preserved_human = _load_preserved_human_records(target_out, fetched_uuids)
    if preserved_human:
        all_records.extend(preserved_human)
        logger.info("Preserved %d human-verified active learning records from %s", len(preserved_human), target_out)

    target_out.parent.mkdir(parents=True, exist_ok=True)
    df = pl.DataFrame(all_records)
    df.write_parquet(target_out)
    logger.info("Successfully wrote %d raw observation records to %s", len(df), target_out)
    return target_out


def ingest_cli(argv: list[str] | None = None) -> int:
    """CLI entrypoint for Stage 1: Ingestion.

    Args:
        argv: Optional command-line arguments.

    Returns:
        Integer exit code (0 on success).
    """
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Ingest observation metadata from iNaturalist into Parquet manifest.")
    parser.add_argument(
        "--taxa",
        type=int,
        nargs="+",
        default=None,
        help="Taxon IDs to fetch observations for (defaults to configured taxa in params.yaml)",
    )
    parser.add_argument(
        "--per-page",
        type=int,
        default=settings.ingestion.per_page,
        help=f"Observations per taxon to retrieve (default: {settings.ingestion.per_page})",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_MANIFEST_DIR / "raw_manifest.parquet",
        help="Target output Parquet file path",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s", force=True)
    asyncio.run(run_ingestion(taxa=args.taxa, per_page=args.per_page, out_path=args.out))
    return 0


if __name__ == "__main__":
    sys.exit(ingest_cli())
