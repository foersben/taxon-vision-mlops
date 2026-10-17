# SPDX-FileCopyrightText: 2026 Benjamin Förster
# SPDX-License-Identifier: MIT
"""Script to fetch metadata from iNaturalist, filter licenses, and create an Arrow Parquet manifest.

This script fetches metadata about species from iNaturalist and filters it based on license type.

Usage:
    pixi run --frozen -e dev python scripts/ingest_data.py
"""

import argparse
import asyncio
from pathlib import Path
from typing import Any

import polars as pl

from taxon_vision.config import get_settings
from taxon_vision.data.inat_client import INatAPIClient
from taxon_vision.service.inference import load_taxa_catalog

MANIFEST_DIR = Path("data/manifests")


async def ingest_taxon_metadata(taxon_id: int, inat_client: INatAPIClient, per_page: int) -> list[dict[str, Any]]:
    """Fetch raw observation metadata for a specific taxon.

    Args:
        taxon_id: The taxon ID to fetch metadata for.
        inat_client: The iNaturalist API client to use for fetching metadata.
        per_page: Number of items per page to fetch.

    Returns:
        A list of dictionaries containing the raw metadata for the specified taxon.
    """
    print(f"Fetching metadata for taxon {taxon_id}...")
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


async def main() -> None:
    """Main function to orchestrate metadata ingestion for specified taxa.

    This function will fetch metadata for the specified taxa and save it to a Parquet file.
    """
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Ingest observation metadata from iNaturalist into Parquet manifest.")
    parser.add_argument(
        "--taxa",
        type=int,
        nargs="+",
        default=None,
        help="Taxon IDs to fetch observations for (defaults to configured taxa in params.yaml / taxa_catalog.yaml)",
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
        default=MANIFEST_DIR / "raw_manifest.parquet",
        help="Target output Parquet file path",
    )
    args = parser.parse_args()

    taxa_to_fetch = args.taxa if args.taxa is not None else settings.ingestion.taxa_to_fetch
    if not taxa_to_fetch:
        catalog = load_taxa_catalog()
        taxa_to_fetch = [int(t["inat_taxon_id"]) for t in catalog if "inat_taxon_id" in t]
    per_page = args.per_page

    inat = INatAPIClient()
    all_records = []

    for taxon_id in taxa_to_fetch:
        records = await ingest_taxon_metadata(taxon_id, inat, per_page)
        all_records.extend(records)
        print(f"Ingested {len(records)} raw metadata records for taxon {taxon_id}.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out

    # Save directly to high-performance Parquet using Polars
    df = pl.DataFrame(all_records)
    df.write_parquet(manifest_path)
    print(f"Successfully wrote {len(all_records)} raw records to {manifest_path}")


if __name__ == "__main__":
    asyncio.run(main())
