# SPDX-FileCopyrightText: 2026 Benjamin Förster
# SPDX-License-Identifier: MIT
"""Script to fetch metadata from iNaturalist, filter licenses, and create an Arrow Parquet manifest.

This script fetches metadata about species from iNaturalist and filters it based on license type.

Usage:
    pixi run --frozen -e dev python scripts/ingest_data.py
"""

import asyncio
from pathlib import Path

import polars as pl
import yaml

from taxon_vision.data.inat_client import INatAPIClient
from taxon_vision.data.license_filter import LicenseFilter

MANIFEST_DIR = Path("data/manifests")


async def ingest_taxon_metadata(taxon_id: int, inat_client: INatAPIClient, per_page: int) -> list[dict[str, str | int]]:
    """Fetch and filter metadata for a specific taxon.

    Args:
        taxon_id: The taxon ID to fetch metadata for.
        inat_client: The iNaturalist API client to use for fetching metadata.
        per_page: Number of items per page to fetch.

    Returns:
        A list of dictionaries containing the metadata for the specified taxon.
    """
    print(f"Fetching metadata for taxon {taxon_id}...")
    observations = await inat_client.fetch_observations(taxon_id=taxon_id, per_page=per_page)

    manifest_records = []

    for obs in observations:
        record = LicenseFilter.process_record(obs)
        if not record:
            continue

        manifest_records.append(
            {
                "observation_uuid": record.observation_uuid,
                "taxon_id": taxon_id,
                "image_url": obs["image_url"],
                "photographer_name": record.photographer_name,
                "license_code": record.license_code.value,
                "license_url": record.license_url,
            }
        )

    return manifest_records


async def main() -> None:
    """Main function to orchestrate metadata ingestion for specified taxa.

    This function will fetch metadata for the specified taxa and save it to a Parquet file.
    """
    with open("config/params.yaml", encoding="utf-8") as f:
        params = yaml.safe_load(f)["ingestion"]

    taxa_to_fetch = params["taxa_to_fetch"]
    per_page = params["per_page"]

    inat = INatAPIClient()
    all_records = []

    for taxon_id in taxa_to_fetch:
        records = await ingest_taxon_metadata(taxon_id, inat, per_page)
        all_records.extend(records)
        print(f"Ingested {len(records)} compliant metadata records for taxon {taxon_id}.")

    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = MANIFEST_DIR / "dataset_manifest.parquet"

    # Save directly to high-performance Parquet using Polars
    df = pl.DataFrame(all_records)
    df.write_parquet(manifest_path)
    print(f"Successfully wrote {len(all_records)} records to {manifest_path}")


if __name__ == "__main__":
    asyncio.run(main())
