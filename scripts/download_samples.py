#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 Benjamin Förster
# SPDX-License-Identifier: MIT
"""Download verified open-license sample images for taxa in config/taxa_catalog.yaml.

Usage:
    pixi run -e dev python scripts/download_samples.py
    pixi run -e dev python scripts/download_samples.py --per-taxon 2
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from pathlib import Path
from typing import Any

import httpx
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

SAMPLE_DIR = Path("data/sample")
METADATA_FILE = SAMPLE_DIR / "metadata.json"
TAXA_CATALOG_FILE = Path("config/taxa_catalog.yaml")


def _slugify(name: str) -> str:
    """Generate a filename-safe slug from a biological name."""
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def _load_catalog() -> list[dict[str, Any]]:
    """Load taxa definitions from the authoritative catalog."""
    if not TAXA_CATALOG_FILE.exists():
        logger.error("Taxa catalog not found at %s", TAXA_CATALOG_FILE)
        sys.exit(1)
    with TAXA_CATALOG_FILE.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data.get("taxa", [])


def _get_catalog_inat_ids() -> dict[str, int]:
    """Dynamically derive scientific_name to inat_taxon_id mapping from config/taxa_catalog.yaml."""
    taxa = _load_catalog()
    return {str(t["scientific_name"]): int(t["inat_taxon_id"]) for t in taxa if "inat_taxon_id" in t}


# Dynamically loaded authoritative iNaturalist IDs from config/taxa_catalog.yaml (SSOT)
INAT_TAXON_IDS: dict[str, int] = _get_catalog_inat_ids()


def _fetch_observations_for_taxon(
    client: httpx.Client,
    inat_id: int,
    limit: int,
) -> list[dict[str, Any]]:
    """Query iNaturalist API for research-grade, open-license observations."""
    url = "https://api.inaturalist.org/v1/observations"
    params: dict[str, str | int] = {
        "taxon_id": inat_id,
        "quality_grade": "research",
        "has[]": "photos",
        "license": "cc0,cc-by,cc-by-nc",
        "per_page": limit,
    }
    response = client.get(url, params=params, timeout=15.0)
    response.raise_for_status()
    data: dict[str, Any] = response.json()
    results: list[dict[str, Any]] = data.get("results", [])
    return results


def _download_and_save_photo(
    client: httpx.Client,
    photo_url: str,
    target_path: Path,
) -> bool:
    """Download binary image content and save to destination path."""
    try:
        resp = client.get(photo_url, timeout=20.0)
        resp.raise_for_status()
        target_path.write_bytes(resp.content)
        return True
    except Exception as err:
        logger.warning("Failed to download image from %s: %s", photo_url, err)
        return False


def _process_taxon_sample(
    client: httpx.Client,
    taxon: dict[str, Any],
    per_taxon: int,
) -> list[dict[str, Any]]:
    """Fetch and download sample images for a single catalog taxon."""
    sci_name = taxon["scientific_name"]
    common_name = taxon.get("common_name", sci_name)
    inat_id = INAT_TAXON_IDS.get(sci_name)
    if not inat_id:
        logger.warning("No iNaturalist ID mapped for '%s', skipping.", sci_name)
        return []

    logger.info("Fetching %d sample(s) for '%s' (%s, iNat ID: %d)...", per_taxon, sci_name, common_name, inat_id)
    try:
        observations = _fetch_observations_for_taxon(client, inat_id, limit=per_taxon)
    except Exception as err:
        logger.warning("API query failed for taxon %d: %s", inat_id, err)
        return []

    collected: list[dict[str, Any]] = []
    slug = _slugify(common_name)

    for obs in observations:
        photos = obs.get("photos", [])
        if not photos:
            continue
        photo = photos[0]
        obs_id = str(obs.get("id"))
        photo_url = photo.get("url", "").replace("square", "medium")
        if not photo_url:
            continue

        filename = f"{obs_id}_taxon_{inat_id}_{slug}.jpg"
        target_path = SAMPLE_DIR / filename

        if _download_and_save_photo(client, photo_url, target_path):
            logger.info("  ✓ Saved %s (%d bytes)", filename, target_path.stat().st_size)
            collected.append(
                {
                    "observation_id": obs_id,
                    "scientific_name": sci_name,
                    "common_name": common_name,
                    "catalog_id": taxon["id"],
                    "inat_taxon_id": inat_id,
                    "filename": filename,
                    "license_code": (photo.get("license_code") or "CC-BY-NC").upper(),
                    "attribution": photo.get("attribution") or f"(c) iNaturalist Observer {obs_id}",
                    "image_url": photo_url,
                    "is_open_dataset_verified": True,
                }
            )

    return collected


def main() -> None:
    """Orchestrate downloading samples for all catalog taxa."""
    parser = argparse.ArgumentParser(description="Download sample images for trained catalog taxa.")
    parser.add_argument("--per-taxon", type=int, default=1, help="Number of sample images per taxon (default: 1).")
    args = parser.parse_args()

    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    catalog = _load_catalog()

    existing_metadata: list[dict[str, Any]] = []
    if METADATA_FILE.exists():
        try:
            existing_metadata = json.loads(METADATA_FILE.read_text(encoding="utf-8"))
        except Exception:
            existing_metadata = []

    all_metadata: dict[str, dict[str, Any]] = {item["observation_id"]: item for item in existing_metadata}

    with httpx.Client(headers={"User-Agent": "TaxonVision-MLOps/0.1 (research@foersben.local)"}) as client:
        for idx, taxon in enumerate(catalog, start=1):
            records = _process_taxon_sample(client, taxon, per_taxon=args.per_taxon)
            for r in records:
                all_metadata[r["observation_id"]] = r
            if idx < len(catalog):
                time.sleep(1.0)  # Respect iNaturalist 1 req/sec rate limit

    updated_records = list(all_metadata.values())
    METADATA_FILE.write_text(json.dumps(updated_records, indent=2), encoding="utf-8")
    logger.info("Successfully synced %d total sample metadata records to %s", len(updated_records), METADATA_FILE)


if __name__ == "__main__":
    main()
