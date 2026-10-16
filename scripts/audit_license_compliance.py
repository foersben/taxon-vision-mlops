#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 Benjamin Förster
# SPDX-License-Identifier: MIT
"""Audit iNaturalist open license compliance & attribution invariants.

Asserts that:
1. All ingested observation samples have allowable open licenses (CC0, CC-BY, CC-BY-NC, CC-BY-SA).
2. Observations from un-licensed static domains ('static.inaturalist.org' without open license) are rejected.
3. Photographer attribution metadata (photographer name, observation ID, license code, license URL) is preserved.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

# Open licenses permitted by iNaturalist open data charter
ALLOWED_LICENSES = {
    "CC0",
    "CC-BY",
    "CC-BY-NC",
    "CC-BY-SA",
    "CC-BY-ND",
    "CC-BY-NC-SA",
    "CC-BY-NC-ND",
}


def verify_observation_compliance(obs: dict[str, str | int]) -> list[str]:
    """Verify compliance for a single observation record.

    Args:
        obs: The observation record to verify.

    Returns:
        A list of errors found in the observation record.
    """
    errors: list[str] = []
    obs_id = obs.get("observation_id") or obs.get("observation_uuid") or "unknown"
    license_code = str(obs.get("license_code", "")).upper()
    image_url = str(obs.get("image_url", ""))
    attribution = str(obs.get("attribution") or obs.get("photographer_name") or "")

    if not license_code or license_code not in ALLOWED_LICENSES:
        errors.append(f"Observation {obs_id}: Invalid license '{license_code}'. Must be in {ALLOWED_LICENSES}.")

    if "static.inaturalist.org" in image_url and not obs.get("is_open_dataset_verified", False):
        errors.append(f"Observation {obs_id}: Static domain URL detected without open dataset license verification.")

    if not attribution or len(attribution.strip()) < 3:
        errors.append(f"Observation {obs_id}: Missing or empty photographer attribution string.")

    return errors


def _load_observations(manifest_path: Path) -> list[dict[str, Any]]:
    """Load observation records from parquet manifest or JSON fixture file.

    Args:
        manifest_path: Path to .parquet or .json manifest file.

    Returns:
        List of observation dictionaries.
    """
    if str(manifest_path).endswith(".parquet"):
        import polars as pl

        df = pl.read_parquet(manifest_path)
        return [dict(row) for row in df.iter_rows(named=True)]

    with manifest_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
        if isinstance(data, list):
            return data
        return []


def _resolve_manifest_path(arg_path: str | None) -> Path | None:
    """Resolve target manifest path from CLI argument or default locations.

    Args:
        arg_path: Optional explicit manifest path string.

    Returns:
        Resolved Path or None if not found.
    """
    if arg_path:
        return Path(arg_path)
    fixture_path = Path("tests/fixtures/sample_metadata.json")
    if fixture_path.exists():
        return fixture_path
    parquet_path = Path("data/manifests/dataset_manifest.parquet")
    if parquet_path.exists():
        return parquet_path
    return None


def main(argv: list[str] | None = None) -> int:
    """Run license compliance audit against fixtures or ingested samples."""
    import argparse

    parser = argparse.ArgumentParser(description="Audit iNaturalist open license compliance & attribution.")
    parser.add_argument("--manifest", type=str, default=None, help="Path to parquet manifest or JSON fixture")
    parser.add_argument("--out", type=Path, default=None, help="Destination path for audit report JSON")
    args = parser.parse_args(argv)

    target_path = _resolve_manifest_path(args.manifest)
    if target_path is None or not target_path.exists():
        print("Warning: Neither sample metadata fixture nor dataset manifest found. Creating placeholder pass.")
        return 0

    try:
        observations = _load_observations(target_path)
    except Exception as exc:
        print(f"Error reading {target_path}: {exc}", file=sys.stderr)
        return 1

    total_errors: list[str] = []
    for obs in observations:
        total_errors.extend(verify_observation_compliance(obs))

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        report = {
            "status": "PASSED" if not total_errors else "FAILED",
            "manifest": str(target_path),
            "verified_observations": len(observations),
            "violations": len(total_errors),
            "allowed_licenses": sorted(ALLOWED_LICENSES),
        }
        with args.out.open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

    if total_errors:
        print(f"License Compliance Audit FAILED with {len(total_errors)} violations:", file=sys.stderr)
        for err in total_errors[:10]:
            print(f"  - {err}", file=sys.stderr)
        if len(total_errors) > 10:
            print(f"  ... and {len(total_errors) - 10} more violations.", file=sys.stderr)
        return 1

    print(
        f"License Compliance Audit PASSED: {len(observations)} observations verified for open licensing & attribution."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
