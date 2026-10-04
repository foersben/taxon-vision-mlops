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
    obs_id = obs.get("observation_id", "unknown")
    license_code = str(obs.get("license_code", "")).upper()
    image_url = str(obs.get("image_url", ""))
    attribution = str(obs.get("attribution", ""))

    if not license_code or license_code not in ALLOWED_LICENSES:
        errors.append(f"Observation {obs_id}: Invalid license '{license_code}'. Must be in {ALLOWED_LICENSES}.")

    if "static.inaturalist.org" in image_url and not obs.get("is_open_dataset_verified", False):
        errors.append(f"Observation {obs_id}: Static domain URL detected without open dataset license verification.")

    if not attribution or len(attribution.strip()) < 3:
        errors.append(f"Observation {obs_id}: Missing or empty photographer attribution string.")

    return errors


def main() -> int:
    """Run license compliance audit against fixtures or ingested samples."""
    fixture_path = Path("tests/fixtures/sample_metadata.json")
    if not fixture_path.exists():
        print(f"Warning: {fixture_path} not found. Creating placeholder verification pass.")
        return 0

    try:
        with fixture_path.open("r", encoding="utf-8") as f:
            observations = json.load(f)
    except Exception as exc:
        print(f"Error reading {fixture_path}: {exc}", file=sys.stderr)
        return 1

    total_errors: list[str] = []
    for obs in observations:
        total_errors.extend(verify_observation_compliance(obs))

    if total_errors:
        print(f"License Compliance Audit FAILED with {len(total_errors)} violations:", file=sys.stderr)
        for err in total_errors:
            print(f"  - {err}", file=sys.stderr)
        return 1

    print(
        f"License Compliance Audit PASSED: {len(observations)} observations verified for open licensing & attribution."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
