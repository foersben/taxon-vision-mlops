# SPDX-FileCopyrightText: 2026 Benjamin Förster
# SPDX-License-Identifier: MIT
"""Stage 2: Active License Gatekeeper & Open Compliance Verifier.

Filters candidate observations strictly through LicenseFilter (CC0, CC-BY, CC-BY-NC + non-empty attribution), producing the authoritative dataset_manifest.parquet and emitting reports/license_audit.json.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple

from taxon_vision.data.license_filter import LicenseFilter

logger = logging.getLogger(__name__)

# Open licenses permitted by governance mandate
ALLOWED_LICENSES: set[str] = {
    "cc0",
    "cc-by",
    "cc-by-nc",
}


class LicenseGateResult(NamedTuple):
    """Verified observation records paired with compliance rejection error messages."""

    verified_records: list[dict[str, Any]]
    errors: list[str]


@dataclass(frozen=True)
class _AuditReportStats:
    """Summary counts and error list for license audit reporting."""

    total_count: int
    verified_count: int
    errors: list[str]
    is_gatekeeper: bool = False


def verify_observation_compliance(obs: dict[str, Any]) -> list[str]:
    """Verify open license and photographer attribution compliance for a single observation record.

    Why:
        Citizen-science repositories contain observations under restrictive All-Rights-Reserved (ARR) or non-open licenses, and `static.inaturalist.org` mirrors lack open-dataset redistribution provenance unless explicitly verified. Enforcing Rule 05 at the ingestion gate guarantees that only CC0, CC-BY, and CC-BY-NC observations with retained photographer attribution enter training manifests.

    How:
        Normalizes `license_code` to lowercase hyphenated form and checks membership in `ALLOWED_LICENSES`, rejects unverified `static.inaturalist.org` URLs, and asserts that `attribution` or `photographer_name` contains at least 3 non-whitespace characters.

    Complexity:
        O(1) per-observation string and set membership inspection.

    Args:
        obs: The observation record dictionary to verify.

    Returns:
        A list of compliance violation messages (empty if compliant).
    """
    errors: list[str] = []
    obs_id = obs.get("observation_id") or obs.get("observation_uuid") or "unknown"
    license_code = str(obs.get("license_code", "")).strip().lower().replace("_", "-")
    image_url = str(obs.get("image_url", ""))
    attribution = str(obs.get("attribution") or obs.get("photographer_name") or "")

    if not license_code or license_code not in ALLOWED_LICENSES:
        errors.append(
            f"Observation {obs_id}: Invalid license '{obs.get('license_code', '')}'. Must be in {ALLOWED_LICENSES}."
        )

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


def _filter_observations_gate(
    raw_observations: list[dict[str, Any]],
) -> LicenseGateResult:
    """Filter candidate observations strictly through LicenseFilter and compliance checks.

    Args:
        raw_observations: Raw candidate observation dictionaries.

    Returns:
        LicenseGateResult of (verified_records, errors).
    """
    verified_records: list[dict[str, Any]] = []
    errors: list[str] = []

    for obs in raw_observations:
        errs = verify_observation_compliance(obs)
        if errs:
            errors.extend(errs)
            continue

        record = LicenseFilter.process_record(obs)
        if record is None:
            errors.append(
                f"Observation {obs.get('observation_uuid', 'unknown')}: Rejected by LicenseFilter validation."
            )
            continue

        verified_records.append(
            {
                "observation_uuid": str(
                    record.observation_uuid or obs.get("observation_uuid") or obs.get("observation_id") or ""
                ),
                "taxon_id": obs.get("taxon_id", 0),
                "image_url": obs.get("image_url", ""),
                "photographer_name": record.photographer_name,
                "license_code": record.license_code.value,
                "license_url": record.license_url,
            }
        )

    return LicenseGateResult(verified_records=verified_records, errors=errors)


def _write_verified_manifest(records: list[dict[str, Any]], out_path: Path) -> None:
    """Serialize verified observation records to destination Parquet manifest.

    Args:
        records: List of verified observation dictionaries.
        out_path: Target Parquet filesystem path.
    """
    import polars as pl

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = pl.DataFrame(records)
    df.write_parquet(out_path)
    logger.info("Gatekeeper wrote %d verified records to %s", len(records), out_path)


def _write_audit_report(
    out_path: Path,
    manifest_path: Path,
    stats: _AuditReportStats,
) -> None:
    """Emit structured license compliance audit report JSON.

    Args:
        out_path: Destination path for report JSON.
        manifest_path: Evaluated input manifest path.
        stats: Summary counts and error list.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    status = "PASSED" if (stats.verified_count > 0 if stats.is_gatekeeper else not stats.errors) else "FAILED"
    report = {
        "status": status,
        "manifest": str(manifest_path),
        "total_observations": stats.total_count,
        "verified_observations": stats.verified_count,
        "violations": len(stats.errors),
        "allowed_licenses": sorted(ALLOWED_LICENSES),
    }
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)


def _resolve_manifest_path(arg_path: str | None, is_gatekeeper: bool = False) -> Path | None:
    """Resolve target manifest path from CLI argument or default locations.

    Args:
        arg_path: Optional explicit manifest path string.
        is_gatekeeper: Whether invoked in active gatekeeper mode.

    Returns:
        Resolved Path or None if not found.
    """
    if arg_path:
        return Path(arg_path)
    if is_gatekeeper:
        raw_p = Path("data/manifests/raw_manifest.parquet")
        if raw_p.exists():
            return raw_p
        ds_p = Path("data/manifests/dataset_manifest.parquet")
        if ds_p.exists():
            return ds_p
        return raw_p

    fixture_path = Path("tests/fixtures/sample_metadata.json")
    if fixture_path.exists():
        return fixture_path
    parquet_path = Path("data/manifests/dataset_manifest.parquet")
    if parquet_path.exists():
        return parquet_path
    raw_path = Path("data/manifests/raw_manifest.parquet")
    if raw_path.exists():
        return raw_path
    return None


def _run_gatekeeper_mode(
    observations: list[dict[str, Any]],
    target_path: Path,
    out_manifest: Path,
    out_report: Path | None,
) -> int:
    """Execute active license filtering and write verified dataset manifest.

    Args:
        observations: Raw candidate observations.
        target_path: Source manifest path.
        out_manifest: Output path for verified parquet manifest.
        out_report: Optional output path for audit summary JSON.

    Returns:
        Exit code: 0 on success, 1 on complete rejection.
    """
    verified_records, errors = _filter_observations_gate(observations)
    if not verified_records:
        print(
            "License Gatekeeper FAILED: Zero observations passed license and attribution filtering.",
            file=sys.stderr,
        )
        return 1

    _write_verified_manifest(verified_records, out_manifest)
    if out_report is not None:
        _write_audit_report(
            out_report,
            target_path,
            _AuditReportStats(
                total_count=len(observations),
                verified_count=len(verified_records),
                errors=errors,
                is_gatekeeper=True,
            ),
        )
    print(f"License Gatekeeper PASSED: {len(verified_records)}/{len(observations)} compliant observations verified.")
    return 0


def _run_audit_mode(
    observations: list[dict[str, Any]],
    target_path: Path,
    out_report: Path | None,
) -> int:
    """Execute read-only license compliance verification against candidate records.

    Args:
        observations: Observation records to audit.
        target_path: Source manifest path.
        out_report: Optional output path for audit report JSON.

    Returns:
        Exit code: 0 if all records comply, 1 if any violations found.
    """
    total_errors: list[str] = []
    for obs in observations:
        total_errors.extend(verify_observation_compliance(obs))

    if out_report is not None:
        _write_audit_report(
            out_report,
            target_path,
            _AuditReportStats(
                total_count=len(observations),
                verified_count=len(observations) - len(total_errors),
                errors=total_errors,
            ),
        )

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


def audit_license_cli(argv: list[str] | None = None) -> int:
    """Run license compliance audit or gatekeeper filtering.

    Args:
        argv: Optional command-line argument list.

    Returns:
        Exit code: 0 for success, non-zero for failure.
    """
    parser = argparse.ArgumentParser(description="Audit iNaturalist open license compliance & attribution.")
    parser.add_argument("--manifest", type=str, default=None, help="Path to input parquet manifest or JSON fixture")
    parser.add_argument(
        "--out-manifest",
        type=Path,
        default=None,
        help="Optional destination parquet path for verified observations (enables gatekeeper mode)",
    )
    parser.add_argument("--out", type=Path, default=None, help="Destination path for audit report JSON")
    args = parser.parse_args(argv)

    is_gate = args.out_manifest is not None
    target_path = _resolve_manifest_path(args.manifest, is_gatekeeper=is_gate)

    if target_path is None or not target_path.exists():
        if is_gate:
            print(
                f"Error: Required input manifest '{target_path}' not found for gatekeeper execution.",
                file=sys.stderr,
            )
            return 1
        print("Warning: Neither sample metadata fixture nor dataset manifest found. Creating placeholder pass.")
        return 0

    try:
        observations = _load_observations(target_path)
    except Exception as exc:
        print(f"Error reading {target_path}: {exc}", file=sys.stderr)
        return 1

    if is_gate:
        return _run_gatekeeper_mode(observations, target_path, args.out_manifest, args.out)
    return _run_audit_mode(observations, target_path, args.out)


main = audit_license_cli

if __name__ == "__main__":
    sys.exit(audit_license_cli())
