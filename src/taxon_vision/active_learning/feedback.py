# SPDX-FileCopyrightText: 2026 Benjamin Förster
# SPDX-License-Identifier: MIT
"""Active learning feedback merger for Label Studio human annotation exports.

Summary:
    Parses completed expert taxonomic annotations from Label Studio JSON exports, enforces open-licensing and attribution invariants, and idempotently appends verified records into the canonical training manifest.

Why:
    Observations deferred to human experts during production inference (due to high conformal ambiguity or OOD energy) yield maximal information gain when reintroduced into model training. Merging must remain strictly idempotent and gatekeeper-compliant: any record with non-compliant licensing or missing photographer attribution must be discarded before touching the training dataset.

How:
    1. Loads canonical taxa catalog from YAML to build bidirectional mapping between scientific names, aliases, and iNaturalist taxon IDs.
    2. Extracts annotations from Label Studio task payloads, separating closed-set taxa from Out-of-Distribution classifications.
    3. Validates that license codes belong to the permitted open whitelist (CC0, CC-BY, CC-BY-NC) and photographer strings are non-empty.
    4. Deduplicates against existing dataset_manifest.parquet observation UUIDs before appending and atomically rewriting the Parquet table.
"""

from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path
from typing import Any, NamedTuple

import pandas as pd
import yaml

logger = logging.getLogger(__name__)

PERMITTED_LICENSES: frozenset[str] = frozenset({"cc0", "cc-by", "cc-by-nc"})


class TaxaMappings(NamedTuple):
    """Bidirectional taxonomic lookup structures derived from the canonical catalog.

    Attributes:
        name_to_id: Mapping from normalized lowercase scientific or common names to iNaturalist taxon IDs.
        valid_ids: Set of valid integer iNaturalist taxon IDs in the target catalog.
    """

    name_to_id: dict[str, int]
    valid_ids: set[int]


class ParsedTaskAnnotation(NamedTuple):
    """Validated outcome of parsing a single Label Studio annotation task.

    Attributes:
        record: Verified manifest row dictionary if valid and in-distribution, otherwise None.
        is_ood: Boolean indicator whether the specimen was classified as Out-of-Distribution.
    """

    record: dict[str, Any] | None
    is_ood: bool


class FeedbackMergeResult(NamedTuple):
    """Structured outcome of an active learning annotation ingestion cycle.

    Attributes:
        new_records_added: Number of newly verified records appended to the dataset manifest.
        ood_records_skipped: Number of Out-of-Distribution annotations safely excluded from closed-set training.
        total_manifest_records: Total count of verified observations in the updated Parquet manifest.
        status: High-level descriptive status ('merged', 'skipped_empty', or 'skipped_no_new_records').
    """

    new_records_added: int
    ood_records_skipped: int
    total_manifest_records: int
    status: str


def load_taxa_mappings(
    catalog_path: Path | str = "config/taxa_catalog.yaml",
) -> TaxaMappings:
    """Build taxonomic lookup dictionaries from the canonical catalog.

    Why:
        Label Studio annotations may contain scientific names, vernacular common names, or raw integer taxon IDs. A pre-indexed lookup guarantees deterministic O(1) resolution.

    How:
        Parses YAML catalog into a normalized lowercase string-to-ID dictionary and an integer ID validation set encapsulated in a typed TaxaMappings container.

    Args:
        catalog_path: Path to the taxa catalog YAML configuration.

    Returns:
        TaxaMappings containing name_to_id dictionary and valid_ids set.

    Raises:
        FileNotFoundError: If the taxa catalog file does not exist at catalog_path.
    """
    path = Path(catalog_path)
    if not path.is_file():
        raise FileNotFoundError(f"Taxa catalog configuration not found at {path}")

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    name_to_id: dict[str, int] = {}
    valid_ids: set[int] = set()

    for entry in data.get("taxa", []):
        inat_id = int(entry["inat_taxon_id"])
        valid_ids.add(inat_id)
        name_to_id[entry["scientific_name"].strip().lower()] = inat_id
        if "common_name" in entry:
            name_to_id[entry["common_name"].strip().lower()] = inat_id

    return TaxaMappings(name_to_id=name_to_id, valid_ids=valid_ids)


def _extract_task_choice(task: dict[str, Any]) -> str:
    """Extract selected taxonomy choice string from task annotations.

    Args:
        task: Individual task dictionary from Label Studio JSON export.

    Returns:
        Selected choice string, or empty string if unselected.
    """
    annotations = task.get("annotations", [])
    if not annotations and "result" in task:
        annotations = [task]

    for ann in annotations:
        for result in ann.get("result", []):
            if result.get("from_name") == "taxon_choice":
                choices = result.get("value", {}).get("choices", [])
                if choices:
                    return str(choices[0]).strip()
    return ""


def _resolve_taxon_id(
    choice: str,
    mappings: TaxaMappings,
) -> int | None:
    """Resolve raw choice string to verified integer taxon ID.

    Args:
        choice: Selected choice string.
        mappings: TaxaMappings containing name_to_id and valid_ids.

    Returns:
        Integer taxon ID or None if unresolved.
    """
    if choice.isdigit() and int(choice) in mappings.valid_ids:
        return int(choice)

    choice_lower = choice.lower()
    for name, tid in mappings.name_to_id.items():
        if name in choice_lower:
            return tid

    return None


def _validate_licensing_and_attribution(
    obs_uuid: str,
    image_url: str,
    photographer: str,
    license_code: str,
) -> bool:
    """Verify that observation complies with open licensing and attribution mandates.

    Args:
        obs_uuid: Unique observation identifier.
        image_url: Hosted crop image URL.
        photographer: Photographer credit string.
        license_code: Normalized lowercase Creative Commons license code.

    Returns:
        True if observation complies with open licensing requirements, otherwise False.
    """
    if license_code not in PERMITTED_LICENSES:
        logger.warning("Observation %s rejected: Non-compliant license '%s'", obs_uuid, license_code)
        return False

    if not photographer:
        logger.warning("Observation %s rejected: Missing mandatory photographer attribution", obs_uuid)
        return False

    if not obs_uuid or not image_url:
        logger.warning("Observation %s rejected: Incomplete UUID or image URL", obs_uuid)
        return False

    return True


def parse_single_task_annotation(
    task: dict[str, Any],
    mappings: TaxaMappings,
) -> ParsedTaskAnnotation:
    """Extract and validate a single Label Studio task annotation.

    Why:
        Protects the training dataset from copyright infringement and invalid annotations by enforcing license whitelists and mandatory attribution.

    How:
        Extracts metadata and choices, identifies Out-of-Distribution tags, resolves the taxon ID, and verifies non-empty attribution before emitting the record.

    Args:
        task: Individual task dictionary from Label Studio JSON export.
        mappings: TaxaMappings containing name_to_id and valid_ids.

    Returns:
        ParsedTaskAnnotation containing the verified record dictionary (or None) and OOD indicator flag.
    """
    data = task.get("data", {})
    obs_uuid = str(data.get("observation_uuid") or task.get("id") or "").strip()
    image_url = str(data.get("image_url") or "").strip()
    photographer = str(data.get("photographer_name") or "").strip()
    license_code = str(data.get("license_code") or "").strip().lower()
    license_url = str(data.get("license_url") or "").strip()

    selected_choice = _extract_task_choice(task)
    if not selected_choice:
        return ParsedTaskAnnotation(record=None, is_ood=False)

    choice_lower = selected_choice.lower()
    if "out of distribution" in choice_lower or choice_lower == "ood":
        logger.info("Observation %s classified as OOD; routing away from species training head", obs_uuid)
        return ParsedTaskAnnotation(record=None, is_ood=True)

    taxon_id = _resolve_taxon_id(selected_choice, mappings)
    if taxon_id is None:
        logger.warning("Observation %s: Could not resolve taxonomy choice '%s'", obs_uuid, selected_choice)
        return ParsedTaskAnnotation(record=None, is_ood=False)

    if not _validate_licensing_and_attribution(obs_uuid, image_url, photographer, license_code):
        return ParsedTaskAnnotation(record=None, is_ood=False)

    record = {
        "observation_uuid": obs_uuid,
        "taxon_id": taxon_id,
        "image_url": image_url,
        "photographer_name": photographer,
        "license_code": license_code.upper(),
        "license_url": license_url,
    }
    return ParsedTaskAnnotation(record=record, is_ood=False)


def _atomic_write_manifest(df: pd.DataFrame, manifest_path: Path) -> None:
    """Write DataFrame atomically to target Parquet file path via temporary file.

    Args:
        df: Updated manifest DataFrame.
        manifest_path: Destination path on disk.
    """
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=manifest_path.parent, delete=False, suffix=".parquet") as tmp:
        tmp_path = Path(tmp.name)
    try:
        df.to_parquet(tmp_path, index=False)
        tmp_path.replace(manifest_path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise


def merge_feedback_to_manifest(
    export_path: Path | str,
    manifest_path: Path | str = "data/manifests/dataset_manifest.parquet",
    catalog_path: Path | str = "config/taxa_catalog.yaml",
) -> FeedbackMergeResult:
    """Ingest Label Studio annotations and merge verified records into the dataset manifest.

    Why:
        Closes the active learning loop by safely persisting expert-verified observations into the training manifest with guaranteed schema compatibility and deduplication.

    How:
        Parses export file, extracts valid records, loads existing Parquet manifest, filters out existing observation UUIDs, and atomically writes the merged table.

    Args:
        export_path: Path to Label Studio exported JSON file.
        manifest_path: Path to canonical dataset manifest Parquet table.
        catalog_path: Path to taxa catalog YAML configuration.

    Returns:
        Structured FeedbackMergeResult with counts and status summary.
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
    mappings = load_taxa_mappings(catalog_path)
    verified_records: list[dict[str, Any]] = []
    ood_count = 0

    for task in task_list:
        parsed = parse_single_task_annotation(task, mappings)
        if parsed.is_ood:
            ood_count += 1
        elif parsed.record is not None:
            verified_records.append(parsed.record)

    man_path = Path(manifest_path)
    existing_df = (
        pd.read_parquet(man_path)
        if man_path.is_file()
        else pd.DataFrame(
            columns=["observation_uuid", "taxon_id", "image_url", "photographer_name", "license_code", "license_url"]
        )
    )
    existing_uuids = set(existing_df["observation_uuid"].astype(str))
    new_records = [r for r in verified_records if str(r["observation_uuid"]) not in existing_uuids]

    if not new_records:
        return FeedbackMergeResult(0, ood_count, len(existing_df), "skipped_no_new_records")

    updated_df = pd.concat([existing_df, pd.DataFrame(new_records)], ignore_index=True)
    _atomic_write_manifest(updated_df, man_path)

    logger.info("Appended %d newly verified human annotations to %s", len(new_records), man_path)
    return FeedbackMergeResult(
        new_records_added=len(new_records),
        ood_records_skipped=ood_count,
        total_manifest_records=len(updated_df),
        status="merged",
    )
