# SPDX-FileCopyrightText: 2026 Benjamin Förster
# SPDX-License-Identifier: MIT
"""Label Studio annotation parsing, open-license verification, and Outcome A/B/C routing.

Summary:
    Parses completed expert annotations exported from Label Studio JSON tasks, validates open Creative Commons licenses and photographer attribution (`PERMITTED_LICENSES`), resolves closed-set or novel taxon IDs, and buckets parsed records into `CollectedFeedback`.

Why (Architectural Rationale & Trade-offs):
    Separating stateless JSON annotation parsing and license/attribution gating from Parquet manifest I/O and Outcome B YAML/XML catalogue persistence keeps every sibling module in `active_learning/` below 500 LOC (Rule 08) and allows isolated testing of parsing edge cases without filesystem mutations.

How (Algorithmic Mechanism & Invariants):
    Loads normalised lowercased lookup tables (`TaxaMappings`) from `taxa_catalog.yaml`, extracts `Choices` and `TextArea` control responses, routes `"out of distribution"` / `"ood"` choices into OOD audit rows, validates positive numeric `NewTaxonSpec` metadata for `"novel_taxon"`, enforces `PERMITTED_LICENSES` and non-empty attribution, and aggregates results in `CollectedFeedback`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple

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


@dataclass(frozen=True)
class NewTaxonSpec:
    """Validated metadata required to expand the canonical taxonomic catalogue with a new species.

    Attributes:
        inat_taxon_id: Verified integer iNaturalist taxon identifier.
        scientific_name: Binomial scientific name for the taxon.
        common_name: English vernacular common name.
        kingdom: Biological kingdom (for example, Animalia, Plantae, or Fungi).
        taxon_class: Taxonomic class (for example, Insecta or Aves).
    """

    inat_taxon_id: int
    scientific_name: str
    common_name: str
    kingdom: str = "Unknown"
    taxon_class: str = "Unknown"


class ParsedTaskAnnotation(NamedTuple):
    """Validated outcome of parsing a single Label Studio annotation task.

    Attributes:
        record: Verified manifest row dictionary if valid and in-distribution, otherwise None.
        is_ood: Boolean indicator whether the specimen was classified as Out-of-Distribution.
        ood_record: Structured OOD audit record when classified as Out-of-Distribution, otherwise None.
        new_taxon: Validated novel taxon registration specification when Outcome B expansion is selected.
    """

    record: dict[str, Any] | None
    is_ood: bool
    ood_record: dict[str, Any] | None = None
    new_taxon: NewTaxonSpec | None = None


@dataclass(frozen=True)
class CollectedFeedback:
    """Intermediate aggregation of parsed feedback records grouped by ingestion outcome."""

    verified_records: list[dict[str, Any]]
    ood_records: list[dict[str, Any]]
    novel_specs: list[NewTaxonSpec]
    ood_count: int


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


def _iter_annotation_results(task: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten Label Studio annotation result dictionaries across top-level annotations.

    Why:
        Label Studio webhooks and JSON exports nest annotator widget outputs (`Choices`, `TextArea`) inside an `annotations[*].result` hierarchy or directly on `task['result']` depending on the API version; normalising this once eliminates schema branching in downstream field extractors.

    How:
        Resolves `task['annotations']` (or wraps `task` when only a top-level `'result'` key exists) and collects every dictionary item from each `'result'` list in order.

    Args:
        task: Raw Label Studio task dictionary from an annotation export.

    Returns:
        Flat list of annotation widget result dictionaries.
    """
    annotations = task.get("annotations", [])
    if not annotations and "result" in task:
        annotations = [task]
    results: list[dict[str, Any]] = []
    for ann in annotations:
        for item in ann.get("result", []):
            if isinstance(item, dict):
                results.append(item)
    return results


def _extract_task_choice(task: dict[str, Any]) -> str:
    """Extract the selected taxonomy choice string from task annotations.

    Why:
        Isolates Label Studio widget extraction from taxon ID resolution so schema changes affect only a single helper.

    How:
        Scans flattened results via `_iter_annotation_results(task)` for `from_name == "taxon_choice"` and returns the stripped first element of `value['choices']`.

    Args:
        task: Individual task dictionary from Label Studio JSON export.

    Returns:
        Selected choice string, or empty string if unselected.
    """
    for result in _iter_annotation_results(task):
        if result.get("from_name") == "taxon_choice":
            choices = result.get("value", {}).get("choices", [])
            if choices:
                return str(choices[0]).strip()
    return ""


def _extract_textarea_values(task: dict[str, Any]) -> dict[str, str]:
    """Extract trimmed TextArea annotation values keyed by Label Studio control name.

    Why:
        When an expert taxonomist selects `"New Species"` (Outcome B), structured novel-taxon metadata (`novel_inat_taxon_id`, `novel_scientific_name`, `novel_common_name`, `novel_kingdom`, `novel_class`) arrives inside Label Studio `TextArea` entries whose `value['text']` can be either a list of strings or a plain string.

    How:
        Iterates over `_iter_annotation_results(task)`, validates non-empty `from_name`, extracts the first string element or raw string value, strips whitespace, and indexes by `from_name`.

    Args:
        task: Individual task dictionary from Label Studio JSON export.

    Returns:
        Dictionary mapping Label Studio control names to trimmed text values.
    """
    extracted: dict[str, str] = {}
    for result in _iter_annotation_results(task):
        from_name = str(result.get("from_name") or "").strip()
        if not from_name:
            continue
        text_val = result.get("value", {}).get("text")
        if isinstance(text_val, list) and text_val:
            extracted[from_name] = str(text_val[0]).strip()
        elif isinstance(text_val, str) and text_val.strip():
            extracted[from_name] = text_val.strip()
    return extracted


def _parse_novel_taxon_spec(task: dict[str, Any]) -> NewTaxonSpec | None:
    """Extract validated novel taxon registration metadata for Outcome B catalogue expansion.

    Why:
        Admitting an unseen species into `config/taxa_catalog.yaml` without a positive integer iNaturalist ID and binomial scientific name would corrupt downstream Stage 1 iNaturalist ingestion and Stage 4 classifier class indices.

    How:
        Reads `TextArea` responses (falling back to `task['data']`), verifies positive numeric `novel_inat_taxon_id` and non-empty `novel_scientific_name`, fills optional `common_name`, `kingdom`, and `taxon_class` defaults, and returns a frozen `NewTaxonSpec`.

    Args:
        task: Individual task dictionary from Label Studio JSON export.

    Returns:
        Validated `NewTaxonSpec` instance if Outcome B metadata is well-formed, otherwise `None`.
    """
    text_fields = _extract_textarea_values(task)
    data = task.get("data", {})
    raw_id = text_fields.get("novel_inat_taxon_id") or str(data.get("novel_inat_taxon_id") or "").strip()
    raw_sci = text_fields.get("novel_scientific_name") or str(data.get("novel_scientific_name") or "").strip()
    if not raw_id.isdigit() or not raw_sci:
        return None
    taxon_id = int(raw_id)
    if taxon_id <= 0:
        return None
    common = text_fields.get("novel_common_name") or str(data.get("novel_common_name") or "").strip() or raw_sci
    kingdom = text_fields.get("novel_kingdom") or str(data.get("novel_kingdom") or "").strip() or "Unknown"
    taxon_class = text_fields.get("novel_class") or str(data.get("novel_class") or "").strip() or "Unknown"
    return NewTaxonSpec(
        inat_taxon_id=taxon_id,
        scientific_name=raw_sci,
        common_name=common,
        kingdom=kingdom,
        taxon_class=taxon_class,
    )


def _resolve_taxon_id(
    choice: str,
    mappings: TaxaMappings,
) -> int | None:
    """Resolve raw choice string to a verified integer taxon ID.

    Why:
        Label Studio choices can submit numeric `alias` values (iNaturalist taxon IDs) or human-readable display labels (`"Apis mellifera (Western Honey Bee)"`); supporting both forms decouples XML UI formatting from numeric manifest schemas.

    How:
        Checks exact membership in `mappings.valid_ids` when `choice` is digits, then scans lowercase `mappings.name_to_id` keys for substring matches.

    Args:
        choice: Selected choice string.
        mappings: `TaxaMappings` containing `name_to_id` and `valid_ids`.

    Returns:
        Integer taxon ID or `None` if unresolved.
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

    Why:
        Enforces Rule 05 Open License Compliance prior to training manifest insertion: copyrighted uploads, stripped credits, or broken S3/HTTP URIs must be blocked before Stage 3 feature extraction downloads assets.

    How:
        Asserts `license_code in PERMITTED_LICENSES`, `len(photographer.strip()) >= 3`, and non-empty `obs_uuid` and `image_url`, logging structured rejection warnings on any violation.

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

    if len(photographer.strip()) < 3:
        logger.warning("Observation %s rejected: Missing mandatory photographer attribution", obs_uuid)
        return False

    if not obs_uuid or not image_url:
        logger.warning("Observation %s rejected: Incomplete UUID or image URL", obs_uuid)
        return False

    return True


def _build_verified_record(
    obs_uuid: str,
    taxon_id: int,
    data: dict[str, Any],
    license_code: str,
) -> dict[str, Any]:
    """Assemble a canonical manifest record retaining open-license and attribution fields.

    Why:
        Downstream Stage 2 (`audit_license.py`) and Stage 3 (`s3_streamer.py`) expect identical schema columns (`observation_uuid`, `taxon_id`, `image_url`, `photographer_name`, `attribution`, `license_code`, `license_url`, `is_open_dataset_verified`) across both iNaturalist API harvests and human-annotated captures.

    How:
        Extracts stripped attribution strings from `data`, uppercases `license_code`, marks `is_open_dataset_verified=True`, and returns a dict matching `RAW_MANIFEST_COLUMNS`.

    Args:
        obs_uuid: Unique observation identifier.
        taxon_id: Verified integer iNaturalist taxon ID.
        data: Task `'data'` payload containing source image URL and license metadata.
        license_code: Validated lowercase Creative Commons code.

    Returns:
        Dictionary formatted to match the canonical Parquet manifest schema.
    """
    photographer = str(data.get("photographer_name") or "").strip()
    image_url = str(data.get("image_url") or "").strip()
    license_url = str(data.get("license_url") or "").strip()
    return {
        "observation_uuid": obs_uuid,
        "taxon_id": taxon_id,
        "image_url": image_url,
        "photographer_name": photographer,
        "attribution": photographer,
        "license_code": license_code.upper(),
        "license_url": license_url,
        "is_open_dataset_verified": True,
    }


def parse_single_task_annotation(
    task: dict[str, Any],
    mappings: TaxaMappings,
) -> ParsedTaskAnnotation:
    """Extract and validate a single Label Studio task annotation.

    Why:
        Protects the training dataset from copyright infringement and invalid annotations by enforcing license whitelists and mandatory attribution across closed-set refinement, novel catalogue expansion, and confirmed OOD routing.

    How:
        Extracts metadata and choices, routes Outcome C ('ood') into an isolated OOD record, parses Outcome B ('novel_taxon') metadata and updates mappings when valid, or resolves Outcome A closed-set taxon IDs before enforcing license and attribution compliance.

    Args:
        task: Individual task dictionary from Label Studio JSON export.
        mappings: TaxaMappings containing name_to_id and valid_ids.

    Returns:
        ParsedTaskAnnotation containing verified record (or None), OOD indicator/record, and optional NewTaxonSpec.
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
        ood_record = (
            {
                "observation_uuid": obs_uuid,
                "image_url": image_url,
                "ood_energy_score": float(data.get("ood_energy_score", 0.0)),
                "photographer_name": photographer or "Anonymous",
                "license_code": license_code.upper(),
                "license_url": license_url,
                "verified_status": "ood",
            }
            if obs_uuid
            else None
        )
        return ParsedTaskAnnotation(record=None, is_ood=True, ood_record=ood_record)

    novel_spec: NewTaxonSpec | None = None
    if choice_lower == "novel_taxon" or "novel target taxon" in choice_lower:
        novel_spec = _parse_novel_taxon_spec(task)
        if novel_spec is None:
            logger.warning(
                "Observation %s selected novel_taxon without valid integer taxon ID and scientific name", obs_uuid
            )
            return ParsedTaskAnnotation(record=None, is_ood=False)
        resolved_novel_id = novel_spec.inat_taxon_id
        taxon_id = resolved_novel_id
        mappings.valid_ids.add(resolved_novel_id)
        mappings.name_to_id[novel_spec.scientific_name.strip().lower()] = resolved_novel_id
        mappings.name_to_id[novel_spec.common_name.strip().lower()] = resolved_novel_id
    else:
        resolved = _resolve_taxon_id(selected_choice, mappings)
        if resolved is None:
            logger.warning("Observation %s: Could not resolve taxonomy choice '%s'", obs_uuid, selected_choice)
            return ParsedTaskAnnotation(record=None, is_ood=False)
        taxon_id = resolved

    if not _validate_licensing_and_attribution(obs_uuid, image_url, photographer, license_code):
        return ParsedTaskAnnotation(record=None, is_ood=False)

    record = _build_verified_record(obs_uuid, taxon_id, data, license_code)
    return ParsedTaskAnnotation(record=record, is_ood=False, new_taxon=novel_spec)


def _collect_feedback_items(task_list: list[dict[str, Any]], mappings: TaxaMappings) -> CollectedFeedback:
    """Parse all exported Label Studio tasks into grouped manifest records and expansions.

    Why:
        Separates per-task annotation parsing and outcome routing (closed-set IDs vs OOD audit vs novel-species registration) cleanly from manifest I/O and keeps cognitive complexity bounded.

    How:
        Invokes `parse_single_task_annotation` across every task in `task_list` and buckets validated in-distribution rows, confirmed-OOD audit records, and `NewTaxonSpec` objects into a frozen `CollectedFeedback` container.

    Args:
        task_list: Sequence of exported Label Studio task dictionaries.
        mappings: Taxonomic lookup mappings loaded from `taxa_catalog.yaml`.

    Returns:
        Aggregated `CollectedFeedback` container.
    """
    verified_records: list[dict[str, Any]] = []
    ood_records: list[dict[str, Any]] = []
    novel_specs: list[NewTaxonSpec] = []
    ood_count = 0

    for task in task_list:
        parsed = parse_single_task_annotation(task, mappings)
        if parsed.is_ood:
            ood_count += 1
            if parsed.ood_record is not None:
                ood_records.append(parsed.ood_record)
            continue
        if parsed.new_taxon is not None:
            novel_specs.append(parsed.new_taxon)
        if parsed.record is not None:
            verified_records.append(parsed.record)

    return CollectedFeedback(
        verified_records=verified_records,
        ood_records=ood_records,
        novel_specs=novel_specs,
        ood_count=ood_count,
    )
