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

import argparse
import json
import logging
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple
from xml.sax.saxutils import escape

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


@dataclass(frozen=True)
class FeedbackMergeOptions:
    """Optional filesystem targets governing OOD logging and catalogue expansion persistence.

    Attributes:
        params_path: Optional path to pipeline parameters YAML containing ``ingestion.taxa_to_fetch``.
        label_studio_config_path: Optional path to the Label Studio labelling configuration XML.
        pending_expansions_path: Optional staging YAML path used when config manifests are mounted read-only.
        ood_manifest_path: Optional Parquet path for persisting expert-confirmed OOD specimens.
    """

    params_path: Path | None = None
    label_studio_config_path: Path | None = None
    pending_expansions_path: Path | None = None
    ood_manifest_path: Path | None = None


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


@dataclass(frozen=True)
class CollectedFeedback:
    """Intermediate aggregation of parsed feedback records grouped by ingestion outcome."""

    verified_records: list[dict[str, Any]]
    ood_records: list[dict[str, Any]]
    novel_specs: list[NewTaxonSpec]
    ood_count: int


@dataclass(frozen=True)
class ParquetAppendStats:
    """Counts resulting from deduplicating and appending rows to a Parquet table."""

    added_count: int
    total_count: int


@dataclass(frozen=True)
class ExpansionPersistenceResult:
    """Outcome of recording or applying novel taxonomic catalogue expansions."""

    catalog_expanded: bool
    new_taxon_ids: tuple[int, ...]


def _atomic_write_text(content: str, target_path: Path) -> None:
    """Atomically write UTF-8 text via a temporary sibling file.

    Why:
        In-place truncation (`open(..., "w")`) leaves corrupted YAML or XML manifests if an Airflow pod is evicted mid-write; sibling temporary files + POSIX `replace()` guarantee all-or-nothing updates.

    How:
        Creates a sibling `NamedTemporaryFile` inside `target_path.parent`, flushes UTF-8 `content`, atomically replaces `target_path`, and unlinks the temp file on failure.

    Args:
        content: Complete UTF-8 text payload to persist.
        target_path: Final destination file path.
    """
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=target_path.parent, delete=False, suffix=target_path.suffix, mode="w", encoding="utf-8"
    ) as tmp:
        tmp.write(content)
        tmp_path = Path(tmp.name)
    try:
        tmp_path.replace(target_path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise


def _atomic_write_manifest(df: pd.DataFrame, manifest_path: Path) -> None:
    """Write a DataFrame atomically to a target Parquet path via a temporary sibling file.

    Why:
        Partial Parquet writes break PyArrow footer parsing across downstream DVC stages and dataset audits; atomic `replace()` prevents concurrent readers from seeing half-written files.

    How:
        Serialises `df` into a sibling temporary `.parquet` file via pandas/PyArrow and renames it atomically onto `manifest_path`, cleaning up on error.

    Args:
        df: Updated manifest DataFrame.
        manifest_path: Destination Parquet path on disk.
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


def _is_path_writable(target: Path) -> bool:
    """Return True when an existing file or its parent directory permits writes.

    Why:
        Airflow Kubernetes pods mount ConfigMap-backed files such as `config/taxa_catalog.yaml` read-only (`0444`), so Outcome B must detect writability safely before attempting direct config updates vs staging `catalog_expansions.yaml`.

    How:
        Checks `os.access(target, os.W_OK)` when `target` exists, or checks writability of `target.parent`.

    Args:
        target: Candidate configuration file path.

    Returns:
        True if mutations to `target` are permitted by filesystem permissions, False otherwise.
    """
    if target.exists():
        return os.access(target, os.W_OK)
    return target.parent.exists() and os.access(target.parent, os.W_OK)


def _append_deduplicated_parquet(
    records: list[dict[str, Any]],
    target_path: Path,
    columns: list[str],
) -> ParquetAppendStats:
    """Deduplicate records by `observation_uuid` and atomically append new rows to Parquet.

    Why:
        Repeated exports from DagsHub Label Studio contain already-ingested tasks; O(N) set-based deduplication guarantees idempotency across scheduled/retried DAG runs for both `raw_manifest.parquet` and `ood_verified.parquet`.

    How:
        Reads `target_path` if present (otherwise initialises an empty frame with `columns`), builds a set of existing `observation_uuid` strings, filters `records`, concatenates new rows, and writes via `_atomic_write_manifest`.

    Args:
        records: Candidate record dicts to append.
        target_path: Destination Parquet file path.
        columns: Canonical column list used when creating a new file.

    Returns:
        `ParquetAppendStats` holding counts of newly added rows and total rows.
    """
    existing_df = pd.read_parquet(target_path) if target_path.is_file() else pd.DataFrame(columns=columns)
    existing_uuids = (
        set(existing_df["observation_uuid"].astype(str)) if "observation_uuid" in existing_df.columns else set()
    )
    new_records = [rec for rec in records if str(rec["observation_uuid"]) not in existing_uuids]
    if not new_records:
        return ParquetAppendStats(added_count=0, total_count=len(existing_df))

    updated_df = pd.concat([existing_df, pd.DataFrame(new_records)], ignore_index=True)
    _atomic_write_manifest(updated_df, target_path)
    return ParquetAppendStats(added_count=len(new_records), total_count=len(updated_df))


def _deduplicate_novel_specs(specs: list[NewTaxonSpec]) -> list[NewTaxonSpec]:
    """Preserve encounter order whilst deduplicating novel taxon specs by iNaturalist taxon ID.

    Why:
        Multiple flagged captures in the same export batch frequently introduce the same novel species; deduplicating before YAML/XML writes guarantees unique sequential class IDs.

    How:
        Inserts each `NewTaxonSpec` into an insertion-ordered map keyed by `inat_taxon_id` on first appearance.

    Args:
        specs: Raw sequence of extracted `NewTaxonSpec` objects.

    Returns:
        Deduplicated list of `NewTaxonSpec` objects preserving first-seen ordering.
    """
    unique: dict[int, NewTaxonSpec] = {}
    for spec in specs:
        if spec.inat_taxon_id not in unique:
            unique[spec.inat_taxon_id] = spec
    return list(unique.values())


def _register_taxa_in_catalog(catalog_path: Path, specs: list[NewTaxonSpec]) -> list[int]:
    """Idempotently append unseen target taxa to `taxa_catalog.yaml`.

    Why:
        Expanding from `K` to `K + Delta K` classes requires assigning contiguous 1-based internal `id` integers in `taxa_catalog.yaml` so classifier logits and taxonomic labels align deterministically.

    How:
        Loads existing entries from `catalog_path`, skips any `inat_taxon_id` already present, appends new entries starting at `max(id) + 1`, and writes atomically if changed.

    Args:
        catalog_path: Writable filesystem path to `config/taxa_catalog.yaml`.
        specs: Deduplicated list of novel species to register.

    Returns:
        List of newly added integer iNaturalist taxon IDs.
    """
    with catalog_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    taxa_list = list(data.get("taxa", []))
    existing_ids = {int(entry["inat_taxon_id"]) for entry in taxa_list}
    next_id = max((int(entry.get("id", 0)) for entry in taxa_list), default=0) + 1
    added_ids: list[int] = []

    for spec in specs:
        if spec.inat_taxon_id in existing_ids:
            continue
        taxa_list.append(
            {
                "id": next_id,
                "scientific_name": spec.scientific_name,
                "common_name": spec.common_name,
                "kingdom": spec.kingdom,
                "class": spec.taxon_class,
                "inat_taxon_id": spec.inat_taxon_id,
            }
        )
        existing_ids.add(spec.inat_taxon_id)
        added_ids.append(spec.inat_taxon_id)
        next_id += 1

    if added_ids:
        data["taxa"] = taxa_list
        _atomic_write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), catalog_path)
    return added_ids


def _register_taxa_in_params(params_path: Path, specs: list[NewTaxonSpec]) -> None:
    """Idempotently append unseen target taxon IDs to `params.yaml` `ingestion.taxa_to_fetch`.

    Why:
        DVC Stage 1 (`ingest.py`) harvests API observations from `params.yaml:ingestion.taxa_to_fetch`; updating this list alongside `taxa_catalog.yaml` ensures newly admitted species receive balanced citizen-science samples automatically.

    How:
        Parses `params_path`, appends missing `spec.inat_taxon_id` integers to `ingestion['taxa_to_fetch']`, and writes back atomically when modified.

    Args:
        params_path: Writable path to root `params.yaml`.
        specs: Novel species specifications to add.
    """
    if not params_path.is_file():
        return
    with params_path.open("r", encoding="utf-8") as handle:
        params_data = yaml.safe_load(handle) or {}
    ingestion = params_data.setdefault("ingestion", {})
    taxa_to_fetch = [int(tid) for tid in ingestion.get("taxa_to_fetch", [])]
    changed = False
    for spec in specs:
        if spec.inat_taxon_id not in taxa_to_fetch:
            taxa_to_fetch.append(spec.inat_taxon_id)
            changed = True
    if changed:
        ingestion["taxa_to_fetch"] = taxa_to_fetch
        _atomic_write_text(yaml.safe_dump(params_data, sort_keys=False, allow_unicode=True), params_path)


def _register_taxa_in_label_studio(xml_path: Path, specs: list[NewTaxonSpec]) -> None:
    """Idempotently insert newly registered taxa into the Label Studio XML interface.

    Why:
        Keeps future DagsHub Label Studio triage tasks synchronised with newly admitted species so annotators select them directly instead of re-entering Outcome B text fields.

    How:
        Checks `xml_path` for `alias="{inat_taxon_id}"` markers, formats XML-escaped `<Choice>` elements, and inserts them immediately before the `"Out of Distribution"` anchor via `_atomic_write_text`.

    Args:
        xml_path: Writable path to `deploy/label_studio/project_config.xml`.
        specs: Novel species specifications to expose in the annotation UI.
    """
    if not xml_path.is_file():
        return
    xml_text = xml_path.read_text(encoding="utf-8")
    new_lines: list[str] = []
    for spec in specs:
        alias_marker = f'alias="{spec.inat_taxon_id}"'
        if alias_marker in xml_text:
            continue
        label_attr = escape(f"{spec.scientific_name} ({spec.common_name})", {'"': "&quot;"})
        new_lines.append(f'    <Choice value="{label_attr}" alias="{spec.inat_taxon_id}" />\n')

    if not new_lines:
        return
    insertion = "".join(new_lines)
    ood_anchor = '    <Choice value="Out of Distribution'
    if ood_anchor in xml_text:
        updated_xml = xml_text.replace(ood_anchor, f"{insertion}{ood_anchor}", 1)
    else:
        updated_xml = xml_text.replace("  </Choices>", f"{insertion}  </Choices>", 1)
    _atomic_write_text(updated_xml, xml_path)


def _write_pending_expansions(expansions_path: Path, specs: list[NewTaxonSpec]) -> tuple[int, ...]:
    """Persist pending catalogue expansion specs to YAML when config files are read-only.

    Why:
        Because Airflow mounts ConfigMap manifests read-only inside K3s, staging novel species into the shared writable manifest PVC (`catalog_expansions.yaml`) transfers expansion intent safely to the Jenkins GPU workspace.

    How:
        Merges new `NewTaxonSpec` dicts with any existing unapplied entries in `expansions_path`, writes atomically on change, and returns the sorted tuple of all pending taxon IDs so downstream retries preserve `catalog_expanded=True`.

    Args:
        expansions_path: Writable staging file path (typically `data/manifests/catalog_expansions.yaml`).
        specs: Newly discovered novel species metadata.

    Returns:
        Sorted tuple of all pending integer iNaturalist taxon IDs.
    """
    existing_entries: list[dict[str, Any]] = []
    if expansions_path.is_file():
        with expansions_path.open("r", encoding="utf-8") as handle:
            existing_data = yaml.safe_load(handle) or {}
            existing_entries = list(existing_data.get("expansions", []))

    seen_ids = {int(item["inat_taxon_id"]) for item in existing_entries if "inat_taxon_id" in item}
    added_ids: list[int] = []
    for spec in specs:
        if spec.inat_taxon_id in seen_ids:
            continue
        existing_entries.append(
            {
                "inat_taxon_id": spec.inat_taxon_id,
                "scientific_name": spec.scientific_name,
                "common_name": spec.common_name,
                "kingdom": spec.kingdom,
                "class": spec.taxon_class,
            }
        )
        seen_ids.add(spec.inat_taxon_id)
        added_ids.append(spec.inat_taxon_id)

    if added_ids:
        _atomic_write_text(
            yaml.safe_dump({"expansions": existing_entries}, sort_keys=False, allow_unicode=True), expansions_path
        )
    return tuple(sorted(seen_ids))


def _resolve_expansion_paths(
    catalog_path: Path,
    manifest_path: Path,
    options: FeedbackMergeOptions | None,
) -> FeedbackMergeOptions:
    """Resolve optional companion configuration paths without mutating unintended workspace files.

    Why:
        Unit tests invoke `merge_feedback_to_manifest` against temporary catalogs, whereas production/CI runs target repository root `params.yaml` and `deploy/label_studio/project_config.xml`. Resolving relative to `catalog_path` ensures tests never leak edits into real repository configs.

    How:
        Honours explicit overrides inside `options` and falls back to repository/manifest sibling defaults (`catalog_expansions.yaml`, `ood_verified.parquet`) only when detected.

    Args:
        catalog_path: Resolved path to `taxa_catalog.yaml`.
        manifest_path: Resolved path to `raw_manifest.parquet`.
        options: Optional user/caller `FeedbackMergeOptions` overrides.

    Returns:
        Fully populated `FeedbackMergeOptions` container.
    """
    opts = options or FeedbackMergeOptions()
    candidate_params = catalog_path.parent.parent / "params.yaml"
    candidate_xml = catalog_path.parent.parent / "deploy" / "label_studio" / "project_config.xml"
    return FeedbackMergeOptions(
        params_path=opts.params_path
        if opts.params_path is not None
        else (candidate_params if candidate_params.is_file() else None),
        label_studio_config_path=(
            opts.label_studio_config_path
            if opts.label_studio_config_path is not None
            else (candidate_xml if candidate_xml.is_file() else None)
        ),
        pending_expansions_path=(
            opts.pending_expansions_path
            if opts.pending_expansions_path is not None
            else manifest_path.parent / "catalog_expansions.yaml"
        ),
        ood_manifest_path=(
            opts.ood_manifest_path
            if opts.ood_manifest_path is not None
            else manifest_path.parent / "ood_verified.parquet"
        ),
    )


def _persist_catalog_expansions(
    specs: list[NewTaxonSpec],
    catalog_path: Path,
    resolved_options: FeedbackMergeOptions,
) -> ExpansionPersistenceResult:
    """Apply novel taxa directly to writable configs or stage them in `catalog_expansions.yaml`.

    Why:
        Supports both local/Jenkins writable checkouts (direct update) and read-only Airflow pod mounts (external YAML staging) uniformly through a single contract.

    How:
        Filters out taxon IDs already registered in `catalog_path`; if `catalog_path` is writable, updates catalog, params, and Label Studio XML directly, otherwise delegates to `_write_pending_expansions`.

    Args:
        specs: Raw novel species specs collected from annotations.
        catalog_path: Path to `taxa_catalog.yaml`.
        resolved_options: Resolved companion target paths from `_resolve_expansion_paths`.

    Returns:
        `ExpansionPersistenceResult` signalling whether catalogue expansion occurred and the affected IDs.
    """
    deduped = _deduplicate_novel_specs(specs)
    if not deduped:
        return ExpansionPersistenceResult(catalog_expanded=False, new_taxon_ids=())

    existing_catalog_ids = load_taxa_mappings(catalog_path).valid_ids
    unseen_specs = [spec for spec in deduped if spec.inat_taxon_id not in existing_catalog_ids]
    if not unseen_specs:
        return ExpansionPersistenceResult(catalog_expanded=False, new_taxon_ids=())

    if _is_path_writable(catalog_path):
        try:
            added_ids = _register_taxa_in_catalog(catalog_path, unseen_specs)
            if resolved_options.params_path is not None and _is_path_writable(resolved_options.params_path):
                _register_taxa_in_params(resolved_options.params_path, unseen_specs)
            if resolved_options.label_studio_config_path is not None and _is_path_writable(
                resolved_options.label_studio_config_path
            ):
                _register_taxa_in_label_studio(resolved_options.label_studio_config_path, unseen_specs)
            return ExpansionPersistenceResult(catalog_expanded=bool(added_ids), new_taxon_ids=tuple(added_ids))
        except OSError:
            logger.info("Catalog path %s is read-only; staging expansions externally", catalog_path)

    if resolved_options.pending_expansions_path is None:
        return ExpansionPersistenceResult(catalog_expanded=False, new_taxon_ids=())
    staged_ids = _write_pending_expansions(resolved_options.pending_expansions_path, unseen_specs)
    return ExpansionPersistenceResult(catalog_expanded=bool(staged_ids), new_taxon_ids=staged_ids)


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


def apply_pending_catalog_expansions(
    expansions_path: Path | str = "data/manifests/catalog_expansions.yaml",
    catalog_path: Path | str = "config/taxa_catalog.yaml",
    params_path: Path | str = "params.yaml",
    label_studio_config_path: Path | str = "deploy/label_studio/project_config.xml",
) -> list[int]:
    """Commit staged Outcome B catalogue expansions into repository YAML and XML files.

    Why:
        In Kubernetes deployments Airflow mounts configuration manifests read-only whilst raw training manifests live on a writable shared volume. Jenkins executes this commit step on the writable CI workspace immediately prior to DVC pipeline execution so K+DeltaK classes propagate cleanly into ingestion, feature extraction, and training.

    How:
        Loads pending NewTaxonSpec entries from expansions_path, idempotently updates taxa_catalog.yaml, params.yaml, and project_config.xml, and clears the pending expansion queue.

    Args:
        expansions_path: YAML file produced by merge_feedback_to_manifest containing queued taxa.
        catalog_path: Writable path to taxa_catalog.yaml.
        params_path: Writable path to params.yaml containing ingestion.taxa_to_fetch.
        label_studio_config_path: Writable path to deploy/label_studio/project_config.xml.

    Returns:
        List of newly committed integer iNaturalist taxon IDs.
    """
    exp_path = Path(expansions_path)
    if not exp_path.is_file():
        return []
    with exp_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    raw_entries = payload.get("expansions", [])
    if not raw_entries:
        return []

    specs = [
        NewTaxonSpec(
            inat_taxon_id=int(entry["inat_taxon_id"]),
            scientific_name=str(entry["scientific_name"]).strip(),
            common_name=str(entry.get("common_name") or entry["scientific_name"]).strip(),
            kingdom=str(entry.get("kingdom") or "Unknown").strip(),
            taxon_class=str(entry.get("class") or "Unknown").strip(),
        )
        for entry in raw_entries
        if "inat_taxon_id" in entry and "scientific_name" in entry
    ]
    cat_path = Path(catalog_path)
    added_ids = _register_taxa_in_catalog(cat_path, specs)
    _register_taxa_in_params(Path(params_path), specs)
    _register_taxa_in_label_studio(Path(label_studio_config_path), specs)
    _atomic_write_text(yaml.safe_dump({"expansions": []}, sort_keys=False), exp_path)
    logger.info("Committed %d pending catalogue expansion(s): %s", len(added_ids), added_ids)
    return added_ids


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
