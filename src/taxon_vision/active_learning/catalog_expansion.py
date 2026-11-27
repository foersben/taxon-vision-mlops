# SPDX-FileCopyrightText: 2026 Benjamin Förster
# SPDX-License-Identifier: MIT
"""Outcome B taxonomic catalogue expansion persistence and Jenkins staging commit helpers.

Summary:
    Handles idempotent registration of expert-verified unseen species (`NewTaxonSpec`) into `taxa_catalog.yaml`, `params.yaml`, and `deploy/label_studio/project_config.xml`, or stages pending expansions to `catalog_expansions.yaml` when Airflow mounts configs read-only.

Why (Architectural Rationale & Trade-offs):
    Decoupling YAML/XML catalogue mutation and read-only Kubernetes staging logic from Parquet manifest ingestion keeps `feedback.py` below 500 LOC (Rule 08) and isolates atomic UTF-8 config updates into one cohesive sibling module.

How (Algorithmic Mechanism & Invariants):
    Checks writability via `_is_path_writable`: writable checkouts (local/Jenkins) receive atomic sibling-temp-file updates across catalog YAML, params YAML, and Label Studio XML; read-only Airflow ConfigMap mounts stage pending `NewTaxonSpec` entries in `catalog_expansions.yaml` for `apply_pending_catalog_expansions` during GPU retraining.
"""

from __future__ import annotations

import logging
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape

import yaml

from taxon_vision.active_learning.feedback_parser import NewTaxonSpec, load_taxa_mappings

logger = logging.getLogger(__name__)


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


def _apply_writable_expansions(
    unseen_specs: list[NewTaxonSpec],
    catalog_path: Path,
    resolved_options: FeedbackMergeOptions,
) -> ExpansionPersistenceResult | None:
    """Apply unseen taxon expansions directly to writable config files when possible.

    Why:
        Isolating the writable-filesystem branch and its `OSError` fallback from `_persist_catalog_expansions` keeps cognitive complexity well below the Rule 00 ceiling whilst preserving deterministic failover to external staging on read-only mounts.

    How:
        Verifies `_is_path_writable(catalog_path)`, atomically registers new taxa in `taxa_catalog.yaml` plus any writable companion `params.yaml` or Label Studio XML paths, and returns `None` if `catalog_path` is read-only or fails with `OSError`.

    Args:
        unseen_specs: Novel species specifications absent from the active catalogue.
        catalog_path: Target `taxa_catalog.yaml` path.
        resolved_options: Resolved companion target paths from `_resolve_expansion_paths`.

    Returns:
        `ExpansionPersistenceResult` on direct write success, or `None` when read-only staging is required.
    """
    if not _is_path_writable(catalog_path):
        return None
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
        return None


def _persist_catalog_expansions(
    specs: list[NewTaxonSpec],
    catalog_path: Path,
    resolved_options: FeedbackMergeOptions,
) -> ExpansionPersistenceResult:
    """Apply novel taxa directly to writable configs or stage them in `catalog_expansions.yaml`.

    Why:
        Supports both local/Jenkins writable checkouts (direct update) and read-only Airflow pod mounts (external YAML staging) uniformly through a single contract.

    How:
        Filters out taxon IDs already registered in `catalog_path`, attempts direct writes via `_apply_writable_expansions`, and falls back to `_write_pending_expansions` when config manifests are mounted read-only.

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

    direct_result = _apply_writable_expansions(unseen_specs, catalog_path, resolved_options)
    if direct_result is not None:
        return direct_result

    if resolved_options.pending_expansions_path is None:
        return ExpansionPersistenceResult(catalog_expanded=False, new_taxon_ids=())
    staged_ids = _write_pending_expansions(resolved_options.pending_expansions_path, unseen_specs)
    return ExpansionPersistenceResult(catalog_expanded=bool(staged_ids), new_taxon_ids=staged_ids)


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
