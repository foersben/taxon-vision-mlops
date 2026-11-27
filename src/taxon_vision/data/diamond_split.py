# SPDX-FileCopyrightText: 2026 Benjamin Förster
# SPDX-License-Identifier: MIT
"""Epistemic Diamond 4-way stratified splitting, taxon ID mapping, and `.pt` artefact persistence.

Summary:
    Encapsulates deterministic iNaturalist taxon-to-index mapping from `config/taxa_catalog.yaml`, multi-stage stratified train/val/cal/test partitioning, and serialisation of the four disjoint `.pt` partition tensors (plus optional composite cache).

Why (Architectural Rationale & Trade-offs):
    Decoupling domain data splitting, cardinality guards, and binary `.pt` partition persistence from S3/backbone forward-pass extraction keeps `pipelines/extract_features.py` below 500 LOC (Rule 08) and isolates the Epistemic Diamond DAG split invariant into a cohesive `data/` module with zero cross-package coupling.

How (Algorithmic Mechanism & Invariants):
    Loads authoritative taxon IDs and `inat_aliases` into dense ordinal indices `[0, K-1]`, asserts `>= 2` classes to prevent cardinality collapse, executes recursive stratified scikit-learn `train_test_split`, computes empirical training class frequencies for Class-Balanced Loss, and persists discrete `{train,val,cal,test}_embeddings.pt` tensors.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import torch
from sklearn.model_selection import train_test_split

from taxon_vision.models.training.types import EmbeddingSplit
from taxon_vision.service.inference import load_taxa_catalog

logger = logging.getLogger(__name__)


class CalTestIndices(NamedTuple):
    """Disjoint calibration and holdout test index lists."""

    cal_idx: list[int]
    test_idx: list[int]


class StratifiedSplitIndices(NamedTuple):
    """Disjoint train, validation, calibration, and test index lists."""

    train_idx: list[int]
    val_idx: list[int]
    cal_idx: list[int]
    test_idx: list[int]


@dataclass(frozen=True)
class SplitAllocationSpec:
    """Target sample proportions and PRNG seed for multi-way stratified splitting."""

    val_split: float = 0.10
    cal_split: float = 0.0
    test_split: float = 0.0
    random_seed: int = 42


def get_catalog_inat_taxon_map(catalog: list[dict[str, Any]] | None = None) -> dict[int, int]:
    r"""Dynamically construct authoritative taxon ID to dense class index mapping from the catalog.

    Args:
        catalog: Optional list of catalog dictionaries. If None, loads from `config/taxa_catalog.yaml`.

    Returns:
        Dictionary mapping external taxon IDs to dense class indices in $[0, K-1]$.
    """
    try:
        records: list[dict[str, Any]] = catalog if catalog is not None else load_taxa_catalog()
    except Exception as exc:
        logger.warning("Could not load taxa catalog for default mapping: %s", exc)
        records = []

    mapping: dict[int, int] = {}
    for idx, item in enumerate(records):
        for key in ("inat_taxon_id", "inaturalist_taxon_id", "taxon_id", "id"):
            raw_val = item.get(key)
            if raw_val is not None:
                mapping[int(raw_val)] = idx
        for alias in item.get("inat_aliases", []):
            mapping[int(alias)] = idx
    return mapping


DEFAULT_INAT_TAXON_MAP: dict[int, int] = get_catalog_inat_taxon_map()


def _resolve_raw_catalog(catalog: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    """Resolve raw taxa catalog entries from provided list or disk configuration."""
    if catalog is not None:
        return catalog
    try:
        return load_taxa_catalog()
    except Exception as exc:
        logger.warning("Failed to load taxa catalog from disk: %s", exc)
        return []


def _populate_taxon_indices(mapping: dict[int, int], idx: int, item: dict[str, Any]) -> None:
    """Map taxon identifier candidates from a catalog record to the ordinal index."""
    for key in ("inat_taxon_id", "inaturalist_taxon_id", "taxon_id", "id"):
        raw_val = item.get(key)
        if raw_val is not None:
            mapping[int(raw_val)] = idx
    for alias in item.get("inat_aliases", []):
        mapping[int(alias)] = idx


def build_taxon_index_map(
    catalog: list[dict[str, Any]] | None = None,
    include_defaults: bool = True,
) -> dict[int, int]:
    r"""Construct an authoritative mapping from external iNaturalist taxon IDs to dense zero-indexed labels."""
    mapping: dict[int, int] = dict(DEFAULT_INAT_TAXON_MAP) if include_defaults else {}
    records = _resolve_raw_catalog(catalog)

    for idx, item in enumerate(records):
        _populate_taxon_indices(mapping, idx, item)

    return mapping


def _can_stratify(y_sub: np.ndarray, test_size: int) -> bool:
    """Check if subset has enough samples per class to permit stratified splitting."""
    if len(y_sub) <= test_size or test_size <= 0:
        return False
    _, counts = np.unique(y_sub, return_counts=True)
    n_classes = len(counts)
    return bool(np.min(counts) >= 2 and test_size >= n_classes and (len(y_sub) - test_size) >= n_classes)


def _split_cal_and_test(
    cal_test_idx: list[int],
    labels: np.ndarray,
    counts: tuple[int, int],
    random_seed: int,
) -> CalTestIndices:
    """Partition candidate calibration and test indices into two disjoint subsets."""
    n_cal, n_test = counts
    empty_list: list[int] = []
    if n_test == 0:
        return CalTestIndices(cal_idx=cal_test_idx, test_idx=empty_list)
    if n_cal == 0:
        return CalTestIndices(cal_idx=empty_list, test_idx=cal_test_idx)

    cal_test_labels = labels[cal_test_idx]
    stratify_labels_test = cal_test_labels if _can_stratify(cal_test_labels, n_test) else None
    cal_idx, test_idx = train_test_split(
        cal_test_idx,
        test_size=n_test,
        stratify=stratify_labels_test,
        random_state=random_seed,
    )
    return CalTestIndices(cal_idx=cal_idx, test_idx=test_idx)


def _partition_stratified_subsets(
    indices: list[int],
    labels: np.ndarray,
    spec: SplitAllocationSpec,
) -> StratifiedSplitIndices:
    """Partition observation indices into stratified train, val, cal, and test subsets."""
    n_total = len(indices)
    n_val = max(1, round(n_total * spec.val_split)) if spec.val_split > 0 else 0
    n_cal = max(1, round(n_total * spec.cal_split)) if spec.cal_split > 0 else 0
    n_test = max(1, round(n_total * spec.test_split)) if spec.test_split > 0 else 0
    n_holdout = n_val + n_cal + n_test

    if n_holdout <= 0 or n_holdout >= n_total:
        raise ValueError(f"Holdout sample count ({n_holdout}) must be strictly between 1 and {n_total - 1}.")

    stratify_labels_holdout = labels if _can_stratify(labels, n_holdout) else None
    train_idx, holdout_idx = train_test_split(
        indices,
        test_size=n_holdout,
        stratify=stratify_labels_holdout,
        random_state=spec.random_seed,
    )

    empty_list: list[int] = []
    if n_cal == 0 and n_test == 0:
        return StratifiedSplitIndices(train_idx=train_idx, val_idx=holdout_idx, cal_idx=empty_list, test_idx=empty_list)

    holdout_labels = labels[holdout_idx]
    if n_val == 0:
        val_idx: list[int] = empty_list
        cal_test_idx = holdout_idx
    else:
        stratify_labels_cal_test = holdout_labels if _can_stratify(holdout_labels, n_cal + n_test) else None
        val_idx, cal_test_idx = train_test_split(
            holdout_idx,
            test_size=(n_cal + n_test),
            stratify=stratify_labels_cal_test,
            random_state=spec.random_seed,
        )

    cal_idx, test_idx = _split_cal_and_test(cal_test_idx, labels, (n_cal, n_test), spec.random_seed)
    return StratifiedSplitIndices(train_idx=train_idx, val_idx=val_idx, cal_idx=cal_idx, test_idx=test_idx)


def partition_epistemic_diamond(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    spec: SplitAllocationSpec,
) -> EmbeddingSplit:
    """Partition extracted representations into stratified 4-way Epistemic Diamond splits."""
    unique_classes = torch.unique(labels)
    if unique_classes.numel() < 2:
        raise RuntimeError(
            f"Cardinality collapse detected: Extracted embeddings contain only {unique_classes.numel()} "
            f"distinct class ({unique_classes.tolist()}), but at least 2 are required for multi-class classification."
        )

    indices = list(range(embeddings.shape[0]))
    labels_np = labels.cpu().numpy()

    train_idx, val_idx, cal_idx, test_idx = _partition_stratified_subsets(
        indices=indices,
        labels=labels_np,
        spec=spec,
    )

    train_z, train_y = embeddings[train_idx], labels[train_idx]
    val_z, val_y = embeddings[val_idx], labels[val_idx]
    cal_z = embeddings[cal_idx] if cal_idx else None
    cal_y = labels[cal_idx] if cal_idx else None
    test_z = embeddings[test_idx] if test_idx else None
    test_y = labels[test_idx] if test_idx else None

    samples_per_class = [max(1, int((train_y == c).sum().item())) for c in range(num_classes)]

    return EmbeddingSplit(
        train_embeddings=train_z,
        train_labels=train_y,
        val_embeddings=val_z,
        val_labels=val_y,
        cal_embeddings=cal_z,
        cal_labels=cal_y,
        test_embeddings=test_z,
        test_labels=test_y,
        samples_per_class=samples_per_class,
    )


def save_split_artifacts(
    emb_dir: Path,
    split: EmbeddingSplit,
    extractor: str,
    num_classes: int,
    *,
    output_file: Path | None = None,
) -> None:
    """Save composite embedding split and individual 4-way partition files to disk."""
    if emb_dir.suffix in (".pt", ".bin"):
        actual_output_file: Path | None = emb_dir
        actual_emb_dir: Path = emb_dir.parent
    else:
        actual_output_file = output_file
        actual_emb_dir = emb_dir

    actual_emb_dir.mkdir(parents=True, exist_ok=True)
    if actual_output_file is not None:
        actual_output_file.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "extractor": extractor,
            "feature_dim": split.feature_dim,
            "num_classes": num_classes,
            "split": split,
            "train_embeddings": split.train_embeddings,
            "train_labels": split.train_labels,
            "val_embeddings": split.val_embeddings,
            "val_labels": split.val_labels,
            "cal_embeddings": split.cal_embeddings,
            "cal_labels": split.cal_labels,
            "test_embeddings": split.test_embeddings,
            "test_labels": split.test_labels,
            "samples_per_class": split.samples_per_class,
        }
        torch.save(payload, actual_output_file)
        logger.info("Serialized feature embedding split artifact to: %s", actual_output_file)
    else:
        stale_composite = actual_emb_dir / "cached_features.pt"
        if stale_composite.exists():
            stale_composite.unlink()
            logger.info("Purged obsolete composite cache %s to eliminate dead-ends", stale_composite)

    torch.save(
        {
            "embeddings": split.train_embeddings,
            "labels": split.train_labels,
            "samples_per_class": split.samples_per_class,
            "feature_dim": split.feature_dim,
        },
        actual_emb_dir / "train_embeddings.pt",
    )
    torch.save(
        {
            "embeddings": split.val_embeddings,
            "labels": split.val_labels,
            "feature_dim": split.feature_dim,
        },
        actual_emb_dir / "val_embeddings.pt",
    )
    if split.cal_embeddings is not None and split.cal_labels is not None:
        torch.save(
            {
                "embeddings": split.cal_embeddings,
                "labels": split.cal_labels,
                "feature_dim": split.feature_dim,
            },
            actual_emb_dir / "cal_embeddings.pt",
        )
    if split.test_embeddings is not None and split.test_labels is not None:
        torch.save(
            {
                "embeddings": split.test_embeddings,
                "labels": split.test_labels,
                "feature_dim": split.feature_dim,
            },
            actual_emb_dir / "test_embeddings.pt",
        )
    logger.info("Materialized individual 4-way split artifacts in: %s", actual_emb_dir)
