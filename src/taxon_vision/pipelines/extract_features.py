# SPDX-FileCopyrightText: 2026 Benjamin Förster
# SPDX-License-Identifier: MIT
"""Vision foundation feature extraction and offline embedding caching pipeline."""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torchvision import transforms

from taxon_vision.config import get_settings
from taxon_vision.data.s3_streamer import S3ImageStreamer
from taxon_vision.models.factory import create_feature_extractor, extract_features
from taxon_vision.models.training.types import EmbeddingSplit
from taxon_vision.service.inference import load_taxa_catalog

logger = logging.getLogger(__name__)


def get_catalog_inat_taxon_map(catalog: list[dict[str, Any]] | None = None) -> dict[int, int]:
    r"""Dynamically construct authoritative taxon ID to dense class index mapping from the catalog.

    Why:
        Hardcoding taxon IDs and integer maps inside pipeline scripts is an antipattern that causes configuration drift. Deriving mappings directly from `config/taxa_catalog.yaml` guarantees a Single Source of Truth (SSOT).

    How:
        Iterates over catalog entries, extracting primary `inat_taxon_id`, `id`, and any defined `inat_aliases`, mapping each to the dense ordinal class index in $[0, K-1]$.

    Args:
        catalog: Optional list of catalog dictionaries. If None, loads from `config/taxa_catalog.yaml`.

    Returns:
        Dictionary mapping external taxon IDs to dense class indices in $[0, K-1]$.
    """
    try:
        records: list[dict[str, Any]] = catalog if catalog is not None else load_taxa_catalog()
    except Exception:
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


# Canonical mapping dynamically built from authoritative config/taxa_catalog.yaml (SSOT)
DEFAULT_INAT_TAXON_MAP: dict[int, int] = get_catalog_inat_taxon_map()


@dataclass(frozen=True)
class FeatureExtractionConfig:
    """Immutable configuration container governing feature extraction and representation caching.

    Why:
        Decouples pipeline execution parameters from command-line parsing or ad-hoc function arguments. Providing a strongly-typed, frozen dataclass guarantees parameter immutability throughout the long-running extraction process, preventing accidental runtime mutation of paths or hyperparameters.

    Attributes:
        extractor: Name of the vision foundation backbone (e.g. `mobilenetv4_conv_small`, `dinov3_vits14`).
        manifest_path: Filesystem path to the Parquet dataset manifest containing observation URLs.
        output_path: Filesystem destination path where the serialized `EmbeddingSplit` dictionary is saved.
        batch_size: Number of images grouped into a single forward tensor pass through the backbone.
        val_split: Fraction of extracted representations reserved for the holdout validation split (0.0 to 1.0).
        device: PyTorch device identifier (`cuda`, `cpu`, or `None` for automatic hardware discovery).
        limit: Optional maximum number of observation records to process from the manifest (for dry-runs).
        image_size: Target square pixel dimension for cropped model input tensors (default: 224).
        resize_size: Dimension to which the shorter image edge is resized prior to center cropping (default: 256).
        random_seed: PRNG seed for deterministic stratified train/validation partitioning.
    """

    extractor: str = ""
    manifest_path: Path = Path("data/manifests/dataset_manifest.parquet")
    output_path: Path | None = None
    out_dir: Path = Path("data/embeddings")
    batch_size: int = 0
    val_split: float = 0.0
    cal_split: float = 0.0
    test_split: float = 0.0
    device: str | None = None
    limit: int | None = None
    image_size: int = 0
    resize_size: int = 0
    random_seed: int = 0

    def __post_init__(self) -> None:
        """Resolve unspecified hyperparameters dynamically from centralized settings."""
        if self.output_path is not None and self.out_dir == Path("data/embeddings"):
            object.__setattr__(self, "out_dir", self.output_path.parent)
        settings = get_settings()
        if not self.extractor:
            object.__setattr__(self, "extractor", settings.model.default_extractor)
        if self.batch_size <= 0:
            object.__setattr__(self, "batch_size", settings.training.batch_size)
        if self.val_split <= 0.0:
            object.__setattr__(self, "val_split", settings.training.val_split)
        if self.cal_split <= 0.0:
            object.__setattr__(self, "cal_split", settings.training.cal_split)
        if self.test_split <= 0.0:
            object.__setattr__(self, "test_split", settings.training.test_split)
        if self.image_size <= 0:
            object.__setattr__(self, "image_size", settings.model.image_size)
        if self.resize_size <= 0:
            resolved_size = self.image_size if self.image_size > 0 else settings.model.image_size
            object.__setattr__(self, "resize_size", round(resolved_size * settings.model.eval_resize_ratio))
        if self.random_seed == 0:
            object.__setattr__(self, "random_seed", settings.training.random_seed)


def get_evaluation_transform(
    image_size: int | None = None,
    resize_size: int | None = None,
    mean: list[float] | None = None,
    std: list[float] | None = None,
) -> transforms.Compose:
    """Construct deterministic image preprocessing transform for vision foundation model evaluation.

    Why:
        Linear probing on pre-trained vision representations requires rigorous adherence to the upstream evaluation preprocessing protocol. Stochastic data augmentations (e.g., random affine transformations, color jittering, or horizontal flips) are omitted during representation caching because foundation backbone features must represent the canonical visual signal. Pre-normalizing images against standard ImageNet channel statistics ensures that downstream linear heads receive correctly calibrated vectors.

    How:
        Composes a sequential torchvision transformation pipeline comprising:
        1. Bilinear aspect-ratio preserving resizing of the shorter edge to `resize_size`.
        2. Exact square center cropping to `(image_size, image_size)`.
        3. Conversion from uint8 PIL RGB image buffer to float32 tensor scaled to $[0.0, 1.0]$.
        4. Channel-wise mean and variance normalization from model configuration.

    Args:
        image_size: Spatial height and width of the output tensor (defaults to settings.model.image_size).
        resize_size: Interpolation target dimension for the shorter edge (defaults to round(image_size * eval_resize_ratio)).
        mean: Normalization channel means (defaults to settings.model.mean).
        std: Normalization channel standard deviations (defaults to settings.model.std).

    Returns:
        Callable torchvision Compose transformation pipeline.
    """
    cfg = get_settings().model
    resolved_image_size = image_size if image_size is not None else cfg.image_size
    resolved_resize_size = (
        resize_size if resize_size is not None else round(resolved_image_size * cfg.eval_resize_ratio)
    )
    resolved_mean = mean if mean is not None else cfg.mean
    resolved_std = std if std is not None else cfg.std
    return transforms.Compose(
        [
            transforms.Resize(resolved_resize_size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(resolved_image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=resolved_mean,
                std=resolved_std,
            ),
        ]
    )


def _resolve_raw_catalog(catalog: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    """Resolve raw taxa catalog entries from provided list or disk configuration.

    Args:
        catalog: Optional user-supplied catalog entries.

    Returns:
        List of taxon dictionary records.
    """
    if catalog is not None:
        return catalog
    try:
        return load_taxa_catalog()
    except Exception:
        return []


def _populate_taxon_indices(mapping: dict[int, int], idx: int, item: dict[str, Any]) -> None:
    """Map taxon identifier candidates from a catalog record to the ordinal index.

    Args:
        mapping: Target mapping dictionary to mutate.
        idx: Ordinal class index.
        item: Taxon metadata record.
    """
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
    r"""Construct an authoritative mapping from external iNaturalist taxon IDs to dense zero-indexed labels.

    Why:
        Biological taxonomies in citizen-science repositories use arbitrary, non-contiguous integer IDs (e.g., Monarch Butterfly is `48662`, Western Honey Bee is `47120`). Deep learning classification heads and loss functions (such as PyTorch `CrossEntropyLoss` or `ClassBalancedLoss`) mathematically require target labels to be dense integers bounded by $0 \le y < K$, where $K$ is the class count. Creating a deterministic mapping guarantees alignment between observation metadata and head logits.

    How:
        Initializes baseline mappings from `DEFAULT_INAT_TAXON_MAP` when `include_defaults` is True. Resolves catalog items via `_resolve_raw_catalog` and maps candidate keys via `_populate_taxon_indices`.

    Args:
        catalog: Optional pre-loaded list of taxa metadata dictionaries.
        include_defaults: Whether to include default iNaturalist ID mappings.

    Returns:
        Dictionary mapping external iNaturalist integer taxon IDs to dense indices in $[0, K-1]$.
    """
    mapping: dict[int, int] = dict(DEFAULT_INAT_TAXON_MAP) if include_defaults else {}
    records = _resolve_raw_catalog(catalog)

    for idx, item in enumerate(records):
        _populate_taxon_indices(mapping, idx, item)

    return mapping


def extract_batch_representations(
    backbone: torch.nn.Module,
    image_tensors: list[torch.Tensor],
    device: torch.device,
) -> torch.Tensor:
    """Execute a zero-gradient forward pass over an image batch to extract pooled representation vectors.

    Why:
        Encapsulates tensor batch collation and device dispatch into an isolated, unit-testable function. Enforces `torch.no_grad()` to prevent activation graph construction, drastically bounding VRAM consumption during high-throughput feature caching.

    How:
        Stacks individual 3D image tensors `(3, H, W)` into a contiguous 4D batch tensor `(B, 3, H, W)`, transfers the tensor to the designated accelerator device, invokes `extract_features`, and returns the detached representation matrix `(B, D)` residing in host CPU memory.

    Args:
        backbone: Frozen vision foundation model in evaluation mode (`eval()`).
        image_tensors: List of pre-processed 3D PyTorch image tensors.
        device: Hardware device (`cuda` or `cpu`) executing the forward matrix multiplications.

    Returns:
        2D tensor of dense representations of shape `(batch_size, feature_dim)` on CPU.
    """
    if not image_tensors:
        return torch.empty((0, 0), dtype=torch.float32)

    with torch.no_grad():
        batch = torch.stack(image_tensors).to(device)
        embeddings = extract_features(backbone, batch, device=device)
        return embeddings.detach().cpu()


def stream_and_extract_representations(
    streamer: S3ImageStreamer | Any,
    backbone: torch.nn.Module,
    taxon_to_index: dict[int, int],
    transform: transforms.Compose,
    batch_size: int,
    device: torch.device,
    limit: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Lazily stream remote imagery, apply transformations, and extract representations in mini-batches.

    Why:
        Streaming biodiversity imagery lazily from open cloud object stores (e.g., AWS Open Data iNaturalist S3) avoids pre-allocating hundreds of gigabytes of disk storage on edge workstations. Grouping streamed images into mini-batches maximizes GPU Tensor Core saturation compared to single-image inference.

    How:
        Iterates over the `streamer.stream_images` lazy generator. Validates each observation's taxon identifier against `taxon_to_index`. Preprocesses valid images into tensors, accumulates mini-batches of size `batch_size`, and dispatches them to `extract_batch_representations`. Concatenates all batch tensors into unified feature matrix $X \in \mathbb{R}^{N \times D}$ and label vector $y \in \mathbb{Z}^N$.

    Args:
        streamer: Instantiated S3ImageStreamer pointing to a valid Parquet dataset manifest.
        backbone: Frozen neural feature extractor model.
        taxon_to_index: Dictionary mapping remote taxon IDs to integer classes.
        transform: Image preprocessing transform.
        batch_size: Batch size for representation extraction passes.
        device: PyTorch device running extraction.
        limit: Optional upper bound on observations to stream.

    Returns:
        Tuple of `(embeddings, labels)` where `embeddings` has shape `(N, D)` and `labels` has shape `(N,)`.

    Raises:
        ValueError: If zero compliant images could be streamed or mapped from the manifest.
    """
    collected_embeddings: list[torch.Tensor] = []
    collected_labels: list[int] = []

    pending_tensors: list[torch.Tensor] = []
    pending_labels: list[int] = []

    count = 0
    for obs_uuid, taxon_id, pil_img in streamer.stream_images(limit=limit, shuffle=True, balanced=True):
        if taxon_id not in taxon_to_index:
            logger.debug("Skipping unmapped taxon ID %d for observation %s", taxon_id, obs_uuid)
            continue

        label_idx = taxon_to_index[taxon_id]
        tensor = transform(pil_img)

        pending_tensors.append(tensor)
        pending_labels.append(label_idx)
        count += 1

        if len(pending_tensors) >= batch_size:
            feats = extract_batch_representations(backbone, pending_tensors, device=device)
            collected_embeddings.append(feats)
            collected_labels.extend(pending_labels)
            pending_tensors.clear()
            pending_labels.clear()
            logger.info("Processed %d observation embeddings...", count)

    if pending_tensors:
        feats = extract_batch_representations(backbone, pending_tensors, device=device)
        collected_embeddings.append(feats)
        collected_labels.extend(pending_labels)
        pending_tensors.clear()
        pending_labels.clear()

    if not collected_embeddings:
        raise ValueError(
            "Feature extraction yielded zero samples. Ensure dataset_manifest.parquet contains "
            "valid image URLs and taxon IDs matching the target perimeter."
        )

    all_x = torch.cat(collected_embeddings, dim=0)
    all_y = torch.tensor(collected_labels, dtype=torch.long)
    return all_x, all_y


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
    n_cal: int,
    n_test: int,
    random_seed: int,
) -> tuple[list[int], list[int]]:
    """Partition candidate calibration and test indices into two disjoint subsets.

    Args:
        cal_test_idx: Indices representing the combined calibration and test pool.
        labels: 1D NumPy array of ground truth labels for stratification.
        n_cal: Target sample count for calibration.
        n_test: Target sample count for holdout test.
        random_seed: PRNG seed for deterministic splitting.

    Returns:
        Tuple of (cal_idx, test_idx).
    """
    empty_list: list[int] = []
    if n_test == 0:
        return cal_test_idx, empty_list
    if n_cal == 0:
        return empty_list, cal_test_idx

    cal_test_labels = labels[cal_test_idx]
    stratify_labels_test = cal_test_labels if _can_stratify(cal_test_labels, n_test) else None
    cal_idx, test_idx = train_test_split(
        cal_test_idx,
        test_size=n_test,
        stratify=stratify_labels_test,
        random_state=random_seed,
    )
    return cal_idx, test_idx


def _partition_stratified_subsets(
    indices: list[int],
    labels: np.ndarray,
    val_split: float,
    cal_split: float,
    test_split: float,
    random_seed: int,
) -> tuple[list[int], list[int], list[int], list[int]]:
    """Partition observation indices into stratified train, val, cal, and test subsets.

    Args:
        indices: List of index integers into the source dataset.
        labels: 1D NumPy array of ground truth class labels.
        val_split: Fraction allocated to validation.
        cal_split: Fraction allocated to conformal calibration.
        test_split: Fraction allocated to holdout testing.
        random_seed: PRNG seed for deterministic partitioning.

    Returns:
        Tuple of (train_idx, val_idx, cal_idx, test_idx).
    """
    n_total = len(indices)
    n_val = max(1, round(n_total * val_split)) if val_split > 0 else 0
    n_cal = max(1, round(n_total * cal_split)) if cal_split > 0 else 0
    n_test = max(1, round(n_total * test_split)) if test_split > 0 else 0
    n_holdout = n_val + n_cal + n_test

    if n_holdout <= 0 or n_holdout >= n_total:
        raise ValueError(f"Holdout sample count ({n_holdout}) must be strictly between 1 and {n_total - 1}.")

    stratify_labels_holdout = labels if _can_stratify(labels, n_holdout) else None
    train_idx, holdout_idx = train_test_split(
        indices,
        test_size=n_holdout,
        stratify=stratify_labels_holdout,
        random_state=random_seed,
    )

    empty_list: list[int] = []
    if n_cal == 0 and n_test == 0:
        return train_idx, holdout_idx, empty_list, empty_list

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
            random_state=random_seed,
        )

    cal_idx, test_idx = _split_cal_and_test(cal_test_idx, labels, n_cal, n_test, random_seed)
    return train_idx, val_idx, cal_idx, test_idx


def create_stratified_embedding_split(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    val_split: float = 0.10,
    cal_split: float = 0.0,
    test_split: float = 0.0,
    random_seed: int = 42,
) -> EmbeddingSplit:
    """Partition extracted representations into stratified training, validation, calibration, and test splits.

    Why:
        In severe long-tail biological distributions, unstratified random partitioning risks allocating zero validation instances for rare or endangered species, rendering validation PR-AUC metrics statistically undefined. Stratified splitting preserves class proportions across all subsets. Furthermore, Split Conformal Prediction requires an exchangeable calibration split that was not seen during model training.

    How:
        Asserts cardinality invariant (>= 2 classes). Executes multi-stage stratified splitting using scikit-learn `train_test_split` with `stratify=labels`, extracts corresponding tensor rows, calculates empirical per-class frequency vectors from training labels for Class-Balanced Loss, and returns an `EmbeddingSplit`.

    Args:
        embeddings: Full 2D tensor of representation vectors of shape `(N, D)`.
        labels: Full 1D tensor of ground-truth target labels of shape `(N,)`.
        num_classes: Total number of target classification categories.
        val_split: Proportion of data allocated to validation (default: 0.10).
        cal_split: Proportion of data allocated to conformal calibration (default: 0.0).
        test_split: Proportion of data allocated to holdout testing (default: 0.0).
        random_seed: PRNG seed ensuring reproducible split partitioning.

    Returns:
        Instantiated and validated EmbeddingSplit container.
    """
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
        val_split=val_split,
        cal_split=cal_split,
        test_split=test_split,
        random_seed=random_seed,
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


def run_feature_extraction(config: FeatureExtractionConfig) -> Path:
    """Execute the end-to-end representation extraction, caching, and serialization workflow.

    Why:
        Provides the primary pipeline entry point for offline feature caching. Automates device selection, manifest parsing, vision backbone factory instantiation, lazy batch extraction, stratified dataset partitioning, and atomic filesystem serialization into a single callable API.

    How:
        1. Resolves execution device (`cuda` if available and not overridden).
        2. Instantiates `S3ImageStreamer` over the designated Parquet manifest.
        3. Loads `taxa_catalog.yaml` to derive target class count and index map.
        4. Initializes frozen pre-trained feature extractor via `create_feature_extractor`.
        5. Streams images and extracts pooled feature representations.
        6. Constructs stratified `EmbeddingSplit`.
        7. Saves binary dictionary artifact containing backbone metadata and embeddings to disk.

    Args:
        config: FeatureExtractionConfig specifying execution parameters.

    Returns:
        Path to the serialized `.pt` embedding cache artifact on disk.

    Raises:
        FileNotFoundError: If the manifest path cannot be resolved.
        ValueError: If zero valid representations could be extracted.
    """
    target_device = (
        torch.device(config.device) if config.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    logger.info("Initializing feature extraction pipeline on device: %s", target_device)

    streamer = S3ImageStreamer(manifest_path=config.manifest_path)
    catalog = load_taxa_catalog()
    if not catalog:
        raise RuntimeError("Authoritative taxa catalog contains 0 taxa.")
    num_classes = len(catalog)
    taxon_to_index = build_taxon_index_map(catalog)

    logger.info(
        "Loaded manifest '%s' (%d records). Target perimeter encompasses %d taxa.",
        config.manifest_path,
        len(streamer.df),
        num_classes,
    )

    backbone = create_feature_extractor(config.extractor, pretrained=True).to(target_device)
    backbone.eval()

    transform = get_evaluation_transform(
        image_size=config.image_size,
        resize_size=config.resize_size,
    )

    embeddings, labels = stream_and_extract_representations(
        streamer=streamer,
        backbone=backbone,
        taxon_to_index=taxon_to_index,
        transform=transform,
        batch_size=config.batch_size,
        device=target_device,
        limit=config.limit,
    )

    manifest_total = min(len(streamer.df), config.limit) if config.limit is not None else len(streamer.df)
    extracted_total = len(embeddings)
    success_ratio = extracted_total / max(manifest_total, 1)

    settings = get_settings()
    min_success_ratio = settings.data.min_extraction_success_ratio
    if success_ratio < min_success_ratio and not settings.model.allow_untrained_fallback:
        logger.error(
            "Extraction completeness gate failed: extracted %d/%d (%.2f%%) < required %.2f%%",
            extracted_total,
            manifest_total,
            success_ratio * 100.0,
            min_success_ratio * 100.0,
        )
        raise RuntimeError(
            f"Extraction completeness gate failed: extracted {extracted_total}/{manifest_total} "
            f"({success_ratio:.2%}) which is below the minimum threshold of {min_success_ratio:.2%}."
        )

    logger.info(
        "Successfully extracted %d representations of dimension %d.",
        embeddings.shape[0],
        embeddings.shape[1],
    )

    split = create_stratified_embedding_split(
        embeddings=embeddings,
        labels=labels,
        num_classes=num_classes,
        val_split=config.val_split,
        cal_split=config.cal_split,
        test_split=config.test_split,
        random_seed=config.random_seed,
    )

    emb_dir = config.out_dir
    _save_split_artifacts(emb_dir, split, config.extractor, num_classes, output_file=config.output_path)
    return config.output_path if config.output_path is not None else (emb_dir / "train_embeddings.pt")


def _save_split_artifacts(
    emb_dir: Path,
    split: EmbeddingSplit,
    extractor: str,
    num_classes: int,
    output_file: Path | None = None,
) -> None:
    """Save composite embedding split and individual 4-way partition files to disk.

    Why:
        Encapsulates binary tensor serialization, preserving backward compatibility with
        single-file caches while materializing individual 4-way partitions for granular
        Directed Acyclic Graph (DAG) stage tracking in DVC.

    Args:
        emb_dir: Target directory for individual partition files.
        split: Instantiated EmbeddingSplit container.
        extractor: Name of the feature extractor backbone.
        num_classes: Total number of target classification categories.
        output_file: Optional legacy path for composite cached_features.pt file.
    """
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


def extract_features_cli(argv: list[str] | None = None) -> int:
    """Command-line interface entry point for representation extraction.

    Why:
        Enables seamless invocation from shell scripts, Makefile automation, Justfile recipes, and automated CI/CD runners without coupling to Python import hierarchies.

    How:
        Parses CLI flags using `argparse`, constructs a `FeatureExtractionConfig` instance, configures standard output logging, and triggers `run_feature_extraction`.

    Args:
        argv: Optional list of command-line arguments (defaults to `sys.argv[1:]`).

    Returns:
        Integer exit code (0 for success, non-zero for fatal execution failure).
    """
    settings = get_settings()
    parser = argparse.ArgumentParser(
        description="Stream biological images from S3 open data and cache dense backbone feature representations."
    )
    parser.add_argument(
        "--extractor",
        type=str,
        default=settings.model.default_extractor,
        help=f"Vision foundation backbone name (default: {settings.model.default_extractor})",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default="data/manifests/dataset_manifest.parquet",
        help="Path to Parquet dataset manifest",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Destination directory for discrete 4-way split artifacts (default: data/embeddings)",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Optional destination path for legacy composite EmbeddingSplit artifact",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=settings.training.batch_size,
        help=f"Mini-batch size for forward passes (default: {settings.training.batch_size})",
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=settings.training.val_split,
        help=f"Validation split ratio (default: {settings.training.val_split})",
    )
    parser.add_argument(
        "--cal-split",
        type=float,
        default=settings.training.cal_split,
        help=f"Conformal calibration split ratio (default: {settings.training.cal_split})",
    )
    parser.add_argument(
        "--test-split",
        type=float,
        default=settings.training.test_split,
        help=f"Holdout test split ratio (default: {settings.training.test_split})",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=settings.model.image_size,
        help=f"Spatial resolution for image resize and center crop (default: {settings.model.image_size})",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of records to process",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Execution device ('cuda', 'cpu', or auto-detect)",
    )

    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )

    out_dir_path = Path(args.out_dir) if args.out_dir else Path("data/embeddings")
    output_path = Path(args.out) if args.out else None
    if output_path is not None and args.out_dir is None:
        out_dir_path = output_path.parent

    cfg = FeatureExtractionConfig(
        extractor=args.extractor,
        manifest_path=Path(args.manifest),
        output_path=output_path,
        out_dir=out_dir_path,
        batch_size=args.batch_size,
        val_split=args.val_split,
        cal_split=args.cal_split,
        test_split=args.test_split,
        image_size=args.image_size,
        limit=args.limit,
        device=args.device,
    )

    try:
        run_feature_extraction(cfg)
        return 0
    except Exception as exc:
        logger.exception("Feature extraction pipeline failed: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(extract_features_cli())
