# SPDX-FileCopyrightText: 2026 Benjamin Förster
# SPDX-License-Identifier: MIT
"""Stage 3: Vision foundation feature extraction and offline embedding caching pipeline."""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple

import torch
from torchvision import transforms

from taxon_vision.config import get_settings
from taxon_vision.data.diamond_split import (
    DEFAULT_INAT_TAXON_MAP,
    SplitAllocationSpec,
    build_taxon_index_map,
    get_catalog_inat_taxon_map,
    partition_epistemic_diamond,
    save_split_artifacts,
)
from taxon_vision.data.s3_streamer import S3ImageStreamer
from taxon_vision.models.factory import create_feature_extractor, extract_features
from taxon_vision.models.training.types import EmbeddingSplit
from taxon_vision.service.inference import load_taxa_catalog

logger = logging.getLogger(__name__)

_save_split_artifacts = save_split_artifacts


class ExtractedRepresentations(NamedTuple):
    """Concatenated feature embedding matrix and ground-truth integer labels."""

    embeddings: torch.Tensor
    labels: torch.Tensor


@dataclass(frozen=True)
class FeatureExtractionConfig:
    """Immutable configuration container governing feature extraction and representation caching.

    Why:
        Decouples pipeline execution parameters from command-line parsing or ad-hoc function arguments. Providing a strongly-typed, frozen dataclass guarantees parameter immutability throughout the long-running extraction process, preventing accidental runtime mutation of paths or hyperparameters.

    Attributes:
        extractor: Name of the vision foundation backbone (e.g. `mobilenetv4_conv_small`, `dinov3_vits14`).
        manifest_path: Filesystem path to the Parquet dataset manifest containing observation URLs.
        output_path: Filesystem destination path where the serialized `EmbeddingSplit` dictionary is saved.
        out_dir: Destination directory for discrete 4-way Epistemic Diamond `.pt` splits.
        batch_size: Number of images grouped into a single forward tensor pass through the backbone.
        val_split: Fraction of extracted representations reserved for the holdout validation split (0.0 to 1.0).
        cal_split: Fraction reserved for Split Conformal calibration.
        test_split: Fraction reserved for unbiased holdout evaluation.
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
    """Construct deterministic image preprocessing transform for vision foundation model evaluation."""
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


def extract_batch_representations(
    backbone: torch.nn.Module,
    image_tensors: list[torch.Tensor],
    device: torch.device,
) -> torch.Tensor:
    """Execute a zero-gradient forward pass over an image batch to extract pooled representation vectors."""
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
) -> ExtractedRepresentations:
    r"""Lazily stream remote imagery, apply transformations, and extract representations in mini-batches."""
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
    return ExtractedRepresentations(embeddings=all_x, labels=all_y)


def create_stratified_embedding_split(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    val_split: float = 0.10,
    cal_split: float = 0.0,
    test_split: float = 0.0,
    random_seed: int = 42,
) -> EmbeddingSplit:
    """Partition extracted representations into stratified training, validation, calibration, and test splits."""
    return partition_epistemic_diamond(
        embeddings=embeddings,
        labels=labels,
        num_classes=num_classes,
        spec=SplitAllocationSpec(
            val_split=val_split,
            cal_split=cal_split,
            test_split=test_split,
            random_seed=random_seed,
        ),
    )


def run_feature_extraction(config: FeatureExtractionConfig) -> Path:
    """Execute the end-to-end representation extraction, caching, and serialisation workflow."""
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
    min_success_ratio = settings.streaming.min_extraction_success_ratio
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
    save_split_artifacts(emb_dir, split, config.extractor, num_classes, output_file=config.output_path)
    return config.output_path if config.output_path is not None else (emb_dir / "train_embeddings.pt")


def extract_features_cli(argv: list[str] | None = None) -> int:
    """Command-line interface entry point for representation extraction."""
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
        force=True,
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


__all__ = [
    "DEFAULT_INAT_TAXON_MAP",
    "ExtractedRepresentations",
    "FeatureExtractionConfig",
    "build_taxon_index_map",
    "create_stratified_embedding_split",
    "extract_batch_representations",
    "extract_features_cli",
    "get_catalog_inat_taxon_map",
    "get_evaluation_transform",
    "run_feature_extraction",
    "stream_and_extract_representations",
]

if __name__ == "__main__":
    sys.exit(extract_features_cli())
