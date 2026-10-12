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

import torch
from sklearn.model_selection import train_test_split
from torchvision import transforms

from taxon_vision.data.s3_streamer import S3ImageStreamer
from taxon_vision.models.factory import create_feature_extractor, extract_features
from taxon_vision.models.training.types import EmbeddingSplit
from taxon_vision.service.inference import load_taxa_catalog

logger = logging.getLogger(__name__)

# Canonical mapping from authoritative iNaturalist / GBIF taxon IDs to catalog indices
DEFAULT_INAT_TAXON_MAP: dict[int, int] = {
    48662: 0,  # Danaus plexippus (Monarch Butterfly)
    47120: 1,  # Apis mellifera (Western Honey Bee)
    14704: 2,  # Turdus migratorius (American Robin)
    8229: 3,  # Cyanocitta cristata (Blue Jay)
    47602: 4,  # Taraxacum officinale (Common Dandelion)
    58737: 5,  # Helianthus annuus (Common Sunflower)
    42048: 6,  # Canis lupus (Gray Wolf)
    42054: 7,  # Vulpes vulpes (Red Fox)
    48715: 8,  # Amanita muscaria (Fly Agaric)
    48701: 9,  # Boletus edulis (King Bolete)
    47119: 1,  # Bombus (mapped to Hymenoptera/Bees cluster if present)
    47114: 1,  # Vespidae (mapped to Hymenoptera cluster if present)
}


@dataclass(frozen=True)
class FeatureExtractionConfig:
    """Immutable configuration container governing feature extraction and representation caching.

    Why:
        Decouples pipeline execution parameters from command-line parsing or ad-hoc function arguments.
        Providing a strongly-typed, frozen dataclass guarantees parameter immutability throughout the
        long-running extraction process, preventing accidental runtime mutation of paths or hyperparameters.

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

    extractor: str = "mobilenetv4_conv_small"
    manifest_path: Path = Path("data/manifests/dataset_manifest.parquet")
    output_path: Path = Path("data/embeddings/cached_features.pt")
    batch_size: int = 32
    val_split: float = 0.2
    device: str | None = None
    limit: int | None = None
    image_size: int = 224
    resize_size: int = 256
    random_seed: int = 42


def get_evaluation_transform(image_size: int = 224, resize_size: int = 256) -> transforms.Compose:
    """Construct deterministic image preprocessing transform for vision foundation model evaluation.

    Why:
        Linear probing on pre-trained vision representations requires rigorous adherence to the upstream
        evaluation preprocessing protocol. Stochastic data augmentations (e.g., random affine transformations,
        color jittering, or horizontal flips) are omitted during representation caching because foundation
        backbone features must represent the canonical visual signal. Pre-normalizing images against standard
        ImageNet channel statistics ensures that downstream linear heads receive correctly calibrated vectors.

    How:
        Composes a sequential torchvision transformation pipeline comprising:
        1. Bilinear aspect-ratio preserving resizing of the shorter edge to `resize_size`.
        2. Exact square center cropping to `(image_size, image_size)`.
        3. Conversion from uint8 PIL RGB image buffer to float32 tensor scaled to $[0.0, 1.0]$.
        4. Standard ImageNet channel-wise mean and variance normalization.

    Args:
        image_size: Spatial height and width of the output tensor passed to the neural network.
        resize_size: Interpolation target dimension for the shorter edge before center cropping.

    Returns:
        Callable torchvision Compose transformation pipeline.
    """
    return transforms.Compose(
        [
            transforms.Resize(resize_size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def build_taxon_index_map(
    catalog: list[dict[str, Any]] | None = None,
    include_defaults: bool = True,
) -> dict[int, int]:
    r"""Construct an authoritative mapping from external iNaturalist taxon IDs to dense zero-indexed labels.

    Why:
        Biological taxonomies in citizen-science repositories use arbitrary, non-contiguous integer IDs
        (e.g., Monarch Butterfly is `48662`, Western Honey Bee is `47120`). Deep learning classification
        heads and loss functions (such as PyTorch `CrossEntropyLoss` or `ClassBalancedLoss`) mathematically
        require target labels to be dense integers bounded by $0 \le y < K$, where $K$ is the class count.
        Creating a deterministic mapping guarantees alignment between observation metadata and head logits.

    How:
        Initializes baseline mappings from `DEFAULT_INAT_TAXON_MAP` when `include_defaults` is True.
        Inspects catalog metadata entries (checking `inaturalist_taxon_id`, `taxon_id`, and `id` keys),
        mapping each identified taxonomic integer to its ordinal position in the catalog.

    Args:
        catalog: Optional pre-loaded list of taxa metadata dictionaries.
        include_defaults: Whether to include default iNaturalist ID mappings.

    Returns:
        Dictionary mapping external iNaturalist integer taxon IDs to dense indices in $[0, K-1]$.
    """
    mapping: dict[int, int] = dict(DEFAULT_INAT_TAXON_MAP) if include_defaults else {}

    if catalog is not None:
        for idx, item in enumerate(catalog):
            for key in ("inaturalist_taxon_id", "taxon_id", "id"):
                if key in item and item[key] is not None:
                    mapping[int(item[key])] = idx
    else:
        try:
            loaded = load_taxa_catalog()
            for idx, item in enumerate(loaded):
                for key in ("inaturalist_taxon_id", "taxon_id", "id"):
                    if key in item and item[key] is not None:
                        mapping[int(item[key])] = idx
        except Exception:
            pass

    return mapping


def extract_batch_representations(
    backbone: torch.nn.Module,
    image_tensors: list[torch.Tensor],
    device: torch.device,
) -> torch.Tensor:
    """Execute a zero-gradient forward pass over an image batch to extract pooled representation vectors.

    Why:
        Encapsulates tensor batch collation and device dispatch into an isolated, unit-testable function.
        Enforces `torch.no_grad()` to prevent activation graph construction, drastically bounding VRAM
        consumption during high-throughput feature caching.

    How:
        Stacks individual 3D image tensors `(3, H, W)` into a contiguous 4D batch tensor `(B, 3, H, W)`,
        transfers the tensor to the designated accelerator device, invokes `extract_features`, and returns
        the detached representation matrix `(B, D)` residing in host CPU memory.

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
    streamer: S3ImageStreamer,
    backbone: torch.nn.Module,
    taxon_to_index: dict[int, int],
    transform: transforms.Compose,
    batch_size: int,
    device: torch.device,
    limit: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Lazily stream remote imagery, apply transformations, and extract representations in mini-batches.

    Why:
        Streaming biodiversity imagery lazily from open cloud object stores (e.g., AWS Open Data iNaturalist S3)
        avoids pre-allocating hundreds of gigabytes of disk storage on edge workstations. Grouping streamed
        images into mini-batches maximizes GPU Tensor Core saturation compared to single-image inference.

    How:
        Iterates over the `streamer.stream_images` lazy generator. Validates each observation's taxon identifier
        against `taxon_to_index`. Preprocesses valid images into tensors, accumulates mini-batches of size
        `batch_size`, and dispatches them to `extract_batch_representations`. Concatenates all batch tensors
        into unified feature matrix $X \in \mathbb{R}^{N \times D}$ and label vector $y \in \mathbb{Z}^N$.

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
    for obs_uuid, taxon_id, pil_img in streamer.stream_images(limit=limit):
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


def create_stratified_embedding_split(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    val_split: float = 0.2,
    random_seed: int = 42,
) -> EmbeddingSplit:
    """Partition extracted representations into stratified training and validation splits.

    Why:
        In severe long-tail biological distributions, unstratified random partitioning risks allocating
        zero validation instances for rare or endangered species, rendering validation PR-AUC metrics
        statistically undefined. Stratified splitting preserves class proportions across both subsets.

    How:
        Converts labels to NumPy format, executes scikit-learn `train_test_split` with `stratify=labels`,
        extracts corresponding tensor rows, calculates empirical per-class frequency vectors for
        effective sample weighting in Class-Balanced Loss, and validates the `EmbeddingSplit` invariant.

    Args:
        embeddings: Full 2D tensor of representation vectors of shape `(N, D)`.
        labels: Full 1D tensor of ground-truth target labels of shape `(N,)`.
        num_classes: Total number of target classification categories.
        val_split: Proportion of data allocated to the validation split (default: 0.2).
        random_seed: PRNG seed ensuring reproducible split partitioning.

    Returns:
        Instantiated and validated EmbeddingSplit container.
    """
    total_samples = embeddings.shape[0]
    indices = list(range(total_samples))

    train_idx, val_idx = train_test_split(
        indices,
        test_size=val_split,
        stratify=labels.numpy(),
        random_state=random_seed,
    )

    train_z, train_y = embeddings[train_idx], labels[train_idx]
    val_z, val_y = embeddings[val_idx], labels[val_idx]

    samples_per_class = [max(1, int((train_y == c).sum().item())) for c in range(num_classes)]

    return EmbeddingSplit(
        train_embeddings=train_z,
        train_labels=train_y,
        val_embeddings=val_z,
        val_labels=val_y,
        samples_per_class=samples_per_class,
    )


def run_feature_extraction(config: FeatureExtractionConfig) -> Path:
    """Execute the end-to-end representation extraction, caching, and serialization workflow.

    Why:
        Provides the primary pipeline entry point for offline feature caching. Automates device
        selection, manifest parsing, vision backbone factory instantiation, lazy batch extraction,
        stratified dataset partitioning, and atomic filesystem serialization into a single callable API.

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
    num_classes = len(catalog) if catalog else 10
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
        random_seed=config.random_seed,
    )

    output_file = Path(config.output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "extractor": config.extractor,
        "feature_dim": split.feature_dim,
        "num_classes": num_classes,
        "split": split,
    }
    torch.save(payload, output_file)
    logger.info("Serialized feature embedding split artifact to: %s", output_file)
    return output_file


def extract_features_cli(argv: list[str] | None = None) -> int:
    """Command-line interface entry point for representation extraction.

    Why:
        Enables seamless invocation from shell scripts, Makefile automation, Justfile recipes,
        and automated CI/CD runners without coupling to Python import hierarchies.

    How:
        Parses CLI flags using `argparse`, constructs a `FeatureExtractionConfig` instance,
        configures standard output logging, and triggers `run_feature_extraction`.

    Args:
        argv: Optional list of command-line arguments (defaults to `sys.argv[1:]`).

    Returns:
        Integer exit code (0 for success, non-zero for fatal execution failure).
    """
    parser = argparse.ArgumentParser(
        description="Stream biological images from S3 open data and cache dense backbone feature representations."
    )
    parser.add_argument(
        "--extractor",
        type=str,
        default="mobilenetv4_conv_small",
        help="Vision foundation backbone name (default: mobilenetv4_conv_small)",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default="data/manifests/dataset_manifest.parquet",
        help="Path to Parquet dataset manifest",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="data/embeddings/cached_features.pt",
        help="Destination path for serialized EmbeddingSplit artifact",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Mini-batch size for forward passes (default: 32)",
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.2,
        help="Validation split ratio (default: 0.2)",
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

    cfg = FeatureExtractionConfig(
        extractor=args.extractor,
        manifest_path=Path(args.manifest),
        output_path=Path(args.out),
        batch_size=args.batch_size,
        val_split=args.val_split,
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
