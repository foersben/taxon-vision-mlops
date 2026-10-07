# SPDX-FileCopyrightText: 2026 Benjamin Förster
# SPDX-License-Identifier: MIT
"""Inference service module encapsulating model loading, prediction, and uncertainty."""

import io
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchvision.transforms as T
import yaml
from PIL import Image

from taxon_vision.config import get_settings
from taxon_vision.models.factory import create_feature_extractor, get_feature_dimension
from taxon_vision.models.head import TaxonClassifier
from taxon_vision.uncertainty.conformal import ConformalPredictionEngine


@lru_cache(maxsize=1)
def load_taxa_catalog() -> list[dict[str, Any]]:
    """Load and cache the supported biological taxa catalog from disk.

    Why:
        Taxonomic classification targets are defined in an authoritative metadata catalog
        (`config/taxa_catalog.yaml`). Reading and parsing YAML from disk on every prediction
        incurs file I/O latency and memory fragmentation. Caching the parsed list in memory
        ensures single-digit microsecond lookups during inference mapping.

    How:
        Resolves the absolute path to `config/taxa_catalog.yaml`, deserializes it via PyYAML
        safe loader, and caches the resulting list of taxa metadata dictionaries.

    Returns:
        List of taxa metadata dictionaries with scientific and common names.
    """
    catalog_path = Path(__file__).resolve().parent.parent.parent.parent / "config" / "taxa_catalog.yaml"
    with open(catalog_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data.get("taxa", [])


@lru_cache(maxsize=1)
def get_classifier() -> TaxonClassifier:
    """Instantiate, restore checkpoint weights, and cache the vision classifier model.

    Why:
        Deep neural network models (e.g. MobileNetV4, DINOv2) require tens or hundreds of
        megabytes of memory and non-trivial weight initialization time. Instantiating a new
        model per request would exhaust process memory and cause hundred-millisecond cold starts.
        Caches a single eval-mode TaxonClassifier instance in process memory.

    How:
        Loads backbone architecture configured in project settings, retrieves feature projection
        dimension, instantiates a TaxonClassifier with the corresponding linear classification head,
        restores checkpoint weights from disk if available, and locks the model in evaluation mode (`eval()`).

    Returns:
        Cached TaxonClassifier ready for zero-gradient evaluation.
    """
    settings = get_settings()
    backbone = create_feature_extractor(settings.model.default_extractor, pretrained=False)
    dim = get_feature_dimension(backbone)
    num_classes = len(load_taxa_catalog())

    classifier = TaxonClassifier(backbone=backbone, feature_dim=dim, num_classes=num_classes)

    checkpoint_path = Path(__file__).resolve().parent.parent.parent.parent / settings.model.head_checkpoint_path

    if checkpoint_path.exists():
        classifier.head.load_state_dict(torch.load(checkpoint_path, weights_only=True))

    classifier.eval()
    return classifier


@lru_cache(maxsize=1)
def get_conformal_engine() -> ConformalPredictionEngine:
    """Instantiate and cache the conformal prediction engine.

    Why:
        The conformal prediction engine contains pre-calibrated error tolerances (alpha) and
        quantile non-conformity cutoffs (q_hat). Caching a singleton engine ensures consistent
        calibration parameters across concurrent requests without per-request instantiation overhead.

    How:
        Initializes a ConformalPredictionEngine with default calibrated thresholds and returns
        the cached instance.

    Returns:
        The cached ConformalPredictionEngine singleton.
    """
    return ConformalPredictionEngine()


@lru_cache(maxsize=1)
def get_image_transform() -> T.Compose:
    """Construct and cache the deterministic image preprocessing transform pipeline.

    Why:
        In high-throughput biological inference, re-instantiating torchvision transform
        pipelines on every request introduces significant memory allocation overhead,
        triggering garbage collection cycles on the hot inference path. Pre-instantiating
        and caching an immutable composition preserves the zero-allocation hot-path
        invariant required for sub-25ms p95 latency SLAs.

    How:
        Retrieves global model configurations (target spatial resolution, per-channel normalization
        mean and standard deviation) and bundles Resize, ToTensor, and Normalize operations into
        a single cached torchvision.transforms.Compose instance.

    Returns:
        Cached Compose pipeline transforming PIL Images into normalized 3xHxW float tensors.
    """
    cfg = get_settings().model
    return T.Compose(
        [
            T.Resize((cfg.image_size, cfg.image_size)),
            T.ToTensor(),
            T.Normalize(mean=cfg.mean, std=cfg.std),
        ]
    )


def _get_image_tensor(image_bytes: bytes) -> torch.Tensor:
    """Convert raw image bytes into a normalized model input tensor.

    Why:
        Incoming HTTP multipart payloads provide raw binary image buffers. These must be
        decoded, color-space verified, resized, normalized, and batched into a standard
        4D tensor (1, C, H, W) before executing backbone feature extraction.

    How:
        Wraps binary bytes in an in-memory BytesIO stream, decodes as RGB PIL Image, applies
        the pre-cached torchvision transform pipeline, and prepends a batch dimension via `unsqueeze(0)`.

    Args:
        image_bytes: Raw binary bytes of the input image.

    Returns:
        Torch tensor of shape (1, 3, H, W) ready for backbone consumption.
    """
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    transform = get_image_transform()
    return transform(image).unsqueeze(0)


def run_prediction(image_bytes: bytes) -> dict[str, Any]:
    """Execute the full vision inference pipeline with conformal uncertainty quantification.

    Why:
        Provides the end-to-end classification pipeline for an organism observation, mapping
        from raw image bytes to calibrated taxonomic predictions and conformal guarantee sets.
        Enforces epistemic safety by evaluating set cardinality and determining whether human
        taxonomist triage is mandated.

    How:
        1. Preprocesses image bytes into normalized 4D float tensor.
        2. Evaluates model logits inside a `torch.no_grad()` block.
        3. Computes softmax probabilities over target classes.
        4. Identifies top-1 argmax prediction and confidence score.
        5. Evaluates non-conformity scores via ConformalPredictionEngine to construct
           valid prediction sets and identify ambiguous or unclassifiable cases.
        6. Maps class indices back to authoritative taxa catalog metadata.

    Complexity:
        O(1) forward pass over the backbone plus O(K) softmax and conformal set evaluation,
        where K is the number of taxa in the catalog.

    Args:
        image_bytes: Raw binary bytes of the input organism image.

    Returns:
        Dictionary containing scientific name, common name, top-1 confidence,
        conformal set candidate names, human referral flag, and execution latency.
    """
    start_time = time.perf_counter()

    taxa_catalog = load_taxa_catalog()

    tensor = _get_image_tensor(image_bytes)

    classifier = get_classifier()
    with torch.no_grad():
        logits = classifier(tensor)  # Shape: (1, num_classes)
        probabilities = torch.softmax(logits, dim=1).numpy()[0]

    # Extract top prediction
    top1_idx = int(np.argmax(probabilities))
    top1_prob = float(probabilities[top1_idx])

    # Uncertainty quantification
    conformal_engine = get_conformal_engine()
    pred_set_indices, is_uncertain = conformal_engine.predict_set(probabilities)

    # Map to taxa names
    predicted_taxon = taxa_catalog[top1_idx]
    conformal_set_names = [taxa_catalog[i]["scientific_name"] for i in pred_set_indices]

    elapsed_time_ms = (time.perf_counter() - start_time) * 1000.0

    return {
        "scientific_name": predicted_taxon["scientific_name"],
        "common_name": predicted_taxon["common_name"],
        "confidence": top1_prob,
        "conformal_set": conformal_set_names,
        "requires_human_review": is_uncertain,
        "latency_ms": elapsed_time_ms,
    }
