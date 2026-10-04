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
    """Load the supported taxa catalog from YAML.

    Returns:
        List of taxa with scientific and common names.
    """
    catalog_path = Path(__file__).resolve().parent.parent.parent.parent / "config" / "taxa_catalog.yaml"
    with open(catalog_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data.get("taxa", [])


@lru_cache(maxsize=1)
def get_classifier() -> TaxonClassifier:
    """Instantiate and cache the vision classifier model.

    Returns:
        The vision classifier model.
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
    """Instantiate the conformal prediction engine.

    Returns:
        The conformal prediction engine.
    """
    return ConformalPredictionEngine()


def _get_image_tensor(image_bytes: bytes) -> torch.Tensor:
    """Convert image bytes to tensor.

    Args:
        image_bytes: Image bytes.

    Returns:
        Tensor with image data.
    """
    cfg = get_settings().model
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    transform = T.Compose(
        [T.Resize((cfg.image_size, cfg.image_size)), T.ToTensor(), T.Normalize(mean=cfg.mean, std=cfg.std)]
    )
    tensor = transform(image).unsqueeze(0)  # Shape: (1, 3, H, W)
    return tensor


def run_prediction(image_bytes: bytes) -> dict[str, Any]:
    """Execute the full vision inference pipeline with conformal uncertainty.

    Args:
        image_bytes: Image bytes.

    Returns:
        Dictionary with prediction results.
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
    top1_prob = probabilities[top1_idx]

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
