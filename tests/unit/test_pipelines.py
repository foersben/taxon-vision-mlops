# SPDX-FileCopyrightText: 2026 Benjamin Förster
# SPDX-License-Identifier: MIT
"""Unit tests for pipeline components (feature extraction and training entry points)."""

from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest
import torch
from PIL import Image

from taxon_vision.pipelines.extract_features import (
    DEFAULT_INAT_TAXON_MAP,
    FeatureExtractionConfig,
    build_taxon_index_map,
    create_stratified_embedding_split,
    extract_batch_representations,
    extract_features_cli,
    get_evaluation_transform,
)
from taxon_vision.pipelines.train import main as train_pipeline_main


def test_feature_extraction_config_immutability() -> None:
    """Ensure FeatureExtractionConfig is an immutable dataclass."""
    cfg = FeatureExtractionConfig(extractor="mobilenetv4_conv_small", batch_size=16)
    assert cfg.extractor == "mobilenetv4_conv_small"
    assert cfg.batch_size == 16
    with pytest.raises(FrozenInstanceError):
        # Frozen dataclass should reject attribute mutations
        cfg.__setattr__("batch_size", 32)


def test_get_evaluation_transform() -> None:
    """Verify evaluation transform outputs standard 3x224x224 normalized float tensors."""
    transform = get_evaluation_transform(image_size=224, resize_size=256)
    dummy_img = Image.new("RGB", (300, 400), color=(128, 128, 128))
    tensor = transform(dummy_img)

    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (3, 224, 224)
    assert tensor.dtype == torch.float32


def test_build_taxon_index_map_catalog_mapping() -> None:
    """Verify taxon index map correctly reflects catalog IDs and default fallback."""
    catalog = [{"id": 1, "scientific_name": "Danaus plexippus"}]
    mapping = build_taxon_index_map(catalog, include_defaults=False)
    assert mapping == {1: 0}

    # When include_defaults=True, canonical iNat IDs are included
    default_mapping = build_taxon_index_map(catalog, include_defaults=True)
    assert 48662 in default_mapping
    assert default_mapping[1] == 0

    # None catalog loads default catalog and defaults
    empty_mapping = build_taxon_index_map([], include_defaults=True)
    assert empty_mapping == DEFAULT_INAT_TAXON_MAP


def test_create_stratified_embedding_split() -> None:
    """Verify stratified splitting produces correct train/validation shapes and class counts."""
    torch.manual_seed(42)
    num_samples = 40
    dim = 16
    num_classes = 4

    embeddings = torch.randn(num_samples, dim)
    labels = torch.tensor([i % num_classes for i in range(num_samples)])

    split = create_stratified_embedding_split(
        embeddings=embeddings,
        labels=labels,
        num_classes=num_classes,
        val_split=0.25,
        random_seed=42,
    )

    assert split.feature_dim == dim
    assert split.train_embeddings.shape == (30, dim)
    assert split.val_embeddings.shape == (10, dim)
    assert split.train_labels.shape == (30,)
    assert split.val_labels.shape == (10,)
    assert split.samples_per_class is not None
    assert len(split.samples_per_class) == num_classes
    assert sum(split.samples_per_class) == 30


def test_extract_batch_representations_mock_backbone() -> None:
    """Verify extract_batch_representations executes forward pass and returns 2D tensor on CPU."""

    class DummyBackbone(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.ones((x.shape[0], 64))

    backbone = DummyBackbone()
    image_tensors = [torch.zeros((3, 224, 224)), torch.zeros((3, 224, 224))]
    out = extract_batch_representations(backbone, image_tensors, device=torch.device("cpu"))

    assert out.shape == (2, 64)
    assert out.device.type == "cpu"


def test_train_pipeline_main_wrapper() -> None:
    """Verify train pipeline main entrypoint dispatches to train_cli cleanly."""
    exit_code = train_pipeline_main(["--epochs", "1", "--batch-size", "8"])
    assert exit_code == 0


def test_extract_features_cli_dry_run(tmp_path: Path) -> None:
    """Verify extract_features_cli fails gracefully on non-existent manifest without crashing unhandled."""
    fake_manifest = tmp_path / "non_existent.parquet"
    out_file = tmp_path / "features.pt"

    ret = extract_features_cli(["--manifest", str(fake_manifest), "--out", str(out_file)])
    assert ret == 1
