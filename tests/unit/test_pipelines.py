# SPDX-FileCopyrightText: 2026 Benjamin Förster
# SPDX-License-Identifier: MIT
"""Unit tests for pipeline components (feature extraction and training entry points)."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any

import polars as pl
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
    run_feature_extraction,
    stream_and_extract_representations,
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


def test_stream_and_extract_representations() -> None:
    """Verify stream_and_extract_representations processes batches and skips unmapped taxa."""

    class DummyStreamer:
        def stream_images(self, **_kwargs: Any) -> Any:
            # Yield 3 images: 2 mapped to taxon 1, 1 unmapped taxon 999
            img = Image.new("RGB", (32, 32), color="red")
            yield "uuid-1", 1, img
            yield "uuid-2", 999, img
            yield "uuid-3", 1, img

    class DummyBackbone(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.full((x.shape[0], 16), 1.0)

    def dummy_transform(img: Image.Image) -> torch.Tensor:
        return torch.zeros((3, 224, 224))

    backbone = DummyBackbone()
    taxon_to_index = {1: 0}

    embs, labels = stream_and_extract_representations(
        streamer=DummyStreamer(),  # type: ignore[arg-type]
        backbone=backbone,
        taxon_to_index=taxon_to_index,
        transform=dummy_transform,
        batch_size=2,
        device=torch.device("cpu"),
    )

    assert embs.shape == (2, 16)
    assert labels.shape == (2,)
    assert labels.tolist() == [0, 0]


def test_stream_and_extract_representations_empty_raises() -> None:
    """Verify ValueError is raised if extraction yields zero compliant representations."""

    class EmptyStreamer:
        def stream_images(self, **_kwargs: Any) -> Any:
            return iter([])

    class DummyBackbone(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x

    def dummy_transform(img: Image.Image) -> torch.Tensor:
        return torch.zeros((3, 224, 224))

    with pytest.raises(ValueError, match="Feature extraction yielded zero samples"):
        stream_and_extract_representations(
            streamer=EmptyStreamer(),  # type: ignore[arg-type]
            backbone=DummyBackbone(),
            taxon_to_index={1: 0},
            transform=dummy_transform,
            batch_size=2,
            device=torch.device("cpu"),
        )


def test_run_feature_extraction_pipeline_and_cli(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Verify run_feature_extraction runs end-to-end and serializes artifact."""
    manifest_file = tmp_path / "valid_manifest.parquet"
    df = pl.DataFrame(
        {
            "observation_uuid": [f"uuid-{i}" for i in range(8)],
            "taxon_id": [1, 1, 1, 1, 2, 2, 2, 2],
            "image_url": [f"https://mock.s3/img_{i}.jpg" for i in range(8)],
        }
    )
    df.write_parquet(manifest_file)
    out_file = tmp_path / "cached_features.pt"

    class DummyBackbone(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.randn(x.shape[0], 32)

    class MockStreamer:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            self.df = df

        def stream_images(self, **_kwargs: Any) -> Any:
            for i in range(8):
                tid = 1 if i < 4 else 2
                yield f"uuid-{i}", tid, Image.new("RGB", (32, 32))

    monkeypatch.setattr("taxon_vision.pipelines.extract_features.S3ImageStreamer", MockStreamer)
    monkeypatch.setattr(
        "taxon_vision.pipelines.extract_features.create_feature_extractor",
        lambda *_args, **_kwargs: DummyBackbone(),
    )
    monkeypatch.setattr(
        "taxon_vision.pipelines.extract_features.load_taxa_catalog",
        lambda: [{"id": 1, "scientific_name": "Taxon A"}, {"id": 2, "scientific_name": "Taxon B"}],
    )

    cfg = FeatureExtractionConfig(
        extractor="mobilenetv4_conv_small",
        manifest_path=manifest_file,
        output_path=out_file,
        batch_size=4,
        val_split=0.25,
        device="cpu",
    )

    saved_path = run_feature_extraction(cfg)
    assert saved_path.exists()
    payload = torch.load(saved_path, weights_only=False)
    assert "split" in payload
    assert payload["feature_dim"] == 32

    # CLI successful invocation
    ret = extract_features_cli(
        ["--manifest", str(manifest_file), "--out", str(tmp_path / "cli_features.pt"), "--device", "cpu"]
    )
    assert ret == 0
