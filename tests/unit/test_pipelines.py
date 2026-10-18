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


def test_create_stratified_embedding_split_4way() -> None:
    """Verify 4-way stratified splitting partitions data into train, val, cal, and test sets."""
    torch.manual_seed(42)
    num_samples = 100
    dim = 16
    num_classes = 5

    embeddings = torch.randn(num_samples, dim)
    labels = torch.tensor([i % num_classes for i in range(num_samples)])

    split = create_stratified_embedding_split(
        embeddings=embeddings,
        labels=labels,
        num_classes=num_classes,
        val_split=0.10,
        cal_split=0.10,
        test_split=0.10,
        random_seed=42,
    )

    assert split.feature_dim == dim
    assert split.train_embeddings.shape == (70, dim)
    assert split.val_embeddings.shape == (10, dim)
    assert split.cal_embeddings is not None and split.cal_embeddings.shape == (10, dim)
    assert split.test_embeddings is not None and split.test_embeddings.shape == (10, dim)
    assert split.train_labels.shape == (70,)
    assert split.val_labels.shape == (10,)
    assert split.cal_labels is not None and split.cal_labels.shape == (10,)
    assert split.test_labels is not None and split.test_labels.shape == (10,)
    assert split.samples_per_class is not None
    assert sum(split.samples_per_class) == 70


def test_create_stratified_embedding_split_cardinality_guard() -> None:
    """Verify cardinality guard raises RuntimeError on single-class degenerate dataset (E2)."""
    embeddings = torch.randn(30, 8)
    labels = torch.zeros(30, dtype=torch.long)  # All class 0

    with pytest.raises(RuntimeError, match="Cardinality collapse detected"):
        create_stratified_embedding_split(
            embeddings=embeddings,
            labels=labels,
            num_classes=5,
            val_split=0.2,
        )


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
        streamer=DummyStreamer(),
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
            streamer=EmptyStreamer(),
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

    def _mock_create_extractor(*_args: Any, **_kwargs: Any) -> DummyBackbone:
        return DummyBackbone()

    monkeypatch.setattr("taxon_vision.pipelines.extract_features.create_feature_extractor", _mock_create_extractor)
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


def test_extract_features_completeness_gate_raises(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Verify run_feature_extraction raises RuntimeError when completeness falls below threshold and untrained fallback is False.

    Why:
        Guarantees that silent dataset truncation under network error conditions is intercepted before downstream linear probing.

    How:
        Configures an 8-sample manifest where the mock streamer only yields 4 samples (50% completeness < 95% threshold). Disables TAXON_ALLOW_UNTRAINED_FALLBACK and asserts RuntimeError is raised.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
        tmp_path: Pytest temporary directory fixture.
    """
    from taxon_vision.config import get_settings

    manifest_file = tmp_path / "manifest.parquet"
    df = pl.DataFrame(
        {
            "observation_uuid": [f"uuid-{i}" for i in range(8)],
            "taxon_id": [1, 1, 1, 1, 2, 2, 2, 2],
            "image_url": [f"https://mock.s3/img_{i}.jpg" for i in range(8)],
        }
    )
    df.write_parquet(manifest_file)
    out_file = tmp_path / "truncated_features.pt"

    class IncompleteStreamer:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            self.df = df

        def stream_images(self, **_kwargs: Any) -> Any:
            # Yield only 4 out of 8 images (50% < 95%)
            for i in range(4):
                yield f"uuid-{i}", 1, Image.new("RGB", (32, 32))

    monkeypatch.setattr("taxon_vision.pipelines.extract_features.S3ImageStreamer", IncompleteStreamer)

    def _mock_extractor(*_args: Any, **_kwargs: Any) -> torch.nn.Module:
        return torch.nn.Linear(3, 16)

    def _mock_catalog() -> list[dict[str, Any]]:
        return [{"id": 1, "scientific_name": "Taxon A"}, {"id": 2, "scientific_name": "Taxon B"}]

    def _mock_stream_extract(**_kwargs: Any) -> tuple[torch.Tensor, torch.Tensor]:
        return (torch.zeros((4, 16)), torch.zeros(4, dtype=torch.long))

    monkeypatch.setattr("taxon_vision.pipelines.extract_features.create_feature_extractor", _mock_extractor)
    monkeypatch.setattr("taxon_vision.pipelines.extract_features.load_taxa_catalog", _mock_catalog)
    monkeypatch.setattr(
        "taxon_vision.pipelines.extract_features.stream_and_extract_representations",
        _mock_stream_extract,
    )

    monkeypatch.setenv("TAXON_ALLOW_UNTRAINED_FALLBACK", "false")
    get_settings.cache_clear()

    cfg = FeatureExtractionConfig(
        extractor="mobilenetv4_conv_small",
        manifest_path=manifest_file,
        output_path=out_file,
        batch_size=4,
        device="cpu",
    )

    with pytest.raises(RuntimeError, match="Extraction completeness gate failed"):
        run_feature_extraction(cfg)

    get_settings.cache_clear()


def test_embedding_split_directory_loading(tmp_path: Path) -> None:
    """Verify EmbeddingSplit loads correctly from a directory of discrete split files."""
    from taxon_vision.models.training.types import EmbeddingSplit

    split_dir = tmp_path / "splits"
    split_dir.mkdir()

    torch.save(
        {"embeddings": torch.randn(10, 16), "labels": torch.zeros(10, dtype=torch.long), "samples_per_class": [10]},
        split_dir / "train_embeddings.pt",
    )
    torch.save(
        {"embeddings": torch.randn(4, 16), "labels": torch.zeros(4, dtype=torch.long)},
        split_dir / "val_embeddings.pt",
    )
    torch.save(
        {"embeddings": torch.randn(3, 16), "labels": torch.zeros(3, dtype=torch.long)},
        split_dir / "cal_embeddings.pt",
    )
    torch.save(
        {"embeddings": torch.randn(2, 16), "labels": torch.zeros(2, dtype=torch.long)},
        split_dir / "test_embeddings.pt",
    )

    split = EmbeddingSplit.from_cache(split_dir)
    assert split.train_embeddings.shape == (10, 16)
    assert split.val_embeddings.shape == (4, 16)
    assert split.cal_embeddings is not None and split.cal_embeddings.shape == (3, 16)
    assert split.test_embeddings is not None and split.test_embeddings.shape == (2, 16)
    assert split.samples_per_class == [10]
    assert split.feature_dim == 16


def test_save_split_artifacts(tmp_path: Path) -> None:
    """Verify _save_split_artifacts saves both composite and discrete files."""
    from taxon_vision.models.training.types import EmbeddingSplit
    from taxon_vision.pipelines.extract_features import _save_split_artifacts

    output_pt = tmp_path / "embeddings" / "cached_features.pt"
    split = EmbeddingSplit(
        train_embeddings=torch.randn(12, 8),
        train_labels=torch.zeros(12, dtype=torch.long),
        val_embeddings=torch.randn(4, 8),
        val_labels=torch.zeros(4, dtype=torch.long),
        cal_embeddings=torch.randn(3, 8),
        cal_labels=torch.zeros(3, dtype=torch.long),
        test_embeddings=torch.randn(2, 8),
        test_labels=torch.zeros(2, dtype=torch.long),
        samples_per_class=[12],
    )

    _save_split_artifacts(output_pt, split, "mobilenetv4_conv_small", 2)

    assert output_pt.exists()
    parent = output_pt.parent
    assert (parent / "train_embeddings.pt").exists()
    assert (parent / "val_embeddings.pt").exists()
    assert (parent / "cal_embeddings.pt").exists()
    assert (parent / "test_embeddings.pt").exists()

    loaded = EmbeddingSplit.from_cache(output_pt)
    assert loaded.feature_dim == 8


def test_verify_observation_compliance_valid_and_invalid() -> None:
    """Verify license compliance logic handles valid open licenses and detects violations."""
    from taxon_vision.pipelines.audit_license import verify_observation_compliance

    valid = {
        "observation_id": 1,
        "license_code": "CC-BY",
        "image_url": "https://inaturalist-open-data.s3.amazonaws.com/photos/1/medium.jpg",
        "attribution": "(c) Jane Doe, some rights reserved (CC-BY)",
    }
    assert verify_observation_compliance(valid) == []

    invalid = {
        "observation_id": 2,
        "license_code": "All-Rights-Reserved",
        "image_url": "https://static.inaturalist.org/photos/2/medium.jpg",
        "attribution": "",
        "is_open_dataset_verified": False,
    }
    errors = verify_observation_compliance(invalid)
    assert len(errors) == 3


def test_audit_license_pipeline_gatekeeper_mode(tmp_path: Path) -> None:
    """Verify Stage 2 audit_license pipeline in gatekeeper mode."""
    from taxon_vision.pipelines.audit_license import audit_license_cli

    in_parquet = tmp_path / "raw_manifest.parquet"
    out_parquet = tmp_path / "dataset_manifest.parquet"
    out_json = tmp_path / "license_audit.json"

    df = pl.DataFrame(
        [
            {
                "observation_uuid": "obs-1",
                "taxon_id": 47120,
                "image_url": "https://inaturalist-open-data.s3.amazonaws.com/1.jpg",
                "attribution": "(c) John Doe",
                "license_code": "CC0",
                "license_url": "https://creativecommons.org/publicdomain/zero/1.0/",
                "is_open_dataset_verified": True,
            },
            {
                "observation_uuid": "obs-2",
                "taxon_id": 47120,
                "image_url": "https://example.com/2.jpg",
                "attribution": "",
                "license_code": "COPYRIGHT",
                "license_url": "",
                "is_open_dataset_verified": False,
            },
        ]
    )
    df.write_parquet(in_parquet)

    exit_code = audit_license_cli(
        [
            "--manifest",
            str(in_parquet),
            "--out-manifest",
            str(out_parquet),
            "--out",
            str(out_json),
        ]
    )
    assert exit_code == 0
    assert out_parquet.exists()
    assert out_json.exists()

    filtered_df = pl.read_parquet(out_parquet)
    assert len(filtered_df) == 1
    assert filtered_df["observation_uuid"][0] == "obs-1"

    # Test audit mode (without --out-manifest)
    exit_audit = audit_license_cli(["--manifest", str(out_parquet), "--out", str(out_json)])
    assert exit_audit == 0


def test_ingest_pipeline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify Stage 1 ingest pipeline fetches observations and writes raw parquet."""
    import asyncio

    from taxon_vision.data.inat_client import INatAPIClient
    from taxon_vision.pipelines.ingest import ingest_cli, run_ingestion

    mock_records = [
        {
            "id": 101,
            "image_url": "https://inaturalist-open-data.s3.amazonaws.com/101.jpg",
            "attribution": "(c) Tester",
            "license_code": "CC-BY",
            "license_url": "https://creativecommons.org/licenses/by/4.0/",
            "is_open_dataset_verified": True,
        }
    ]

    async def mock_fetch(self: INatAPIClient, taxon_id: int, per_page: int) -> list[dict[str, Any]]:
        return mock_records

    monkeypatch.setattr(INatAPIClient, "fetch_observations", mock_fetch)

    out_file = tmp_path / "raw.parquet"
    asyncio.run(run_ingestion(taxa=[47120], per_page=1, out_path=out_file))
    assert out_file.exists()
    df = pl.read_parquet(out_file)
    assert len(df) == 1
    assert df["taxon_id"][0] == 47120

    cli_out = tmp_path / "cli_raw.parquet"
    ret = ingest_cli(["--taxa", "47120", "--per-page", "1", "--out", str(cli_out)])
    assert ret == 0
    assert cli_out.exists()


def test_export_onnx_pipeline(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify Stage 5 export_onnx pipeline CLI wrapper delegates to exporter."""
    import taxon_vision.pipelines.export_onnx as export_module

    def mock_export(_argv: list[str] | None = None) -> int:
        return 0

    monkeypatch.setattr(export_module, "export_classifier_cli", mock_export)
    assert export_module.main(["--help"]) == 0


def test_calibrate_and_evaluate_pipelines(tmp_path: Path) -> None:
    """Verify Stage 6 calibrate and Stage 7 evaluate pipelines against INT8 ONNX graph."""
    from taxon_vision.inference.onnx_exporter import export_classifier_cli
    from taxon_vision.pipelines.calibrate import calibrate_cli, load_embeddings_and_labels
    from taxon_vision.pipelines.evaluate import evaluate_cli, get_canonical_pareto_profiles

    # 1. Export small quantized ONNX model
    int8_model = tmp_path / "model_int8.onnx"
    export_ret = export_classifier_cli(
        [
            "--quantize-int8",
            "--quantize-mode",
            "dynamic",
            "--quantized-output",
            str(int8_model),
            "--extractor",
            "mobilenetv4_conv_small",
            "--allow-untrained",
        ]
    )
    assert export_ret == 0

    # 2. Create calibration and test embedding partitions (dim=1280 for mobilenetv4)
    cal_pt = tmp_path / "cal_embeddings.pt"
    test_pt = tmp_path / "test_embeddings.pt"
    torch.save(
        {"embeddings": torch.randn(20, 1280), "labels": torch.tensor([i % 10 for i in range(20)])},
        cal_pt,
    )
    torch.save(
        {"embeddings": torch.randn(20, 1280), "labels": torch.tensor([i % 10 for i in range(20)])},
        test_pt,
    )

    # 3. Test load_embeddings_and_labels
    embs, lbls = load_embeddings_and_labels(cal_pt)
    assert embs.shape == (20, 1280)
    assert lbls.shape == (20,)

    # 4. Run Stage 6 calibrate CLI
    conf_yaml = tmp_path / "conformal_config.yaml"
    cal_metrics = tmp_path / "conformal_metrics.json"
    cal_ret = calibrate_cli(
        [
            "--onnx-model",
            str(int8_model),
            "--embeddings",
            str(cal_pt),
            "--out",
            str(conf_yaml),
            "--metrics-out",
            str(cal_metrics),
            "--alpha",
            "0.05",
        ]
    )
    assert cal_ret == 0
    assert conf_yaml.exists()
    assert cal_metrics.exists()

    # 5. Run Stage 7 evaluate CLI
    pareto_json = tmp_path / "pareto_results.json"
    eval_metrics = tmp_path / "pareto_evaluation.json"
    eval_ret = evaluate_cli(
        [
            "--onnx-model",
            str(int8_model),
            "--embeddings",
            str(test_pt),
            "--conformal-config",
            str(conf_yaml),
            "--out",
            str(pareto_json),
            "--metrics-out",
            str(eval_metrics),
            "--latency-sla-ms",
            "50.0",
        ]
    )
    assert eval_ret == 0
    assert pareto_json.exists()
    assert eval_metrics.exists()

    profiles = get_canonical_pareto_profiles()
    assert len(profiles) == 5
