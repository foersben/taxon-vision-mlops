# SPDX-FileCopyrightText: 2026 Benjamin Förster
# SPDX-License-Identifier: MIT
"""Unit tests for training CLI, prediction CLI, and training pipeline execution."""

from __future__ import annotations

import subprocess
import sys
import urllib.request
from pathlib import Path
from typing import Any
from urllib.error import URLError

import pytest
import torch
from PIL import Image

from taxon_vision.inference.cli import main as predict_cli_main
from taxon_vision.models.training import runner as runner_mod
from taxon_vision.models.training.runner import (
    configure_cuda_memory_budget,
    run_training_pipeline,
)
from taxon_vision.models.training.types import TrainingPipelineResult
from taxon_vision.pipelines.train import (
    dispatch_remote_training,
    main as train_pipeline_main,
    train_cli,
)


@pytest.fixture
def sample_img(tmp_path: Path) -> str:
    """Create a temporary JPEG image file for prediction CLI tests."""
    img_path = tmp_path / "dummy_image.jpg"
    img = Image.new("RGB", (224, 224), color="red")
    img.save(img_path, "JPEG")
    return str(img_path)


def test_run_training_pipeline_direct() -> None:
    """Verify programmatic execution of the training pipeline returns completed status."""
    res = run_training_pipeline(extractor="mobilenetv4_conv_small", epochs=1, batch_size=8)
    assert res["status"] == "completed"
    assert res["epochs_trained"] == 1
    assert "checkpoint_path" in res
    assert Path(res["checkpoint_path"]).exists()


def test_train_cli_execution() -> None:
    """Verify train_cli runs successfully with epoch and CUDA fraction arguments."""
    ret = train_cli(["--epochs", "1", "--batch-size", "16", "--cuda-memory-fraction", "0.7"])
    assert ret == 0


def test_configure_cuda_memory_budget_cpu_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify configure_cuda_memory_budget safely returns None when CUDA is unavailable."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert configure_cuda_memory_budget(0.7) is None


def test_configure_cuda_memory_budget_cuda_mocked(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify configure_cuda_memory_budget applies memory fraction when CUDA is present."""
    called_fractions: list[float] = []

    def _mock_set_fraction(frac: float) -> None:
        called_fractions.append(frac)

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "set_per_process_memory_fraction", _mock_set_fraction)

    applied = configure_cuda_memory_budget(0.75)
    assert applied == 0.75
    assert called_fractions == [0.75]


def test_training_module_execution() -> None:
    """Verify execution of taxon_vision.pipelines.train module via subprocess."""
    proc = subprocess.run(
        [sys.executable, "-m", "taxon_vision.pipelines.train", "--epochs", "1", "--batch-size", "16"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0
    assert "Training Completed Successfully" in proc.stdout


def test_predict_cli_function(sample_img: str) -> None:
    """Verify prediction CLI entrypoint with single input image."""
    ret = predict_cli_main([sample_img])
    assert ret == 0


def test_predict_module_subprocess(sample_img: str) -> None:
    """Verify prediction CLI subprocess execution outputs formatted text report."""
    proc = subprocess.run(
        [sys.executable, "-m", "taxon_vision.inference.cli", sample_img],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0
    assert "TaxonVision Species Prediction Report" in proc.stdout


def test_predict_module_json_mode(sample_img: str) -> None:
    """Verify prediction CLI in --json mode outputs valid JSON serialization."""
    proc = subprocess.run(
        [sys.executable, "-m", "taxon_vision.inference.cli", "--json", sample_img],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0
    assert '"scientific_name":' in proc.stdout


def test_predict_module_missing_file() -> None:
    """Verify prediction CLI exits with error status when target image file is missing."""
    proc = subprocess.run(
        [sys.executable, "-m", "taxon_vision.inference.cli", "non_existent_image_12345.jpg"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 1


def test_train_cli_require_cache_missing() -> None:
    """Verify --require-cache raises FileNotFoundError when cache file is missing."""
    with pytest.raises(FileNotFoundError, match="Synthetic fallback is disabled via --require-cache"):
        train_cli(["--epochs", "1", "--cache-path", "non_existent_cache_file.pt", "--require-cache"])


def test_train_cli_require_promotion_archived(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify --require-promotion returns non-zero when candidate model fails promotion."""
    orig_run = runner_mod.run_training_pipeline

    def mock_run(*args: Any, **kwargs: Any) -> TrainingPipelineResult:
        res = orig_run(*args, **kwargs)
        res["promotion_status"] = "Archived"
        return res

    monkeypatch.setattr(runner_mod, "run_training_pipeline", mock_run)
    ret = train_cli(["--epochs", "1", "--require-promotion"])
    assert ret == 1


def test_train_pipeline_main_entrypoint() -> None:
    """Verify pipeline train main entrypoint execution."""
    ret = train_pipeline_main(["--epochs", "1", "--batch-size", "16"])
    assert ret == 0


def test_dispatch_remote_training(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify HTTP remote training dispatch success and failure error handling."""

    class MockResponse:
        def __init__(self, data: bytes) -> None:
            self._data = data

        def read(self) -> bytes:
            return self._data

        def __enter__(self) -> MockResponse:
            return self

        def __exit__(self, *args: object) -> None:
            pass

    def mock_urlopen_success(req: Any, timeout: float = 30) -> MockResponse:
        return MockResponse(b'{"status": "ok", "message": "Success"}')

    def mock_urlopen_fail(req: Any, timeout: float = 30) -> Any:
        raise URLError("Connection refused")

    monkeypatch.setattr(urllib.request, "urlopen", mock_urlopen_success)
    ret_ok = dispatch_remote_training("http://localhost:8000", "dinov3", 1, 16, 0.001)
    assert ret_ok == 0

    monkeypatch.setattr(urllib.request, "urlopen", mock_urlopen_fail)
    ret_fail = dispatch_remote_training("http://localhost:8000", "dinov3", 1, 16, 0.001)
    assert ret_fail == 1


def test_build_training_components_typed() -> None:
    """Verify build_training_components typed container and sequence unpacking."""
    from taxon_vision.models.training.runner import build_training_components
    from taxon_vision.models.training.types import TrainingComponents, TrainingHyperparameters

    hparams = TrainingHyperparameters(
        learning_rate=0.001,
        weight_decay=1e-4,
        dropout=0.2,
        cb_beta=0.999,
        batch_size=16,
        noise_std=0.01,
        hyperparameter_source="test",
    )

    # Test canonical build_training_components
    components = build_training_components(
        dim=32,
        num_classes=3,
        samples=[10, 20, 30],
        hparams=hparams,
        epochs=2,
        extractor="dinov3",
    )
    assert isinstance(components, TrainingComponents)
    assert components.params["extractor"] == "dinov3"
    assert components.params["num_classes"] == 3
    assert components.config.epochs == 2

    # Verify sequence unpacking support
    head, criterion, optimizer, config, params = components
    assert head is components.head
    assert criterion is components.criterion
    assert optimizer is components.optimizer
    assert config is components.config
    assert params is components.params
