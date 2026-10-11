import subprocess
import sys
from pathlib import Path

import pytest
import torch
from PIL import Image

from taxon_vision.inference.cli import main as predict_cli_main
from taxon_vision.models.training.runner import run_training_pipeline, train_cli


@pytest.fixture
def sample_img(tmp_path: Path) -> str:
    img_path = tmp_path / "dummy_image.jpg"
    img = Image.new("RGB", (224, 224), color="red")
    img.save(img_path, "JPEG")
    return str(img_path)


def test_run_training_pipeline_direct() -> None:
    res = run_training_pipeline(extractor="mobilenetv4_conv_small", epochs=1, batch_size=8)
    assert res["status"] == "completed"
    assert res["epochs_trained"] == 1
    assert "checkpoint_path" in res
    assert Path(res["checkpoint_path"]).exists()


def test_train_cli_execution() -> None:
    ret = train_cli(["--epochs", "1", "--batch-size", "16", "--cuda-memory-fraction", "0.7"])
    assert ret == 0


def test_configure_cuda_memory_budget_cpu_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    from taxon_vision.models.training.runner import configure_cuda_memory_budget

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert configure_cuda_memory_budget(0.7) is None


def test_configure_cuda_memory_budget_cuda_mocked(monkeypatch: pytest.MonkeyPatch) -> None:
    from taxon_vision.models.training.runner import configure_cuda_memory_budget

    called_fractions: list[float] = []

    def _mock_set_fraction(frac: float) -> None:
        called_fractions.append(frac)

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "set_per_process_memory_fraction", _mock_set_fraction)

    applied = configure_cuda_memory_budget(0.75)
    assert applied == 0.75
    assert called_fractions == [0.75]


def test_training_module_execution() -> None:
    proc = subprocess.run(
        [sys.executable, "-m", "taxon_vision.models.trainer", "--epochs", "1", "--batch-size", "16"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0
    assert "Training Completed Successfully" in proc.stdout


def test_predict_cli_function(sample_img: str) -> None:
    ret = predict_cli_main([sample_img])
    assert ret == 0


def test_predict_module_subprocess(sample_img: str) -> None:
    proc = subprocess.run(
        [sys.executable, "-m", "taxon_vision.inference.cli", sample_img],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0
    assert "TaxonVision Species Prediction Report" in proc.stdout


def test_predict_module_json_mode(sample_img: str) -> None:
    proc = subprocess.run(
        [sys.executable, "-m", "taxon_vision.inference.cli", "--json", sample_img],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0
    assert '"scientific_name":' in proc.stdout


def test_predict_module_missing_file() -> None:
    proc = subprocess.run(
        [sys.executable, "-m", "taxon_vision.inference.cli", "non_existent_image_12345.jpg"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 1
