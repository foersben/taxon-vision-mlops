import subprocess
import sys
from pathlib import Path

from taxon_vision.inference.cli import main as predict_cli_main
from taxon_vision.models.training.runner import run_training_pipeline, train_cli


def test_run_training_pipeline_direct() -> None:
    res = run_training_pipeline(extractor="mobilenetv4_conv_small", epochs=1, batch_size=8)
    assert res["status"] == "completed"
    assert res["epochs_trained"] == 1
    assert "checkpoint_path" in res
    assert Path(res["checkpoint_path"]).exists()


def test_train_cli_execution() -> None:
    ret = train_cli(["--epochs", "1", "--batch-size", "16"])
    assert ret == 0


def test_training_module_execution() -> None:
    proc = subprocess.run(
        [sys.executable, "-m", "taxon_vision.models.trainer", "--epochs", "1", "--batch-size", "16"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0
    assert "Training Completed Successfully" in proc.stdout


def test_predict_cli_function() -> None:
    sample_img = "data/sample/406185885_taxon_47120.jpg"
    ret = predict_cli_main([sample_img])
    assert ret == 0


def test_predict_module_subprocess() -> None:
    sample_img = "data/sample/406185885_taxon_47120.jpg"
    proc = subprocess.run(
        [sys.executable, "-m", "taxon_vision.inference.cli", sample_img],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0
    assert "TaxonVision Species Prediction Report" in proc.stdout


def test_predict_module_json_mode() -> None:
    sample_img = "data/sample/406185885_taxon_47120.jpg"
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
