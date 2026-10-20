# SPDX-FileCopyrightText: 2026 Benjamin Förster
# SPDX-License-Identifier: MIT
"""Unit tests for automated hyperparameter tuning with Optuna and MLflow integration."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import mlflow
import optuna
import pytest
import torch

from taxon_vision.models.training.tuning import (
    HyperparameterObjective,
    _log_best_run,
    _log_optuna_diagnostic_plots,
    _resolve_optuna_storage,
    _setup_mlflow,
    tune_hyperparameters,
)
from taxon_vision.models.training.types import EmbeddingSplit, TuningConfig


@pytest.fixture
def synthetic_split() -> EmbeddingSplit:
    """Create a minimal synthetic embedding split for fast deterministic testing."""
    torch.manual_seed(42)
    dim = 8
    num_train = 20
    num_val = 10
    num_classes = 2

    return EmbeddingSplit(
        train_embeddings=torch.randn(num_train, dim),
        train_labels=torch.tensor([i % num_classes for i in range(num_train)]),
        val_embeddings=torch.randn(num_val, dim),
        val_labels=torch.tensor([i % num_classes for i in range(num_val)]),
        samples_per_class=[10, 10],
    )


def test_setup_mlflow_unconfigured(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify _setup_mlflow returns False when tracking URI is missing or empty."""
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "")
    mock_settings = MagicMock()
    mock_settings.mlflow.tracking_uri = ""
    mock_settings.mlflow.experiment_name = "test_exp"
    monkeypatch.setattr("taxon_vision.models.training.tuning.get_settings", lambda: mock_settings)

    cfg = TuningConfig(tracking_uri=None)
    assert _setup_mlflow(cfg) is False


def test_setup_mlflow_success_and_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify _setup_mlflow sets tracking URI and safely handles connection errors."""
    mock_set_uri = MagicMock()
    mock_set_exp = MagicMock()
    monkeypatch.setattr(mlflow, "set_tracking_uri", mock_set_uri)
    monkeypatch.setattr(mlflow, "set_experiment", mock_set_exp)

    cfg = TuningConfig(tracking_uri="http://localhost:5000", experiment_name="tuning_test")
    assert _setup_mlflow(cfg) is True
    mock_set_uri.assert_called_once_with("http://localhost:5000")
    mock_set_exp.assert_called_once_with("tuning_test")

    # When set_experiment raises, returns False safely
    mock_set_exp.side_effect = ConnectionError("Server down")
    assert _setup_mlflow(cfg) is False


def test_log_best_run(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify _log_best_run logs best params, metric, and tags when active."""
    # When inactive: no-op
    _log_best_run({"lr": 1e-3}, 0.85, active=False)

    # When active: starts run and logs details
    mock_start_run = MagicMock()
    mock_log_params = MagicMock()
    mock_log_metric = MagicMock()
    mock_set_tag = MagicMock()
    mock_log_param = MagicMock()

    monkeypatch.setattr(mlflow, "start_run", mock_start_run)
    monkeypatch.setattr(mlflow, "log_params", mock_log_params)
    monkeypatch.setattr(mlflow, "log_metric", mock_log_metric)
    monkeypatch.setattr(mlflow, "set_tag", mock_set_tag)
    monkeypatch.setattr(mlflow, "log_param", mock_log_param)
    monkeypatch.setattr("taxon_vision.models.training.tracking._get_dvc_dataset_hash", lambda: "hash123")

    _log_best_run({"lr": 1e-3}, 0.85, active=True)
    mock_start_run.assert_called_once_with(run_name="optuna_best_model")
    mock_log_params.assert_called_once_with({"lr": 1e-3})
    mock_log_metric.assert_called_once_with("best_val_pr_auc", 0.85)


def test_hyperparameter_objective_trial(synthetic_split: EmbeddingSplit, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify HyperparameterObjective executes trial training loop and logs metrics."""
    mock_start_run = MagicMock()
    mock_end_run = MagicMock()
    mock_log_metric = MagicMock()
    monkeypatch.setattr(mlflow, "start_run", mock_start_run)
    monkeypatch.setattr(mlflow, "end_run", mock_end_run)
    monkeypatch.setattr(mlflow, "log_params", MagicMock())
    monkeypatch.setattr(mlflow, "log_metric", mock_log_metric)
    monkeypatch.setattr(mlflow, "set_tag", MagicMock())

    cfg = TuningConfig(n_trials=1, epochs_per_trial=1, batch_size=16)
    objective = HyperparameterObjective(
        data=synthetic_split,
        num_classes=2,
        config=cfg,
        mlflow_available=True,
    )

    study = optuna.create_study(direction="maximize")
    trial = study.ask()
    score = objective(trial)
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0


def test_tune_hyperparameters_fast(synthetic_split: EmbeddingSplit, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify tune_hyperparameters completes an Optuna study across 1 trial."""
    monkeypatch.setattr("taxon_vision.models.training.tuning._setup_mlflow", lambda _cfg=None: False)
    monkeypatch.setattr("taxon_vision.models.training.tuning._log_best_run", MagicMock())

    cfg = TuningConfig(n_trials=1, epochs_per_trial=1, batch_size=16, seed=42, storage_url=None)
    best_params, study = tune_hyperparameters(
        data=synthetic_split,
        num_classes=2,
        config=cfg,
    )

    assert "lr" in best_params
    assert "weight_decay" in best_params
    assert len(study.trials) == 1


def test_resolve_optuna_storage(tmp_path: Path) -> None:
    """Verify _resolve_optuna_storage handles memory, paths, and SQLite WAL pragmas."""
    assert _resolve_optuna_storage(None) is None
    assert _resolve_optuna_storage("") is None
    assert _resolve_optuna_storage("memory") is None

    # PostgreSQL / non-sqlite URI returned as-is
    pg_uri = "postgresql://user:pass@localhost:5432/optuna_db"
    assert _resolve_optuna_storage(pg_uri) == pg_uri

    # SQLite URI creates directory and configures WAL mode
    db_file = tmp_path / "studies" / "sub" / "optuna.db"
    sqlite_uri = f"sqlite:///{db_file}"

    resolved = _resolve_optuna_storage(sqlite_uri)
    assert resolved == sqlite_uri
    assert db_file.exists()

    conn = sqlite3.connect(str(db_file))
    try:
        journal_mode = conn.execute("PRAGMA journal_mode;").fetchone()[0]
        assert journal_mode.lower() == "wal"
    finally:
        conn.close()


def test_log_optuna_diagnostic_plots_zero_variance(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify _log_optuna_diagnostic_plots safely suppresses param importance on zero variance."""
    mock_log_figure = MagicMock()
    monkeypatch.setattr(mlflow, "log_figure", mock_log_figure)

    # Empty study or None study returns cleanly
    _log_optuna_diagnostic_plots(None)
    empty_study = optuna.create_study()
    _log_optuna_diagnostic_plots(empty_study)
    assert mock_log_figure.call_count == 0

    # Study with identical values (zero variance: all trials 1.0)
    study = optuna.create_study(direction="maximize")
    for _ in range(3):
        trial = study.ask()
        study.tell(trial, 1.0)

    _log_optuna_diagnostic_plots(study)
    # plot_optimization_history is called, but plot_param_importances is skipped because len(set) == 1
    assert mock_log_figure.call_count == 1
    call_args = mock_log_figure.call_args_list[0]
    assert "visualizations/optuna_optimization_history.png" in call_args.args[1]


def test_tune_hyperparameters_custom_study_and_reset(
    synthetic_split: EmbeddingSplit,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify tune_hyperparameters honors study_name and reset_study flags."""
    monkeypatch.setattr("taxon_vision.models.training.tuning._setup_mlflow", lambda _cfg=None: False)
    monkeypatch.setattr("taxon_vision.models.training.tuning._log_best_run", MagicMock())
    monkeypatch.setattr("taxon_vision.models.training.tuning._log_optuna_diagnostic_plots", MagicMock())

    cfg = TuningConfig(
        n_trials=1,
        epochs_per_trial=1,
        batch_size=16,
        seed=42,
        storage_url=None,
        study_name="custom_test_study",
        reset_study=True,
    )
    _, study = tune_hyperparameters(
        data=synthetic_split,
        num_classes=2,
        config=cfg,
    )
    assert study.study_name == "custom_test_study"
    assert len(study.trials) == 1


def test_load_persisted_best_params(tmp_path: Path) -> None:
    """Verify _load_persisted_best_params handles absent files, corrupt yaml, and valid entries."""
    from taxon_vision.models.training.runner import _load_persisted_best_params

    # Non-existent file returns None
    assert _load_persisted_best_params("test_extractor", tmp_path / "nonexistent.yaml") is None

    # Valid YAML with best_params
    yaml_file = tmp_path / "best_params.yaml"
    yaml_file.write_text(
        """
test_model:
  best_params:
    learning_rate: 0.0025
    cb_beta: 0.999
    weight_decay: 0.0001
    dropout: 0.2
    batch_size: 64
    noise_std: 0.05
  best_val_pr_auc: 0.912
other_model:
  learning_rate: 0.001
""",
        encoding="utf-8",
    )
    loaded = _load_persisted_best_params("test_model", yaml_file)
    assert loaded is not None
    assert loaded["learning_rate"] == 0.0025
    assert loaded["cb_beta"] == 0.999

    # Model with flat dictionary
    loaded_other = _load_persisted_best_params("other_model", yaml_file)
    assert loaded_other is not None
    assert loaded_other["learning_rate"] == 0.001

    # Missing extractor returns None
    assert _load_persisted_best_params("missing_model", yaml_file) is None


def test_resolve_tier_hyperparameters(
    synthetic_split: EmbeddingSplit, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify 3-tier hyperparameter resolution: live study, persisted yaml, and defaults."""
    from taxon_vision.config import get_settings
    from taxon_vision.models.training.runner import _resolve_tier_hyperparameters

    settings = get_settings()

    # Tier 1: tune=True -> optuna_live_study
    monkeypatch.setattr("taxon_vision.models.training.tuning._setup_mlflow", lambda _cfg=None: False)
    monkeypatch.setattr("taxon_vision.models.training.tuning._log_best_run", MagicMock())
    monkeypatch.setattr("taxon_vision.models.training.tuning._log_optuna_diagnostic_plots", MagicMock())

    tier1_res = _resolve_tier_hyperparameters(
        extractor="tier1_extractor",
        tune=True,
        n_trials=1,
        epochs=1,
        batch_size=16,
        learning_rate=0.001,
        seed=42,
        data=synthetic_split,
        num_classes=2,
        study_name=None,
        reset_study=False,
        settings=settings,
    )
    assert tier1_res.hyperparameter_source == "optuna_live_study"
    assert tier1_res.learning_rate > 0.0

    # Tier 2: tune=False, persisted best_params.yaml exists
    mock_persisted = {
        "learning_rate": 0.0042,
        "cb_beta": 0.995,
        "weight_decay": 0.0002,
        "dropout": 0.15,
        "batch_size": 32,
        "noise_std": 0.02,
    }

    def _mock_loader(extractor: str) -> dict[str, Any] | None:
        return mock_persisted if extractor == "tier2_extractor" else None

    monkeypatch.setattr("taxon_vision.models.training.runner._load_persisted_best_params", _mock_loader)

    tier2_res = _resolve_tier_hyperparameters(
        extractor="tier2_extractor",
        tune=False,
        n_trials=1,
        epochs=1,
        batch_size=16,
        learning_rate=0.001,
        seed=42,
        data=synthetic_split,
        num_classes=2,
        study_name=None,
        reset_study=False,
        settings=settings,
    )
    assert tier2_res.hyperparameter_source == "persisted_best_params"
    assert tier2_res.learning_rate == 0.0042
    assert tier2_res.cb_beta == 0.995

    # Tier 3: tune=False, no persisted file -> params_yaml_default
    tier3_res = _resolve_tier_hyperparameters(
        extractor="tier3_extractor",
        tune=False,
        n_trials=1,
        epochs=1,
        batch_size=16,
        learning_rate=0.001,
        seed=42,
        data=synthetic_split,
        num_classes=2,
        study_name=None,
        reset_study=False,
        settings=settings,
    )
    assert tier3_res.hyperparameter_source == "params_yaml_default"
    assert tier3_res.learning_rate == 0.001
