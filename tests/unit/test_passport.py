# SPDX-FileCopyrightText: 2026 Benjamin Förster
# SPDX-License-Identifier: MIT
"""Unit tests for active model passport compilation and governance caching."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from taxon_vision.service.passport import (
    ModelPassport,
    clear_passport_cache,
    load_model_passport,
)
from taxon_vision.service.telemetry_store import DriftRunRecord


@pytest.fixture(autouse=True)
def _reset_passport_cache() -> None:
    """Ensure in-memory passport cache is cleared before and after each test."""
    clear_passport_cache()
    yield
    clear_passport_cache()


def test_load_model_passport_with_valid_reports(tmp_path: Path) -> None:
    """Verify load_model_passport parses local reports and populates all governance fields."""
    eval_file = tmp_path / "pareto_evaluation.json"
    eval_file.write_text(
        json.dumps(
            {
                "nominal_coverage": 0.95,
                "test_empirical_coverage": 0.962,
                "mean_set_size": 1.65,
                "singleton_rate": 0.88,
                "p95_latency_ms": 48.5,
            }
        ),
        encoding="utf-8",
    )

    train_file = tmp_path / "train_metrics.json"
    train_file.write_text(
        json.dumps(
            {
                "extractor": "dinov3",
                "version": "3",
                "mlflow_model_id": "m-dinov3-v3",
            }
        ),
        encoding="utf-8",
    )

    drift_file = tmp_path / "data_drift.json"
    drift_file.write_text(
        json.dumps(
            {
                "n_current": 200,
                "drift_detected": False,
                "mmd2_p_value": 0.42,
            }
        ),
        encoding="utf-8",
    )

    passport = load_model_passport(reports_dir=tmp_path, force_refresh=True)

    assert isinstance(passport, ModelPassport)
    assert passport.model_name == "taxon_vision_onnx@Production"
    assert passport.model_version == "3"
    assert passport.mlflow_model_id == "m-dinov3-v3"
    assert passport.backbone == "dinov3 INT8"
    assert passport.nominal_coverage == 0.95
    assert passport.empirical_test_coverage == 0.962
    assert passport.mean_set_size == 1.65
    assert passport.singleton_rate == 0.88
    assert passport.p95_latency_ms == 48.5
    assert passport.exchangeability_status == "Healthy"
    assert passport.mmd2_p_value == 0.42


def test_load_model_passport_drift_warning(tmp_path: Path) -> None:
    """Verify exchangeability status reflects 'Drift Warning' when drift is detected."""
    drift_file = tmp_path / "data_drift.json"
    drift_file.write_text(
        json.dumps(
            {
                "n_current": 150,
                "drift_detected": True,
                "mmd2_p_value": 0.003,
            }
        ),
        encoding="utf-8",
    )

    passport = load_model_passport(reports_dir=tmp_path, force_refresh=True)
    assert passport.exchangeability_status == "Drift Warning"
    assert passport.mmd2_p_value == 0.003


def test_load_model_passport_warming_up(tmp_path: Path) -> None:
    """Verify exchangeability status reflects 'Warming Up' when sample volume is insufficient."""
    drift_file = tmp_path / "data_drift.json"
    drift_file.write_text(
        json.dumps(
            {
                "n_current": 18,  # Below 50 threshold
                "drift_detected": False,
                "mmd2_p_value": 0.50,
            }
        ),
        encoding="utf-8",
    )

    passport = load_model_passport(reports_dir=tmp_path, force_refresh=True)
    assert passport.exchangeability_status == "Warming Up"
    assert passport.mmd2_p_value == 0.50


def test_load_model_passport_empty_directory(tmp_path: Path) -> None:
    """Verify load_model_passport falls back gracefully when reports are missing."""
    empty_dir = tmp_path / "empty_reports"
    empty_dir.mkdir()

    passport = load_model_passport(reports_dir=empty_dir, force_refresh=True)
    assert isinstance(passport, ModelPassport)
    assert passport.model_name == "taxon_vision_onnx@Production"
    assert passport.exchangeability_status == "Warming Up"
    assert passport.mmd2_p_value is None
    assert passport.nominal_coverage == 0.95


def test_load_model_passport_ttl_caching(tmp_path: Path) -> None:
    """Verify in-memory caching returns identical instance without re-reading filesystem."""
    eval_file = tmp_path / "pareto_evaluation.json"
    eval_file.write_text(
        json.dumps({"test_empirical_coverage": 0.95}),
        encoding="utf-8",
    )

    passport_1 = load_model_passport(reports_dir=tmp_path)

    # Modify file on disk; cached loader should still return first object
    eval_file.write_text(
        json.dumps({"test_empirical_coverage": 0.80}),
        encoding="utf-8",
    )

    passport_2 = load_model_passport(reports_dir=tmp_path)
    assert passport_1 is passport_2
    assert passport_2.empirical_test_coverage == 0.95

    # Force refresh should read updated disk value
    passport_3 = load_model_passport(reports_dir=tmp_path, force_refresh=True)
    assert passport_3 is not passport_1
    assert passport_3.empirical_test_coverage == 0.80


def test_load_model_passport_telemetry_store_prioritization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify live telemetry store records take precedence over static disk files."""
    mock_store = MagicMock()
    mock_run = DriftRunRecord(
        window_size=120,
        w1_distance=0.02,
        w1_threshold=0.15,
        mmd2=0.001,
        mmd2_p_value=0.68,
        drift_detected=False,
        mlflow_run_id="run_live_123",
    )
    mock_store.latest_drift_run.return_value = mock_run
    monkeypatch.setattr(
        "taxon_vision.service.passport.get_telemetry_store",
        lambda: mock_store,
    )

    passport = load_model_passport(reports_dir=tmp_path, force_refresh=True)
    assert passport.exchangeability_status == "Healthy"
    assert passport.mmd2_p_value == 0.68
