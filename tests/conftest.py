# SPDX-FileCopyrightText: 2026 Benjamin Förster
# SPDX-License-Identifier: MIT
"""Shared Pytest fixtures for unit and integration test suites."""

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path

import pytest

from taxon_vision.domain.taxonomy import TaxonNode


@pytest.fixture(autouse=True)
def test_environment_isolation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Generator[None, None, None]:
    """Isolate test execution with offline fallbacks and temporary flagged storage directory.

    Why:
        Guarantees that test executions remain 100% offline and deterministic without relying on pre-trained network weights or polluting production data directories with test artifacts.

    How:
        Sets environment variables TAXON_ALLOW_UNTRAINED_FALLBACK and TAXON_FLAGGED_STORAGE_DIR, clearing configuration and model caches before and after test execution.

    Args:
        tmp_path: Pytest temporary directory fixture.
        monkeypatch: Pytest monkeypatch fixture.
    """
    flagged_dir = tmp_path / "flagged_observations"
    flagged_dir.mkdir(parents=True, exist_ok=True)
    mlflow_db = tmp_path / "mlflow_test.db"
    sqlite_uri = f"sqlite:///{mlflow_db}"

    monkeypatch.setenv("TAXON_ALLOW_UNTRAINED_FALLBACK", "true")
    monkeypatch.setenv("TAXON_FLAGGED_STORAGE_DIR", str(flagged_dir))
    monkeypatch.setenv("TAXON_OPTUNA_STORAGE_URI", "memory")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", sqlite_uri)
    monkeypatch.setenv("TAXON_MLFLOW_TRACKING_URI", sqlite_uri)
    monkeypatch.setenv("TAXON_VISION_TESTING", "1")

    from taxon_vision.config import get_settings
    from taxon_vision.service.inference import reset_model_caches
    from taxon_vision.service.storage import get_image_cache

    get_settings.cache_clear()
    get_image_cache.cache_clear()
    reset_model_caches()

    try:
        import mlflow

        mlflow.set_tracking_uri(sqlite_uri)
    except Exception:
        pass

    yield

    try:
        import mlflow

        if mlflow.active_run():
            mlflow.end_run()
    except Exception:
        pass

    get_settings.cache_clear()
    get_image_cache.cache_clear()
    reset_model_caches()


@pytest.fixture
def sample_taxon() -> TaxonNode:
    """Provide a canonical TaxonNode biological entity fixture."""
    return TaxonNode(taxon_id=1, scientific_name="Danaus plexippus", common_name="Monarch Butterfly", rank="species")
