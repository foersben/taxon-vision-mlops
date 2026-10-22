# SPDX-FileCopyrightText: 2026 Benjamin Förster
# SPDX-License-Identifier: MIT
"""Unit tests for MLflow model registry cleanup script."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# Ensure root workspace is in sys.path for scripts import
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mlflow.entities.model_registry import ModelVersion

from scripts.maintenance.cleanup_mlflow_registry import (
    delete_version_safely,
    execute_pruning,
    extract_metric_score,
    fetch_all_model_versions,
    fetch_protected_aliases,
    identify_protected_versions,
    main,
    parse_arguments,
)


def _make_mock_version(
    version: str,
    aliases: list[str] | None = None,
    stage: str = "None",
    tags: dict[str, str] | None = None,
) -> ModelVersion:
    """Helper to instantiate mock ModelVersion entities."""
    mv = MagicMock(spec=ModelVersion)
    mv.version = version
    mv.aliases = aliases or []
    mv.current_stage = stage
    mv.tags = tags or {}
    return mv


def test_parse_arguments() -> None:
    """Verify parse_arguments default values and custom flag parsing."""
    args = parse_arguments(["--model-name", "custom_model", "--keep-last", "5", "--execute"])
    assert args.model_name == "custom_model"
    assert args.keep_last == 5
    assert args.execute is True


def test_fetch_protected_aliases() -> None:
    """Verify fetch_protected_aliases extracts aliased version strings."""
    mock_client = MagicMock()
    mock_model = MagicMock()
    mock_model.aliases = {"Production": "10", "Challenger": "12"}
    mock_client.get_registered_model.return_value = mock_model

    protected = fetch_protected_aliases(mock_client, "test_model")
    assert protected == {"10", "12"}

    # When get_registered_model raises, returns empty set safely
    mock_client.get_registered_model.side_effect = RuntimeError("API error")
    assert fetch_protected_aliases(mock_client, "test_model") == set()


def test_fetch_all_model_versions_pagination() -> None:
    """Verify fetch_all_model_versions traverses pagination tokens."""
    mock_client = MagicMock()
    page1 = MagicMock()
    v1 = _make_mock_version("1")
    page1.__iter__.return_value = [v1]
    page1.token = "token_page_2"

    page2 = MagicMock()
    v2 = _make_mock_version("2")
    page2.__iter__.return_value = [v2]
    page2.token = None

    mock_client.search_model_versions.side_effect = [page1, page2]

    versions = fetch_all_model_versions(mock_client, "test_model")
    assert len(versions) == 2
    assert mock_client.search_model_versions.call_count == 2


def test_extract_metric_score() -> None:
    """Verify extract_metric_score extracts numeric tags or returns 0.0 fallback."""
    v_valid = _make_mock_version("1", tags={"val_pr_auc": "0.915"})
    assert extract_metric_score(v_valid, "val_pr_auc") == 0.915

    v_missing = _make_mock_version("2", tags={})
    assert extract_metric_score(v_missing, "val_pr_auc") == 0.0

    v_invalid = _make_mock_version("3", tags={"val_pr_auc": "not-a-number"})
    assert extract_metric_score(v_invalid, "val_pr_auc") == 0.0


def test_identify_protected_versions() -> None:
    """Verify identify_protected_versions combines aliases, stages, recency, and metrics."""
    v1 = _make_mock_version("1", tags={"val_pr_auc": "0.99"})  # High score
    v2 = _make_mock_version("2", stage="Production")  # Legacy stage
    v3 = _make_mock_version("3", tags={"val_pr_auc": "0.50"})  # Older, low score
    v4 = _make_mock_version("4", tags={"val_pr_auc": "0.60"})  # Recent
    v5 = _make_mock_version("5", tags={"val_pr_auc": "0.70"})  # Recent

    protected = identify_protected_versions(
        versions=[v1, v2, v3, v4, v5],
        protected_aliases={"2"},
        keep_last=2,  # v5, v4
        keep_top_k=1,  # v1 (0.99)
        metric_key="val_pr_auc",
    )

    assert "1" in protected  # Top-1 metric
    assert "2" in protected  # Alias + Stage
    assert "4" in protected  # Recency
    assert "5" in protected  # Recency
    assert "3" not in protected  # Unaliased, low score, not recent


def test_delete_version_safely() -> None:
    """Verify delete_version_safely executes client call and handles exceptions."""
    mock_client = MagicMock()
    assert delete_version_safely(mock_client, "test_model", "1", request_delay=0.0) is True
    mock_client.delete_model_version.assert_called_once_with(name="test_model", version="1")

    mock_client.delete_model_version.side_effect = RuntimeError("Delete error")
    assert delete_version_safely(mock_client, "test_model", "2", request_delay=0.0) is False


def test_execute_pruning_dry_run_and_live() -> None:
    """Verify execute_pruning handles preview in dry-run and deletions in live mode."""
    mock_client = MagicMock()
    candidates = [_make_mock_version("1"), _make_mock_version("2")]

    # Dry run
    dry_count = execute_pruning(mock_client, "test_model", candidates, dry_run=True, request_delay=0.0)
    assert dry_count == 2
    mock_client.delete_model_version.assert_not_called()

    # Live run
    live_count = execute_pruning(mock_client, "test_model", candidates, dry_run=False, request_delay=0.0)
    assert live_count == 2
    assert mock_client.delete_model_version.call_count == 2


def test_cleanup_registry_main(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify CLI main entrypoint executes cleanly."""
    monkeypatch.setattr(
        "scripts.maintenance.cleanup_mlflow_registry.cleanup_registry",
        MagicMock(return_value=0),
    )
    assert main(["--model-name", "taxon_vision_classifier", "--execute"]) == 0
