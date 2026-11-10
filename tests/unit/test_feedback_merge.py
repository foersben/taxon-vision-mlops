# SPDX-FileCopyrightText: 2026 Benjamin Förster
# SPDX-License-Identifier: MIT
"""Unit tests verifying active learning feedback ingestion and licensing gates."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from taxon_vision.active_learning.feedback import (
    TaxaMappings,
    load_taxa_mappings,
    merge_feedback_to_manifest,
    parse_single_task_annotation,
)


@pytest.fixture
def taxa_mappings() -> TaxaMappings:
    """Fixture providing canonical taxa lookup mappings."""
    return load_taxa_mappings("config/taxa_catalog.yaml")


def test_parse_valid_annotation_with_attribution(taxa_mappings: TaxaMappings) -> None:
    """Verify valid annotation with permitted CC license and photographer attribution produces a record."""
    task = {
        "id": 101,
        "data": {
            "observation_uuid": "obs_101",
            "image_url": "https://example.com/crop101.jpg",
            "photographer_name": "Maria Biologist",
            "license_code": "CC-BY",
            "license_url": "https://creativecommons.org/licenses/by/4.0/",
        },
        "annotations": [
            {
                "result": [
                    {
                        "from_name": "taxon_choice",
                        "value": {"choices": ["Danaus plexippus (Monarch Butterfly)"]},
                    }
                ]
            }
        ],
    }

    parsed = parse_single_task_annotation(task, taxa_mappings)
    assert not parsed.is_ood
    assert parsed.record is not None
    assert parsed.record["observation_uuid"] == "obs_101"
    assert parsed.record["taxon_id"] == 48662
    assert parsed.record["license_code"] == "CC-BY"
    assert parsed.record["photographer_name"] == "Maria Biologist"


def test_parse_ood_classification(taxa_mappings: TaxaMappings) -> None:
    """Verify that Out-of-Distribution annotations are flagged as OOD and excluded from training."""
    task = {
        "id": 102,
        "data": {
            "observation_uuid": "obs_102",
            "image_url": "https://example.com/crop102.jpg",
            "photographer_name": "John Doe",
            "license_code": "CC0",
        },
        "annotations": [
            {
                "result": [
                    {
                        "from_name": "taxon_choice",
                        "value": {"choices": ["Out of Distribution (Non-Target / Novel Organism)"]},
                    }
                ]
            }
        ],
    }

    parsed = parse_single_task_annotation(task, taxa_mappings)
    assert parsed.is_ood
    assert parsed.record is None


def test_parse_rejects_unlicensed_observation(taxa_mappings: TaxaMappings) -> None:
    """Verify that observations with copyright or non-permissive licenses are rejected."""
    task = {
        "id": 103,
        "data": {
            "observation_uuid": "obs_103",
            "image_url": "https://example.com/crop103.jpg",
            "photographer_name": "Jane Doe",
            "license_code": "all-rights-reserved",
        },
        "annotations": [
            {
                "result": [
                    {
                        "from_name": "taxon_choice",
                        "value": {"choices": ["Danaus plexippus (Monarch Butterfly)"]},
                    }
                ]
            }
        ],
    }

    parsed = parse_single_task_annotation(task, taxa_mappings)
    assert not parsed.is_ood
    assert parsed.record is None


def test_parse_rejects_missing_photographer(taxa_mappings: TaxaMappings) -> None:
    """Verify that observations lacking photographer attribution are strictly rejected."""
    task = {
        "id": 104,
        "data": {
            "observation_uuid": "obs_104",
            "image_url": "https://example.com/crop104.jpg",
            "photographer_name": "",
            "license_code": "CC-BY",
        },
        "annotations": [
            {
                "result": [
                    {
                        "from_name": "taxon_choice",
                        "value": {"choices": ["Apis mellifera (Western Honey Bee)"]},
                    }
                ]
            }
        ],
    }

    parsed = parse_single_task_annotation(task, taxa_mappings)
    assert not parsed.is_ood
    assert parsed.record is None


def test_merge_feedback_idempotency(tmp_path: Path) -> None:
    """Verify that merging annotations is idempotent and prevents duplicate observation UUIDs."""
    manifest_file = tmp_path / "dataset_manifest.parquet"
    export_file = tmp_path / "label_studio_export.json"

    initial_df = pd.DataFrame(
        [
            {
                "observation_uuid": "existing_obs_001",
                "taxon_id": 48662,
                "image_url": "https://example.com/1.jpg",
                "photographer_name": "Alice",
                "license_code": "CC0",
                "license_url": "",
            }
        ]
    )
    initial_df.to_parquet(manifest_file, index=False)

    export_data = [
        {
            "id": 1,
            "data": {
                "observation_uuid": "existing_obs_001",
                "image_url": "https://example.com/1.jpg",
                "photographer_name": "Alice",
                "license_code": "CC0",
            },
            "annotations": [
                {
                    "result": [
                        {
                            "from_name": "taxon_choice",
                            "value": {"choices": ["Danaus plexippus (Monarch Butterfly)"]},
                        }
                    ]
                }
            ],
        },
        {
            "id": 2,
            "data": {
                "observation_uuid": "new_obs_002",
                "image_url": "https://example.com/2.jpg",
                "photographer_name": "Bob",
                "license_code": "CC-BY",
            },
            "annotations": [
                {
                    "result": [
                        {
                            "from_name": "taxon_choice",
                            "value": {"choices": ["Apis mellifera (Western Honey Bee)"]},
                        }
                    ]
                }
            ],
        },
    ]
    export_file.write_text(json.dumps(export_data), encoding="utf-8")

    res1 = merge_feedback_to_manifest(export_file, manifest_file)
    assert res1.status == "merged"
    assert res1.new_records_added == 1
    assert res1.total_manifest_records == 2

    res2 = merge_feedback_to_manifest(export_file, manifest_file)
    assert res2.status == "skipped_no_new_records"
    assert res2.new_records_added == 0
    assert res2.total_manifest_records == 2
