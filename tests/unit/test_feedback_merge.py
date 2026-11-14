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


def test_merge_feedback_novel_taxon_outcome_b_expands_catalogue_and_xml(tmp_path: Path) -> None:
    """Outcome B novel taxon selection appends verified rows and expands catalogue, params.yaml, and Label Studio XML."""
    import json

    import pandas as pd
    import yaml

    from taxon_vision.active_learning.feedback import FeedbackMergeOptions, merge_feedback_to_manifest

    catalog_path = tmp_path / "taxa_catalog.yaml"
    catalog_path.write_text(
        yaml.safe_dump(
            {
                "taxa": [
                    {
                        "id": 1,
                        "scientific_name": "Danaus plexippus",
                        "common_name": "Monarch Butterfly",
                        "kingdom": "Animalia",
                        "class": "Insecta",
                        "inat_taxon_id": 48662,
                    }
                ]
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    params_path = tmp_path / "params.yaml"
    params_path.write_text(yaml.safe_dump({"ingestion": {"taxa_to_fetch": [48662]}}, sort_keys=False), encoding="utf-8")
    xml_path = tmp_path / "project_config.xml"
    xml_path.write_text(
        '<View><Choices name="taxon_choice" toName="image">\n    <Choice value="Out of Distribution" alias="ood" />\n  </Choices></View>',
        encoding="utf-8",
    )

    export_path = tmp_path / "export.json"
    export_path.write_text(
        json.dumps(
            [
                {
                    "id": 99,
                    "data": {
                        "observation_uuid": "label_studio_novel_1",
                        "image_url": "s3://taxon-vision-mlops/triage-queue/n1.jpg",
                        "photographer_name": "Dr. Botanist",
                        "license_code": "CC-BY",
                        "license_url": "https://creativecommons.org/licenses/by/4.0/",
                    },
                    "annotations": [
                        {
                            "result": [
                                {
                                    "from_name": "taxon_choice",
                                    "value": {"choices": ["Novel Target Taxon (Expand Catalogue)"]},
                                },
                                {"from_name": "novel_inat_taxon_id", "value": {"text": ["52129"]}},
                                {"from_name": "novel_scientific_name", "value": {"text": ["Vanessa cardui"]}},
                                {"from_name": "novel_common_name", "value": {"text": ["Painted Lady"]}},
                                {"from_name": "novel_kingdom", "value": {"text": ["Animalia"]}},
                                {"from_name": "novel_class", "value": {"text": ["Insecta"]}},
                            ]
                        }
                    ],
                }
            ]
        ),
        encoding="utf-8",
    )

    raw_manifest = tmp_path / "raw_manifest.parquet"
    res = merge_feedback_to_manifest(
        export_path,
        manifest_path=raw_manifest,
        catalog_path=catalog_path,
        options=FeedbackMergeOptions(params_path=params_path, label_studio_config_path=xml_path),
    )

    assert res.status == "merged"
    assert res.catalog_expanded is True
    assert res.new_taxon_ids == (52129,)

    df = pd.read_parquet(raw_manifest)
    assert df.loc[0, "taxon_id"] == 52129
    assert df.loc[0, "attribution"] == "Dr. Botanist"
    assert bool(df.loc[0, "is_open_dataset_verified"]) is True

    updated_cat = yaml.safe_load(catalog_path.read_text(encoding="utf-8"))
    assert [int(t["inat_taxon_id"]) for t in updated_cat["taxa"]] == [48662, 52129]
    assert yaml.safe_load(params_path.read_text(encoding="utf-8"))["ingestion"]["taxa_to_fetch"] == [48662, 52129]
    assert 'alias="52129"' in xml_path.read_text(encoding="utf-8")


def test_merge_feedback_readonly_stages_expansions_and_routes_ood(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Read-only ConfigMap catalog stages Outcome B expansions to YAML whilst Outcome C OOD rows isolate into ood_verified.parquet."""
    import json

    import pandas as pd
    import yaml

    from taxon_vision.active_learning import feedback as fb_mod

    catalog_path = tmp_path / "taxa_catalog.yaml"
    catalog_path.write_text(
        yaml.safe_dump({"taxa": [{"id": 1, "scientific_name": "Danaus plexippus", "inat_taxon_id": 48662}]}),
        encoding="utf-8",
    )
    params_path = tmp_path / "params.yaml"
    params_path.write_text(yaml.safe_dump({"ingestion": {"taxa_to_fetch": [48662]}}), encoding="utf-8")
    xml_path = tmp_path / "project_config.xml"
    xml_path.write_text(
        '<View><Choices name="taxon_choice">\n    <Choice value="Out of Distribution" alias="ood" />\n  </Choices></View>',
        encoding="utf-8",
    )
    pending_yaml = tmp_path / "catalog_expansions.yaml"
    ood_parquet = tmp_path / "ood_verified.parquet"
    raw_manifest = tmp_path / "raw_manifest.parquet"

    export_path = tmp_path / "export.json"
    export_path.write_text(
        json.dumps(
            [
                {
                    "id": 1,
                    "data": {
                        "observation_uuid": "label_studio_novel_ro",
                        "image_url": "s3://taxon-vision-mlops/triage-queue/ro.jpg",
                        "photographer_name": "Field Expert",
                        "license_code": "CC0",
                    },
                    "annotations": [
                        {
                            "result": [
                                {"from_name": "taxon_choice", "value": {"choices": ["novel_taxon"]}},
                                {"from_name": "novel_inat_taxon_id", "value": {"text": ["99991"]}},
                                {"from_name": "novel_scientific_name", "value": {"text": ["Bombus terrestris"]}},
                            ]
                        }
                    ],
                },
                {
                    "id": 2,
                    "data": {
                        "observation_uuid": "label_studio_ood_1",
                        "image_url": "s3://taxon-vision-mlops/triage-queue/ood.jpg",
                        "photographer_name": "Naturalist",
                        "license_code": "CC-BY",
                    },
                    "annotations": [{"result": [{"from_name": "taxon_choice", "value": {"choices": ["ood"]}}]}],
                },
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(fb_mod, "_is_path_writable", lambda target: target != catalog_path)
    res = fb_mod.merge_feedback_to_manifest(
        export_path,
        manifest_path=raw_manifest,
        catalog_path=catalog_path,
        options=fb_mod.FeedbackMergeOptions(pending_expansions_path=pending_yaml, ood_manifest_path=ood_parquet),
    )

    assert res.catalog_expanded is True
    assert res.new_taxon_ids == (99991,)
    assert res.ood_verified_added == 1
    assert pending_yaml.is_file()
    assert pd.read_parquet(ood_parquet)["observation_uuid"].tolist() == ["label_studio_ood_1"]
    assert pd.read_parquet(raw_manifest)["observation_uuid"].tolist() == ["label_studio_novel_ro"]

    committed = fb_mod.apply_pending_catalog_expansions(pending_yaml, catalog_path, params_path, xml_path)
    assert committed == [99991]
    assert 'alias="99991"' in xml_path.read_text(encoding="utf-8")


def test_ingest_preserves_existing_label_studio_records(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Re-running iNaturalist ingestion preserves active-learning human annotations in raw_manifest.parquet."""
    import asyncio

    import polars as pl

    from taxon_vision.pipelines import ingest as ingest_mod

    raw_path = tmp_path / "raw_manifest.parquet"
    pl.DataFrame(
        [
            {
                "observation_uuid": "label_studio_obs_77",
                "taxon_id": 48662,
                "image_url": "s3://taxon-vision-mlops/triage-queue/77_crop.jpg",
                "photographer_name": "Contributor A",
                "attribution": "Contributor A",
                "license_code": "CC-BY",
                "license_url": "https://creativecommons.org/licenses/by/4.0/",
                "is_open_dataset_verified": True,
            }
        ]
    ).write_parquet(raw_path)

    class _FakeClient:
        async def fetch_observations(self, **_kwargs: object) -> list[dict[str, object]]:
            return [
                {
                    "observation_id": "inat_100",
                    "image_url": "https://inaturalist-open-data.s3.amazonaws.com/photos/1/medium.jpg",
                    "attribution": "Inat User",
                    "license_code": "CC0",
                    "license_url": "https://creativecommons.org/publicdomain/zero/1.0/",
                    "is_open_dataset_verified": True,
                }
            ]

    monkeypatch.setattr(ingest_mod, "INatAPIClient", _FakeClient)
    asyncio.run(ingest_mod.run_ingestion(taxa=[48662], per_page=1, out_path=raw_path))

    merged_df = pl.read_parquet(raw_path)
    assert set(merged_df["observation_uuid"].to_list()) == {"inat_100", "label_studio_obs_77"}
