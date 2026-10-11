# SPDX-FileCopyrightText: 2026 Benjamin Förster
# SPDX-License-Identifier: MIT
"""DarwinCore observation domain entities."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

from taxon_vision.domain.license import AttributionRecord
from taxon_vision.domain.taxonomy import TaxonNode


class ObservationRecord(BaseModel):
    """Validated observation conforming to DarwinCore standard.

    An observation is a record of an interaction between a person and an organism at a specific time and place.  Each observation can have multiple media (images or sounds) associated with it. However, for the purpose of this project, we only consider the primary image of the observation.

    Attributes:
        observation_id: Unique identifier of observation
        taxon: Taxonomic classification of observation
        attribution: Attribution record for media licensing
        image_url: URL to primary photo
        latitude: Latitude coordinate
        longitude: Longitude coordinate
        observed_on: Date and time of observation
        quality_grade: Quality grade of observation

    Usage Examples:
        - Import and use `ObservationRecord` model for observation data.
        - Validate incoming observation records from data ingestion pipelines.
        - Create and serialize `ObservationRecord` instances for downstream storage or sharing.

    Example:
        >>> from taxon_vision.domain.observation import ObservationRecord
        >>> from taxon_vision.domain.taxonomy import TaxonNode
        >>> from taxon_vision.domain.license import AttributionRecord
        >>>
        >>> observation = ObservationRecord(
        >>>     observation_id="obs-12345-abc",
        >>>     taxon=TaxonNode(
        >>>         taxon_id=12345,
        >>>         scientific_name="Homo sapiens",
        >>>         common_name="Human",
        >>>         rank="species",
        >>>         parent_id=9606
        >>>     ),
        >>>     attribution=AttributionRecord(
        >>>         photographer_name="Jane_Doe",
        >>>         license_code="CC-BY",
        >>>         license_url="https://creativecommons.org/licenses/by/4.0/",
        >>>         observation_uuid="obs-12345-abc",
        >>>     ),
        >>>     image_url="https://example.com/image.jpg",
        >>>     latitude=40.7128,
        >>>     longitude=-74.0060,
        >>>     observed_on="2023-10-27T10:00:00Z",
        >>>     quality_grade="research"
        >>> )
        >>>
        >>> print(observation.observation_id)
        obs-12345-abc
    """

    observation_id: str = Field(..., description="iNaturalist or GBIF observation ID")
    taxon: TaxonNode = Field(..., description="Assigned taxonomic classification")
    attribution: AttributionRecord = Field(..., description="Legal license attribution record")
    image_url: str = Field(..., description="URL to primary photo")
    latitude: float | None = Field(default=None, ge=-90.0, le=90.0)
    longitude: float | None = Field(default=None, ge=-180.0, le=180.0)
    observed_on: datetime | None = Field(default=None)
    quality_grade: str = Field(default="research", description="Quality grade (e.g. research)")
