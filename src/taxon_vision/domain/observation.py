# SPDX-FileCopyrightText: 2026 Benjamin Förster
# SPDX-License-Identifier: MIT
"""DarwinCore observation domain entities."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

from taxon_vision.domain.license import AttributionRecord
from taxon_vision.domain.taxonomy import TaxonNode


class ObservationRecord(BaseModel):
    """Validated observation conforming to DarwinCore standard."""

    observation_id: str = Field(..., description="iNaturalist or GBIF observation ID")
    taxon: TaxonNode = Field(..., description="Assigned taxonomic classification")
    attribution: AttributionRecord = Field(..., description="Legal license attribution record")
    image_url: str = Field(..., description="URL to primary photo")
    latitude: float | None = Field(default=None, ge=-90.0, le=90.0)
    longitude: float | None = Field(default=None, ge=-180.0, le=180.0)
    observed_on: datetime | None = Field(default=None)
    quality_grade: str = Field(default="research", description="Quality grade (e.g. research)")
