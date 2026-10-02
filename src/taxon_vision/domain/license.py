# SPDX-FileCopyrightText: 2026 Benjamin Förster
# SPDX-License-Identifier: MIT
"""Open license and attribution domain models."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field


class OpenLicense(StrEnum):
    """Approved open licenses for iNaturalist citizen science media."""

    CC0 = "CC0"
    CC_BY = "CC-BY"
    CC_BY_NC = "CC-BY-NC"
    CC_BY_SA = "CC-BY-SA"
    CC_BY_ND = "CC-BY-ND"
    CC_BY_NC_SA = "CC-BY-NC-SA"
    CC_BY_NC_ND = "CC-BY-NC-ND"


class AttributionRecord(BaseModel):
    """Attribution metadata mandated for open-licensed media retention."""

    photographer_name: str = Field(..., description="Name or username of photographer")
    license_code: OpenLicense = Field(..., description="Creative commons license variant")
    license_url: str = Field(..., description="Link to legal code")
    observation_uuid: str = Field(..., description="UUID or identifier of observation")
    source_portal: str = Field(default="iNaturalist", description="Originating repository")
