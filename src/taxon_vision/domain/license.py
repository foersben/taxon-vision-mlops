# SPDX-FileCopyrightText: 2026 Benjamin Förster
# SPDX-License-Identifier: MIT
"""Open license and attribution domain models.

This module defines the domain entities governing open-licensed media from platforms such as iNaturalist.

Why:
    Creative Commons and public domain licenses dictate data provenance, reuse rights, and attribution requirements for citizen science datasets. Explicit modeling ensures legal compliance and preserves the integrity of open research workflows without relying on unstructured metadata fields.

How:
    - `OpenLicense` is a constrained enum of recognized Creative Commons licenses approved for model training and derived works. Using StrEnum provides string-based APIs while retaining type safety.
    - `AttributionRecord` embeds structured provenance metadata (photographer, license, observation identifier, source portal) into a Pydantic BaseModel. This ensures consistent attribution when combining multiple models or publishing derived datasets.

Prometheus integration:
    While not directly exposing metrics, this module serves as the schema foundation for tracking license compliance in the telemetry pipeline:
    - `model_metadata_total[status]` (see `src/taxon_vision/telemetry.py`) records observations passing license validation.
    - `model_retention_hours` tracks long-term storage durations for licensed media subsets.

Dependencies:
    - `enum.StrEnum` for typed license enumeration (Python 3.11+).
    - `pydantic.BaseModel` for structured data validation.

Usage Examples:
    Import and use `OpenLicense` enum in validation logic for incoming observation metadata.
    Create and serialize `AttributionRecord` instances when storing or sharing derived datasets that include licensed media.

Example:
    >>> from taxon_vision.domain.license import OpenLicense, AttributionRecord
    >>>
    >>> if license_code == OpenLicense.CC_BY_NC:
    >>>     attribution = AttributionRecord(
    >>>         photographer_name="Jane_Doe",
    >>>         license_code=OpenLicense.CC_BY_NC,
    >>>         license_url="https://creativecommons.org/licenses/by-nc/4.0/",
    >>>         observation_uuid="obs-12345-abc",
    >>>     )
    >>>
    >>>     # Emit telemetry metric for licensed data
    >>>     from taxon_vision import telemetry
    >>>     telemetry.model_metadata_total.labels(status="licensed_data").inc()
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field


class OpenLicense(StrEnum):
    """Approved open licenses for iNaturalist citizen science media.

    Attributes:
        CC0: Public domain dedication.
        CC_BY: Attribution required.
        CC_BY_NC: Attribution-NonCommercial required.
        CC_BY_SA: Attribution-ShareAlike required.
        CC_BY_ND: Attribution-NoDerivatives required.
        CC_BY_NC_SA: Attribution-NonCommercial-ShareAlike required.
        CC_BY_NC_ND: Attribution-NonCommercial-NoDerivatives required.
    """

    CC0 = "CC0"
    CC_BY = "CC-BY"
    CC_BY_NC = "CC-BY-NC"
    CC_BY_SA = "CC-BY-SA"
    CC_BY_ND = "CC-BY-ND"
    CC_BY_NC_SA = "CC-BY-NC-SA"
    CC_BY_NC_ND = "CC-BY-NC-ND"


class AttributionRecord(BaseModel):
    """Attribution metadata mandated for open-licensed media retention.

    Attributes:
        photographer_name: Name or username of photographer
        license_code: Creative commons license variant
        license_url: Link to legal code
        observation_uuid: UUID or identifier of observation
        source_portal: Originating repository
    """

    photographer_name: str = Field(..., description="Name or username of photographer")
    license_code: OpenLicense = Field(..., description="Creative commons license variant")
    license_url: str = Field(..., description="Link to legal code")
    observation_uuid: str = Field(..., description="UUID or identifier of observation")
    source_portal: str = Field(default="iNaturalist", description="Originating repository")
