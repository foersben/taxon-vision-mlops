# SPDX-FileCopyrightText: 2026 Benjamin Förster
# SPDX-License-Identifier: MIT
"""Filter for open licenses and photographer attribution retention."""

from __future__ import annotations

from typing import Any

from taxon_vision.domain.license import AttributionRecord, OpenLicense


class LicenseFilter:
    """Enforces open licensing constraints on citizen science imagery.

    Attributes:
        Various internal state and configuration variables used by the class.
    """

    @staticmethod
    def is_license_permitted(license_str: str) -> bool:
        """Is license permitted.

        Args:
            license_str: The license str parameter.

        Returns:
            The resulting value from the operation.
        """
        norm = license_str.strip().upper().replace("_", "-")
        try:
            OpenLicense(norm)
            return True
        except ValueError:
            return False

    @staticmethod
    def is_static_domain_prohibited(image_url: str, is_open_dataset_verified: bool = False) -> bool:
        """Rejects unverified images hosted on static domain without open license.

        Args:
            image_url: The image url parameter.
            is_open_dataset_verified: The is open dataset verified parameter.

        Returns:
            The resulting value from the operation.
        """
        if "static.inaturalist.org" in image_url and not is_open_dataset_verified:
            return True
        return False

    @classmethod
    def process_record(cls, raw: dict[str, Any]) -> AttributionRecord | None:
        """Extract attribution record if compliant, otherwise None.

        Args:
            raw: The raw parameter.

        Returns:
            The resulting value from the operation.
        """
        license_str = str(raw.get("license_code", "")).strip().upper().replace("_", "-")
        image_url = str(raw.get("image_url", ""))
        verified = bool(raw.get("is_open_dataset_verified", False))

        if not cls.is_license_permitted(license_str):
            return None
        if cls.is_static_domain_prohibited(image_url, verified):
            return None

        photographer = str(raw.get("attribution", "")).strip() or "Unknown Contributor"
        return AttributionRecord(
            photographer_name=photographer,
            license_code=OpenLicense(license_str),
            license_url=str(raw.get("license_url", "https://creativecommons.org/licenses/")),
            observation_uuid=str(raw.get("observation_id", "")),
        )
