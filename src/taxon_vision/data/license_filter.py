# SPDX-FileCopyrightText: 2026 Benjamin Förster
# SPDX-License-Identifier: MIT
"""Filter for open licenses and photographer attribution retention."""

from __future__ import annotations

from typing import Any

from taxon_vision.domain.license import AttributionRecord, OpenLicense


class LicenseFilter:
    """Enforces open licensing constraints on citizen science imagery."""

    @staticmethod
    def is_license_permitted(license_str: str) -> bool:
        """Validate whether a license string corresponds to an admitted open license.

        Args:
            license_str: String representation of license code (e.g. 'CC0', 'CC-BY', 'CC-BY-NC').

        Returns:
            True if the license is permitted for dataset inclusion; False otherwise.
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
            image_url: The image URL to check.
            is_open_dataset_verified: Whether the image is from an open dataset.

        Returns:
            True if the image is from a static domain and not from an open dataset; False otherwise.
        """
        if "static.inaturalist.org" in image_url and not is_open_dataset_verified:
            return True
        return False

    @classmethod
    def process_record(cls, raw: dict[str, Any]) -> AttributionRecord | None:
        """Extract attribution record if compliant, otherwise None.

        Args:
            raw: The raw record to process.

        Returns:
            An attribution record if the record is compliant; None otherwise.
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
