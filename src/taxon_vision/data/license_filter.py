# SPDX-FileCopyrightText: 2026 Benjamin Förster
# SPDX-License-Identifier: MIT
"""Filter for open licenses and photographer attribution retention."""

from __future__ import annotations

from typing import Any

from taxon_vision.domain.license import AttributionRecord, OpenLicense


class LicenseFilter:
    """Enforces open licensing constraints and attribution retention on observation imagery.

    Why:
        Citizen science imagery (such as iNaturalist or GBIF media) is uploaded under diverse
        copyright licenses. Ingesting proprietary "All Rights Reserved" assets or unverified
        static domain scrapes into training datasets exposes production vision pipelines to
        copyright infringement and breaks open science reproducibility. TaxonVision strictly
        permits only open Creative Commons licenses (CC0, CC-BY, CC-BY-NC) and requires complete
        retention of contributor attribution records.

    How:
        Normalizes license string identifiers, validates against the `OpenLicense` enum, checks
        domain provenance to prevent direct unverified hotlinking of raw `static.inaturalist.org`
        assets, and extracts structured `AttributionRecord` containers containing photographer
        credits, license URLs, and observation UUIDs.
    """

    @staticmethod
    def is_license_permitted(license_str: str) -> bool:
        """Validate whether a license identifier corresponds to an admitted open license.

        Why:
            Ensures that only CC0, CC-BY, and CC-BY-NC licensed media enter downstream feature
            extraction and training pipelines. Inadmissible licenses (e.g. CC-BY-ND, All Rights Reserved)
            are filtered out at ingestion before hitting storage tiers.

        How:
            Strips surrounding whitespace, converts to uppercase, replaces underscores with standard
            hyphens, and attempts validation against the `OpenLicense` enumeration. Returns True if
            valid, False if a ValueError is raised.

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
        """Reject unverified image assets hosted directly on static domains without manifest verification.

        Why:
            Directly scraping `static.inaturalist.org` endpoints without AWS Open Data registry
            verification violates API terms of service and risks scraping un-licensed or private
            observations. Ingestion is only permitted for assets verified against the official
            open data registry manifests.

        How:
            Checks if `static.inaturalist.org` is present in the URI string and asserts whether
            `is_open_dataset_verified` is False.

        Args:
            image_url: Remote image locator URI string.
            is_open_dataset_verified: Boolean flag indicating official registry verification.

        Returns:
            True if the image URI is prohibited; False if verified or hosted on permitted mirrors.
        """
        if "static.inaturalist.org" in image_url and not is_open_dataset_verified:
            return True
        return False

    @classmethod
    def process_record(cls, raw: dict[str, Any]) -> AttributionRecord | None:
        """Extract validated attribution metadata from raw observation records.

        Why:
            Under Creative Commons attribution terms (CC-BY, CC-BY-NC), creators must receive
            proper attribution including creator name, license type, and source link. Stripping
            photographer credits invalidates license compliance. This method enforces atomic
            retention of attribution records for compliant observations and rejects non-compliant entries.

        How:
            1. Extracts and normalizes `license_code` and `image_url`.
            2. Verifies open license admissibility via `is_license_permitted`.
            3. Verifies domain provenance via `is_static_domain_prohibited`.
            4. Extracts photographer credit string (defaulting to 'Unknown Contributor' if blank).
            5. Instantiates and returns an immutable `AttributionRecord`.

        Args:
            raw: Raw dictionary containing observation metadata from API or parquet manifests.

        Returns:
            Structured AttributionRecord if observation complies with open licensing; None otherwise.
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
