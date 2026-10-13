# SPDX-FileCopyrightText: 2026 Benjamin Förster
# SPDX-License-Identifier: MIT
"""Unit tests for open license compliance filtering and static domain prohibitions."""

from __future__ import annotations

from taxon_vision.data.license_filter import LicenseFilter


def test_license_permitted() -> None:
    """Verify whitelist allows open licenses (CC0, CC-BY, CC-BY-NC) and rejects restrictive ones."""
    assert LicenseFilter.is_license_permitted("CC0")
    assert LicenseFilter.is_license_permitted("CC-BY")
    assert LicenseFilter.is_license_permitted("CC-BY-NC")
    assert not LicenseFilter.is_license_permitted("ALL_RIGHTS_RESERVED")
    assert not LicenseFilter.is_license_permitted("COMMERCIAL_ONLY")


def test_static_domain_prohibited() -> None:
    """Verify static domain prohibition rejects un-licensed static domains unless allowed."""
    assert LicenseFilter.is_static_domain_prohibited("https://static.inaturalist.org/photos/1.jpg", False)
    assert not LicenseFilter.is_static_domain_prohibited("https://static.inaturalist.org/photos/1.jpg", True)
    assert not LicenseFilter.is_static_domain_prohibited("https://inaturalist-open-data.s3.amazonaws.com/1.jpg", False)


def test_process_record_valid() -> None:
    """Verify process_record transforms compliant raw observation dicts into structured models."""
    raw = {
        "observation_id": "123",
        "license_code": "CC-BY",
        "image_url": "https://s3.amazonaws.com/123.jpg",
        "attribution": "Naturalist X",
    }
    rec = LicenseFilter.process_record(raw)
    assert rec is not None
    assert rec.photographer_name == "Naturalist X"
    assert rec.license_code.value == "CC-BY"
