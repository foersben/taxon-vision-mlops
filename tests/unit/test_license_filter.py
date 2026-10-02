"""Test license filter.py.

This module provides functionality related to test_license_filter.
"""

from taxon_vision.data.license_filter import LicenseFilter


def test_license_permitted() -> None:
    """Test license permitted."""
    assert LicenseFilter.is_license_permitted("CC0")
    assert LicenseFilter.is_license_permitted("CC-BY")
    assert LicenseFilter.is_license_permitted("CC-BY-NC")
    assert not LicenseFilter.is_license_permitted("ALL_RIGHTS_RESERVED")
    assert not LicenseFilter.is_license_permitted("COMMERCIAL_ONLY")


def test_static_domain_prohibited() -> None:
    """Test static domain prohibited."""
    assert LicenseFilter.is_static_domain_prohibited("https://static.inaturalist.org/photos/1.jpg", False)
    assert not LicenseFilter.is_static_domain_prohibited("https://static.inaturalist.org/photos/1.jpg", True)
    assert not LicenseFilter.is_static_domain_prohibited("https://inaturalist-open-data.s3.amazonaws.com/1.jpg", False)


def test_process_record_valid() -> None:
    """Test process record valid."""
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
