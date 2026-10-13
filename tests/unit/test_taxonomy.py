# SPDX-FileCopyrightText: 2026 Benjamin Förster
# SPDX-License-Identifier: MIT
"""Unit tests for TaxonomyCatalog biological identifier lookups."""

from __future__ import annotations

from taxon_vision.domain.taxonomy import TaxonNode, TaxonomyCatalog


def test_taxonomy_catalog() -> None:
    """Verify TaxonomyCatalog retrieves taxa by unique ID and handles missing entries."""
    cat = TaxonomyCatalog(
        taxa=[
            TaxonNode(taxon_id=1, scientific_name="Danaus plexippus", common_name="Monarch"),
            TaxonNode(taxon_id=2, scientific_name="Apis mellifera", common_name="Honey Bee"),
        ]
    )
    node = cat.get_by_id(1)
    assert node is not None
    assert node.scientific_name == "Danaus plexippus"
    assert cat.get_by_id(999) is None
