"""Test taxonomy.py.

This module provides functionality related to test_taxonomy.
"""

from taxon_vision.domain.taxonomy import TaxonNode, TaxonomyCatalog


def test_taxonomy_catalog() -> None:
    """Test taxonomy catalog."""
    cat = TaxonomyCatalog(
        taxa=[
            TaxonNode(taxon_id=1, scientific_name="Danaus plexippus", common_name="Monarch"),
            TaxonNode(taxon_id=2, scientific_name="Apis mellifera", common_name="Honey Bee"),
        ]
    )
    assert cat.get_by_id(1) is not None
    assert cat.get_by_id(1).scientific_name == "Danaus plexippus"
    assert cat.get_by_id(999) is None
