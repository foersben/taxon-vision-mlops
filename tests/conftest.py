# Pytest shared fixtures.
import pytest

from taxon_vision.domain.taxonomy import TaxonNode


@pytest.fixture
def sample_taxon() -> TaxonNode:
    return TaxonNode(taxon_id=1, scientific_name="Danaus plexippus", common_name="Monarch Butterfly", rank="species")
