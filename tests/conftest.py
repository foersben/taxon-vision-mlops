# Pytest shared fixtures.
"""Conftest py.

This module provides functionality related to conftest.
"""

import pytest

from taxon_vision.domain.taxonomy import TaxonNode


@pytest.fixture
def sample_taxon() -> TaxonNode:
    """Sample taxon.

    Returns:
        The resulting value from the operation.
    """
    return TaxonNode(taxon_id=1, scientific_name="Danaus plexippus", common_name="Monarch Butterfly", rank="species")
