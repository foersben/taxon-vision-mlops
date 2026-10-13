# SPDX-FileCopyrightText: 2026 Benjamin Förster
# SPDX-License-Identifier: MIT
"""Shared Pytest fixtures for unit and integration test suites."""

from __future__ import annotations

import pytest

from taxon_vision.domain.taxonomy import TaxonNode


@pytest.fixture
def sample_taxon() -> TaxonNode:
    """Provide a canonical TaxonNode biological entity fixture."""
    return TaxonNode(taxon_id=1, scientific_name="Danaus plexippus", common_name="Monarch Butterfly", rank="species")
