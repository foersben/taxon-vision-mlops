# SPDX-FileCopyrightText: 2026 Benjamin Förster
# SPDX-License-Identifier: MIT
"""Taxonomic classification models conforming to DarwinCore standards."""

from __future__ import annotations

from pydantic import BaseModel, Field


class TaxonNode(BaseModel):
    """Taxonomic rank node in the biological tree."""

    taxon_id: int = Field(..., description="Unique taxonomic ID")
    scientific_name: str = Field(..., description="Binomial or uninomial scientific name")
    common_name: str = Field(default="", description="Vernacular name")
    rank: str = Field(default="species", description="Taxonomic rank (kingdom, class, species)")
    parent_id: int | None = Field(default=None, description="Parent taxon node ID")


class TaxonomyCatalog(BaseModel):
    """Collection of supported taxa in the perimeter."""

    taxa: list[TaxonNode] = Field(default_factory=list)

    def get_by_id(self, taxon_id: int) -> TaxonNode | None:
        """Find a taxonomic node by its unique identifier.

        Args:
            taxon_id: The taxon identifier to search for.

        Returns:
            The matching TaxonNode if found, or None.
        """
        for t in self.taxa:
            if t.taxon_id == taxon_id:
                return t
        return None
