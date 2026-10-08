# SPDX-FileCopyrightText: 2026 Benjamin Förster
# SPDX-License-Identifier: MIT
"""Taxonomic classification models conforming to DarwinCore standards."""

from __future__ import annotations

from pydantic import BaseModel, Field


class TaxonNode(BaseModel):
    """Taxonomic rank node in the biological tree.

    Attributes:
        taxon_id: Unique taxonomic ID
        scientific_name: Binomial or uninomial scientific name
        common_name: Vernacular name
        rank: Taxonomic rank (kingdom, class, species)
        parent_id: Parent taxon node ID

    Usage Examples:
        - Import and use `TaxonNode` model for taxonomic data.
        - Create and validate `TaxonNode` instances for incoming observation records.

    Example:
        >>> from taxon_vision.domain.taxonomy import TaxonNode
        >>>
        >>> taxon = TaxonNode(
        >>>     taxon_id=12345,
        >>>     scientific_name="Homo sapiens",
        >>>     common_name="Human",
        >>>     rank="species",
        >>>     parent_id=9606
        >>> )
        >>>
        >>> print(taxon.scientific_name)
        Homo sapiens
    """

    taxon_id: int = Field(..., description="Unique taxonomic ID")
    scientific_name: str = Field(..., description="Binomial or uninomial scientific name")
    common_name: str = Field(default="", description="Vernacular name")
    rank: str = Field(default="species", description="Taxonomic rank (kingdom, class, species)")
    parent_id: int | None = Field(default=None, description="Parent taxon node ID")


class TaxonomyCatalog(BaseModel):
    """Collection of supported taxa in the perimeter.

    Attributes:
        taxa: List of taxon nodes

    Usage Examples:
        - Import and use `TaxonNode` and `TaxonomyCatalog` models for taxonomic data.
        - Use `get_by_id` to retrieve a specific taxon node from the catalog.

    Example:
        >>> from taxon_vision.domain.taxonomy import TaxonomyCatalog
        >>>
        >>> taxonomy = TaxonomyCatalog(taxa=[
        >>>     TaxonNode(
        >>>         taxon_id=12345,
        >>>         scientific_name="Homo sapiens",
        >>>         common_name="Human",
        >>>         rank="species",
        >>>         parent_id=9606
        >>>     )
        >>> ])
        >>>
        >>> taxon = taxonomy.get_by_id(12345)
        >>> print(taxon.scientific_name)
        Homo sapiens
    """

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
