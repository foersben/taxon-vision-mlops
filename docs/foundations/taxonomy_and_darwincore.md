---
type: Reference
title: Taxonomy & DarwinCore Standards
status: stable
stale_after: "2027-01-01T00:00:00Z"
version: 1.0
description: Exhaustive specification of biological taxonomy models, DarwinCore schema alignments, and taxonomic resolution mechanics.
tags: [taxonomy, darwincore, biology, gbif]
verified: {by: process:jules-agent}
---

# Taxonomy & DarwinCore Standards

The platform strictly organizes biological observations according to the hierarchical Linnaean taxonomy, ensuring that classification systems map deterministically into the international DarwinCore (DwC) biodiversity standard. This framework acts as the foundational schema for all machine learning data ingestion and taxonomic resolution operations.

## DarwinCore Schema Mapping

The core DarwinCore terms define the structural and semantic constraints for species observations. The following table specifies the formal schema mapping implemented within the architecture:

| DwC Term | Data Type | Constraint | Description |
| --- | --- | --- | --- |
| `scientificName` | String | Required | The full scientific name, including authorship and date information if available. |
| `taxonRank` | Enum | Required | The taxonomic rank of the most specific name in the `scientificName`. |
| `taxonomicStatus` | Enum | Optional | The status of the use of the `scientificName` (e.g., `accepted`, `synonym`). |
| `acceptedTaxonKey` | Integer | Optional | The unique identifier for the accepted taxon in the event of synonymy. |
| `kingdom` | String | Required | The full scientific name of the kingdom in which the taxon is classified. |
| `phylum` | String | Optional | The full scientific name of the phylum or division in which the taxon is classified. |
| `class` | String | Optional | The full scientific name of the class in which the taxon is classified. |
| `order` | String | Optional | The full scientific name of the order in which the taxon is classified. |
| `family` | String | Optional | The full scientific name of the family in which the taxon is classified. |
| `genus` | String | Optional | The full scientific name of the genus in which the taxon is classified. |
| `species` | String | Optional | The specific epithet within the genus. |

## GBIF Taxonomic Backbone Resolution Mechanics

The system utilizes the Global Biodiversity Information Facility (GBIF) Taxonomic Backbone to achieve consistent resolution of taxonomic entities.

* **Parent-Child Taxonomic Tree Hierarchy**
    * The architecture maintains a strict directed acyclic graph (DAG) representing the taxonomic hierarchy.
    * Each node validates its structural lineage constraints up to the `kingdom` level to prevent orphan nodes.
* **Synonym Resolution**
    * Ingested observations are subjected to synonym reconciliation against the GBIF backbone.
    * Queries matching a `taxonomicStatus` of `synonym` are automatically remapped to their corresponding `acceptedTaxonKey`.
* **TaxonKey Hashing**
    * To optimize O(1) lookup latency in production, integer `taxonKey` values are mapped to deterministic SHA-256 hashes within the inference cache.
    * This ensures cross-platform consistency of taxonomic keys during model serving operations.

## Taxonomic Revisions and Complex Edge Cases

Machine learning datasets in biological domains require formal logic to handle the continuous flux of taxonomic classification.

* **Handling of Paraphyletic Groups**
    * The system implements specialized masking logic to prevent gradient disruption when evaluating paraphyletic clades.
    * Lineages recognized as paraphyletic are tagged within the hierarchical mapping to override standard monophyletic constraint assumptions.
* **Taxonomic Revisions**
    * Real-time taxonomy updates are processed via a decoupled ingestion pipeline.
    * The schema isolates compute-heavy GPU weight updates from high-velocity schema realignments.
* **Split and Merge Reconciliation**
    * Splits: When a single taxon is divided into multiple descendant taxa, historical observations are routed to a human-in-the-loop (HITL) review queue for targeted reclassification.
    * Merges: When multiple taxa are consolidated, the respective `taxonKey` identifiers are aliased to the newly designated `acceptedTaxonKey`, merging feature embeddings in the unified vector space.
