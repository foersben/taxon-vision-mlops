
# Chapter 2: Data & MLOps Pipelines

This chapter explores the overarching data flow and pipeline topology that powers the TaxonVision-MLOps ecosystem.

In a system dealing with volatile taxonomies and massive observation datasets, a monolithic pipeline is computationally inefficient and brittle. Here, we document our Asymmetric Dual-Pipeline Architecture, which isolates the compute-heavy neural network weight updates from the I/O-bound taxonomic knowledge graph updates. Additionally, we detail how data is streamed, chunked, and deterministically versioned using DVC and S3 to maintain cryptographic provenance.

## Thematic Sections

* **Dual Pipeline Topology**: Decoupling the classification engine from the Multimodal Knowledge Graph.
* **Ingestion & Versioning**: Secure, reproducible data ingestion pipelines utilizing DVC.
