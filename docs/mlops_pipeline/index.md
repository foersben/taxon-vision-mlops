---
type: Architecture
title: MLOps Pipelines & Polyglot Storage
status: stable
stale_after: "2027-01-01T00:00:00Z"
version: "1.0"
description: Overview of the 7-stage Epistemic Diamond DVC pipeline, asymmetric dual-pipeline topology, and multi-tier storage architecture.
tags: [mlops, dvc, pipelines, storage, architecture]
generated: {by: "process:docs-librarian", at: "2026-10-18T18:00:00Z"}
verified: {by: "process:docs-librarian", at: "2026-10-18T18:00:00Z"}
---

# MLOps Pipelines & Polyglot Storage

This section explores the overarching data flow and pipeline topology that powers the TaxonVision-MLOps ecosystem.

In a system dealing with volatile taxonomies and massive observation datasets, a monolithic pipeline is computationally inefficient and brittle. Here, we document our Asymmetric Dual-Pipeline Architecture, which isolates compute-heavy neural network weight updates from I/O-bound taxonomic knowledge graph updates. Additionally, we detail how data is streamed, chunked, and deterministically versioned using DVC and S3 to maintain cryptographic provenance.

## Thematic Sections

* **[Dual Pipeline Topology](dual_pipeline_architecture.md)**: Decoupling the classification engine from the Multimodal Knowledge Graph.
* **[Ingestion & Versioning](ingestion_and_versioning.md)**: Secure, reproducible data ingestion pipelines utilizing DVC and Arrow streaming.
* **[Storage & Databases](storage_and_databases.md)**: Multi-tier storage across SeaweedFS, DuckDB, Polars, and SQLite WAL.
* **[Interactive DVC DAG Explorer](dvc_dag_explorer.md)**: Interactive visualization of the 7-stage Epistemic Diamond build graph.
