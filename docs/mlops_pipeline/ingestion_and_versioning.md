---
type: System Pattern
title: Ingestion & Dataset Versioning with DVC
status: stable
stale_after: "2027-01-01T00:00:00Z"
version: 1.0
description: Data ingestion and reproducible versioning strategies using DVC and S3.
tags: [dvc, versioning, s3, pipelines]
generated: {by: process:scaffold-init, at: "2026-10-02T10:00:00Z"}
verified: {by: process:scaffold-init, at: "2026-10-02T10:00:00Z"}
---

# Ingestion & Dataset Versioning with DVC

All raw observations and processed Parquet manifests are versioned using **DVC** with remote S3 storage. Deterministic SHA-256 hashes ensure immutability across training runs.
