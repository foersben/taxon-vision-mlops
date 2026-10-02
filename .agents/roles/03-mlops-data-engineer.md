---
type: Agent Role
title: Directives
status: stable
stale_after: "2027-01-01T00:00:00Z"
version: 1.0
description: "Builds streaming S3 ingestion, DVC versioning, and open-license attribution filters."
tags: [dvc, ingestion, licensing]
generated: {by: process:scaffold-init, at: "2026-10-02T10:00:00Z"}
verified: {by: process:scaffold-init, at: "2026-10-02T10:00:00Z"}
role: MLOps Data Engineer
---

# Directives for @mlops-data-engineer

* **Primary Mandate:** Builds streaming S3 ingestion, DVC versioning, and open-license attribution filters.
* **Trigger Condition:** AWS S3 streamer, license filtering, DVC.
* **Tooling Standard:** Execute all operations via `pixi run -e dev` or `just`.
* **Verification Gate:** Run test suite and pre-commit checks before completing any task.
