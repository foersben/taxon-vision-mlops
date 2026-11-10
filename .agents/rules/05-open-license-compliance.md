---
type: Agent Rule
title: "Rule 05: Open License Compliance"
status: stable
stale_after: "2027-01-01T00:00:00Z"
version: 1.0
description: "Permitted Creative Commons licenses, attribution preservation, static domain filtering, and audit gates."
tags: [licensing, inat, compliance, open-data]
generated: {by: process:scaffold-init, at: "2026-10-02T10:00:00Z"}
verified: {by: process:okf-updater, at: "2026-10-30T00:00:00Z"}
trigger: always_on
severity: critical
---

# Mandates: Open License Compliance

All training data and photographic assets must adhere strictly to open-licensing invariants to prevent copyright infringement and legal non-compliance.

## 1. Permitted License Whitelist

Only citizen-science observations released under the following explicit Creative Commons licenses are admitted into the TaxonVision dataset manifests:

* **CC0 1.0 Universal (`CC0`):** Public domain dedication.
* **Creative Commons Attribution 4.0 (`CC-BY`):** Open use with attribution.
* **Creative Commons Attribution-NonCommercial 4.0 (`CC-BY-NC`):** Non-commercial open use with attribution.

All observations marked with `all-rights-reserved`, non-derivative (`CC-BY-ND`), or unspecified licenses must be rejected during Stage 1 ingestion and purged during Stage 2 license auditing.

## 2. Photographer Attribution Preservation

* Photographic attribution metadata (`photographer_name`, `license_code`, `photo_id`, `observation_uuid`) must be preserved across every stage of the Epistemic Diamond DAG.
* Pruning or stripping attribution columns from `raw_manifest.parquet` or `dataset_manifest.parquet` is strictly prohibited.
* The web operator interface and API `/explain` endpoints must display original photographer credits alongside predicted taxa.

## 3. Static Domain & Scraping Bans

* S3 streaming is restricted to official open-access buckets (`s3://inaturalist-open-data/`).
* Scraped web imagery, unauthorized static domains, or unverified secondary aggregators are rejected by the domain filter in `src/taxon_vision/data/license_filter.py`.

## 4. Mandatory Audit Gate

Before committing changes to datasets or ingestion logic, the license auditor must execute with zero non-compliant records:

```bash
pixi run -e dev python -m taxon_vision.pipelines.audit_license
```
