---
type: Agent Skill
title: Audit License Compliance
status: stable
stale_after: "2027-01-01T00:00:00Z"
version: 1.0
description: Skill to verify open licenses, photographer attribution retention, and filter compliant manifests.
tags: [skill, python, licensing, data]
generated: {by: process:okf-updater, at: "2026-10-17T22:30:00Z"}
verified: {by: process:okf-updater, at: "2026-10-17T22:30:00Z"}
name: Audit License Compliance
sources:
  - id: audit_license_pipeline
    resource: src/taxon_vision/pipelines/audit_license.py
  - id: audit_license_compliance_script
    resource: scripts/audit_license_compliance.py
---

# Audit License Compliance

Skill to enforce open-license compliance (CC0, CC-BY, CC-BY-NC) and filter observation manifests for downstream feature extraction.

## Execution

* Run canonical Stage 2 active gatekeeper pipeline:

```bash
pixi run -e dev python -m taxon_vision.pipelines.audit_license \
  --manifest data/manifests/raw_manifest.parquet \
  --verified-manifest data/manifests/dataset_manifest.parquet \
  --report reports/license_audit.json
```

* Alternatively, run the backwards-compatible CLI script:

```bash
pixi run -e dev python scripts/audit_license_compliance.py \
  --manifest data/manifests/raw_manifest.parquet \
  --verified-manifest data/manifests/dataset_manifest.parquet \
  --report reports/license_audit.json
```
