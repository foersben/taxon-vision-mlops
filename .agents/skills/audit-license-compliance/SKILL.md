---
type: Agent Skill
title: Audit License Compliance
status: stable
stale_after: "2027-01-01T00:00:00Z"
version: 1.0
description: Skill to verify open licenses and attribution retention.
tags: [skill, python]
generated: {by: process:okf-updater, at: "2026-10-02T10:00:00Z"}
verified: {by: process:okf-updater, at: "2026-10-02T10:00:00Z"}
name: Audit License Compliance
sources:
- id: audit_license_compliance
  resource: scripts/audit_license_compliance.py
---

# Audit License Compliance

Skill to verify open licenses and attribution retention.

## Execution

Run using:

```bash
pixi run -e dev python scripts/audit_license_compliance.py
```
