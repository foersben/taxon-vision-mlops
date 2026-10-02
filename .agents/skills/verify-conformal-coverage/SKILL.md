---
type: Agent Skill
title: Verify Conformal Coverage
status: stable
stale_after: "2027-01-01T00:00:00Z"
version: 1.0
description: Skill to assert empirical coverage meets nominal 1 - alpha rate.
tags: [skill, python]
generated: {by: process:okf-updater, at: "2026-10-02T10:00:00Z"}
verified: {by: process:okf-updater, at: "2026-10-02T10:00:00Z"}
name: Verify Conformal Coverage
sources:
- id: verify_conformal_coverage
  resource: scripts/calibrate_conformal.py
---

# Verify Conformal Coverage

Skill to assert empirical coverage meets nominal 1 - alpha rate.

## Execution

Run using:

```bash
pixi run -e dev python scripts/calibrate_conformal.py
```
