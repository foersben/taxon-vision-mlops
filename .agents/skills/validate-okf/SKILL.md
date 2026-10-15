---
type: Agent Skill
title: Validate Open Knowledge Format
status: stable
stale_after: "2027-01-01T00:00:00Z"
version: 1.0
description: Skill to verify that markdown files contain correct frontmatter.
tags: [skill, python]
generated: {by: process:okf-updater, at: "2026-10-02T10:00:00Z"}
verified: {by: process:okf-updater, at: "2026-10-02T10:00:00Z"}
name: Validate Open Knowledge Format
sources:
- id: validate_okf
  resource: scripts/validate_okf.py
---

# Validate Open Knowledge Format

Skill to verify that markdown files across `.agents/`, `docs/`, and `openwiki/` conform to Google OKF v0.2.

## Execution

Run using:

```bash
pixi run -e dev python scripts/validate_okf.py
```

Optionally pass target files or directories:

```bash
pixi run -e dev python scripts/validate_okf.py openwiki/
```
