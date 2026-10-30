---
type: Agent Memory
title: Cognitive Complexity Ratchet Memory
status: stable
stale_after: "2027-01-01T00:00:00Z"
version: 1.0
description: Record of codebase cognitive complexity baseline, legacy hotspots, and ratcheting history via complexipy.
tags: [complexity, refactoring, memory, complexipy]
generated: {by: process:okf-updater, at: "2026-10-30T00:00:00Z"}
verified: {by: process:okf-updater, at: "2026-10-30T00:00:00Z"}
---

# Cognitive Complexity Ratchet Memory

## 1. Governance Policy

* All newly created or modified functions in `src/` and `scripts/` must satisfy cognitive complexity score $\le 15$ measured by `complexipy src scripts --max-complexity-allowed 15 --check-script --no-ignore`.
* Zero tolerance for `# complexipy: ignore` or disabling pragmas.
* Decreasing complexity during refactoring automatically ratchets the baseline watermark in `complexipy-snapshot.json`.

## 2. Baseline Status

* **Core Application Code (`src/taxon_vision/`):** 100% compliant ($\le 15$ across all modules).
    * `src/taxon_vision/data/s3_streamer.py`: `_stream_concurrent` = 13, `_stream_sequential` = 12.
    * `src/taxon_vision/inference/cam.py`: `_resolve_default_classifier` = 11.
    * `src/taxon_vision/pipelines/`: all stages compliant ($\le 14$).
* **Governance Scripts:** 100% compliant ($\le 15$ across all functions with zero snapshot exemptions).
    * `scripts/governance/validate_okf.py`: all functions $\le 10$, `main` = 4.
    * `scripts/governance/visualize_okf.py`: all functions $\le 10$.
    * `scripts/governance/verify_code_invariants.py`: all functions $\le 10$.
