---
type: Agent Rule
title: "Rule 00: Python Modernization"
status: stable
stale_after: "2027-01-01T00:00:00Z"
version: 1.0
description: "Solely rely on Pixi (.pixi/). Ban pip, poetry, uv, .venv, hasattr/getattr, magic numbers, and eager imports."
tags: [python, tooling]
generated: {by: process:scaffold-init, at: "2026-10-02T10:00:00Z"}
verified: {by: process:scaffold-init, at: "2026-10-02T10:00:00Z"}
trigger: always_on
severity: critical
---

# Mandates: Python Modernization

* Solely rely on Pixi (`.pixi/`). Ban `pip`, `poetry`, `uv`, and `.venv` creation.
* Zero dynamic reflection workarounds: strictly forbid `hasattr` and `getattr`; use explicit static types, protocols, or typed Pydantic models.
* Zero hardcoded magic numbers in application and pipeline code: all default numerical values, tolerances, and timeouts must reside in `config/default_config.yaml` and map to typed fields in `src/taxon_vision/config.py`.
* Zero eager submodule imports in `__init__.py`: package initialization files must never eagerly import submodules or CLI entrypoints, preventing namespace pollution and `RuntimeWarning` issues during `python -m` execution.
* Cognitive complexity invariant: all functions and methods must maintain a cognitive complexity score strictly $< 15$ (`complexipy`).
