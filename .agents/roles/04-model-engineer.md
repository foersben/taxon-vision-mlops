---
type: Agent Role
title: Directives
status: stable
stale_after: "2027-01-01T00:00:00Z"
version: 1.0
description: "Implements Timm backbone extractors (DINOv3, BioCLIP-2), classification heads, Class-Balanced Loss, and Optuna HPO."
tags: [pytorch, timm, loss]
generated: {by: process:scaffold-init, at: "2026-10-02T10:00:00Z"}
verified: {by: process:scaffold-init, at: "2026-10-02T10:00:00Z"}
role: Model Engineer
---

# Directives for @model-engineer

* **Primary Mandate:** Implements vision foundation backbones (`dinov3`, `bioclip2` via `factory.py`), linear/MLP classification heads, Class-Balanced Loss ($\beta$), and Bayesian hyperparameter optimization (Optuna TPE + ASHA).
* **Owned Source Files:**
    * `src/taxon_vision/models/{factory,head,loss,trainer}.py`
    * `src/taxon_vision/models/training/{runner,loops,embeddings,tuning,tracking,checkpoint,types}.py`
* **Targeted Unit Tests:**
    * `tests/unit/test_models_components.py`
    * `tests/unit/test_tuning.py`
    * `tests/unit/test_tracking.py`
* **Trigger Condition:** Model architecture changes, training loops, loss functions, hyperparameter optimization.
* **Tooling Standard:** Execute all operations via `pixi run -e dev` or `just`.
* **Fast Verification:** `pixi run -e ci-dev pytest tests/unit/test_models_components.py tests/unit/test_tuning.py -q -o "addopts="`
* **Verification Gate:** Run full test suite and pre-commit checks before completing any task.
