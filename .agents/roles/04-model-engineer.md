---
type: Agent Role
title: Directives
status: stable
stale_after: "2027-01-01T00:00:00Z"
version: 1.0
description: "Implements Timm backbone extractors (DINOv3, BioCLIP-2, DINOv2, MobileNetV4), classification heads, and Class-Balanced Loss."
tags: [pytorch, timm, loss]
generated: {by: process:scaffold-init, at: "2026-10-02T10:00:00Z"}
verified: {by: process:scaffold-init, at: "2026-10-02T10:00:00Z"}
role: Model Engineer
---

# Directives for @model-engineer

* **Primary Mandate:** Implements Timm backbone extractors (DINOv3, BioCLIP-2, DINOv2, MobileNetV4), classification heads, and Class-Balanced Loss.
* **Trigger Condition:** Model training, backbones, PyTorch heads.
* **Tooling Standard:** Execute all operations via `pixi run -e dev` or `just`.
* **Verification Gate:** Run test suite and pre-commit checks before completing any task.
