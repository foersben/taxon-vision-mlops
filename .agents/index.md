---
okf_version: "0.2"
type: Index
title: TaxonVision Agent Ecosystem - Algorithmic Decision Matrix
status: stable
stale_after: "2027-01-01T00:00:00Z"
version: 1.0
description: Primary decision matrix and skill dispatch index for AI agents in TaxonVision.
tags: [agents, index, decision-matrix, router]
generated: {by: process:okf-updater, at: "2026-10-30T00:00:00Z"}
verified: {by: process:okf-updater, at: "2026-10-30T00:00:00Z"}
---

# TaxonVision Agent Ecosystem - Algorithmic Decision Matrix

* **For AI agents:** Use the decision matrix to select your role.
* **For human readers:** Full annotated directory index follows the matrix.

---

## Quick-Route: Identify Your Task Type

| If you are modifying... | Adopt role | Pre-commit gate? |
| --- | --- | --- |
| `src/taxon_vision/domain/` or `config/taxa_catalog.yaml` | `02-taxonomic-architect` | YES |
| `src/taxon_vision/data/` or DVC pipelines | `03-mlops-data-engineer` | YES (license audit) |
| `src/taxon_vision/models/` or training loops | `04-model-engineer` | YES |
| `src/taxon_vision/inference/` or ONNX exports | `05-inference-optimizer` | YES |
| `src/taxon_vision/uncertainty/` | `06-uncertainty-safety-engineer` | YES (conformal test) |
| `src/taxon_vision/active_learning/` or `feedback/` | `07-active-learning-engineer` | YES |
| `src/taxon_vision/monitoring/` or Pareto scripts | `08-observability-engineer` | YES |
| `src/taxon_vision/service/` or `templates/` | `09-api-and-ui-developer` | YES |
| `tests/` | `10-qa-automator` | YES |
| `docs/` or `.agents/` | `11-docs-librarian` | YES (validate-okf, verify-hardware) |
| `.github/` or `pyproject.toml` | `12-git-operator` | NO |
| Cross-cutting / Planning | `01-orchestrator` | DEPENDS |

---

## Skill Dispatch

| Slash Command | Skill | Pre-load roles |
| --- | --- | --- |
| `/validate-full-stack` | [full-pipeline-validation.md](skills/full-pipeline-validation/SKILL.md) | all |
| `/benchmark-pareto` | [run-pareto-benchmarks.md](skills/run-pareto-benchmarks/SKILL.md) | `04`, `05`, `08` |
| `/audit-conformal` | [conformal-uncertainty-audit.md](skills/conformal-uncertainty-audit/SKILL.md) | `06`, `10` |
| `/active-learning-step` | [active-learning-cycle.md](skills/active-learning-cycle/SKILL.md) | `03`, `06`, `07` |
| `/doc-synchronization-pipeline` | [doc-synchronization-pipeline.md](skills/doc-synchronization-pipeline/SKILL.md) | `01`, `11` |
