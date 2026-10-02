# TaxonVision Agent Ecosystem - Algorithmic Decision Matrix

* **For MCP agents (Jules):** Use the decision matrix to select your role.
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
| `docs/` or `.agents/` | `11-docs-librarian` | YES (validate-okf) |
| `.github/` or `pyproject.toml` | `12-git-operator` | NO |
| Cross-cutting / Planning | `01-orchestrator` | DEPENDS |

---

## Workflow Dispatch

| Slash Command | Workflow File | Pre-load roles |
| --- | --- | --- |
| `/validate-full-stack` | [full-pipeline-validation.md](workflows/full-pipeline-validation.md) | all |
| `/benchmark-pareto` | [model-benchmark-pareto.md](workflows/model-benchmark-pareto.md) | `04`, `05`, `08` |
| `/audit-conformal` | [conformal-uncertainty-audit.md](workflows/conformal-uncertainty-audit.md) | `06`, `10` |
| `/active-learning-step` | [active-learning-cycle.md](workflows/active-learning-cycle.md) | `03`, `06`, `07` |
| `/doc-synchronization-pipeline` | [doc-synchronization-pipeline.md](workflows/doc-synchronization-pipeline.md) | `01`, `11` |
| `/jules-session-triage` | [jules-session-triage.md](workflows/jules-session-triage.md) | `01`, `10`, `12` |
