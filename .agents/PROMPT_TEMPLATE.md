---
type: Reference
title: Jules Standard Prompt Appendix
status: stable
stale_after: "2027-06-01T00:00:00Z"
version: 1.0
description: Copy-paste prompt appendix for Jules cloud sessions.
tags: [agents, jules, prompt-template]
generated: {by: process:scaffold-init, at: "2026-10-02T10:00:00Z"}
verified: {by: process:scaffold-init, at: "2026-10-02T10:00:00Z"}
---

# Jules Standard Prompt Appendix

Append the guardrail block below to any Jules task description.

---

## Universal Guardrail Block

```text
MCP Guidance & Token Guardrails:
* Read AGENTS.md at repo root, then open .agents/index.md.
* Use the decision matrix in .agents/index.md to select your role.
* Do NOT run generic directory listings across .agents/.
* Load only the role file (.agents/roles/) for the paths you are modifying.
* All Python: pixi run -e dev or just only.
* Never run act in sandbox.
* Push PR feature branch directly without git commit -S.
* Run pre-commit gates before pushing:
    pixi run -e dev ruff check . && pixi run -e dev mypy src scripts
    pixi run -e dev python scripts/validate_okf.py
    pixi run -e dev python scripts/audit_license_compliance.py
```
