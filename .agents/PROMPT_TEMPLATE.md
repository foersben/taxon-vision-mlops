---
type: Reference
title: Autonomous Agent Standard Prompt Appendix
status: stable
stale_after: "2027-06-01T00:00:00Z"
version: 1.0
description: Copy-paste prompt appendix for autonomous agent sessions.
tags: [agents, prompt-template]
generated: {by: process:scaffold-init, at: "2026-10-02T10:00:00Z"}
verified: {by: process:scaffold-init, at: "2026-10-02T10:00:00Z"}
---

# Agent Task Execution Template (Token-Optimal)

Append the guardrail block below to any agent task prompt.

---

## Universal Guardrail Block

```text
> Act as @<role-id>.
> Task: <1-2 sentence objective>
> Allowed Read Scope: ONLY read <explicit list of 2-4 files>. Do NOT scan unrelated directories, openwiki/.claims/, docs/latex/*.pdf, or pixi.lock.
> Forbidden Write Scope: Do NOT modify <files owned by other tasks>.
>
> Step 1 (Plan): Inspect ONLY the Allowed Read Scope and output a concise bulleted plan (< 15 lines) listing exact functions/lines to change.
> Step 2 (Execute): Apply changes adhering to .agents/rules/00-python-modernization.md (zero magic numbers, zero hasattr/getattr, cognitive complexity < 15, Google+OKF docstrings) and .agents/rules/04-markdown-formatting.md (British English orthography, analytical German-native idiom).
> Step 3 (Fast Targeted Verify): Run ONLY the targeted checks for your touched files:
>     pixi run -e ci-dev ruff check <touched_files> && pixi run -e ci-dev mypy <touched_files> && pixi run -e ci-dev pytest <targeted_test_file> -q -o "addopts="
> Step 4 (Final Gate): Once Step 3 passes, run:
>     pixi run -e ci-dev python scripts/governance/validate_okf.py && pixi run -e ci-dev complexipy <touched_src_files>
```
