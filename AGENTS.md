# TaxonVision-MLOps Master Agent Router

<!-- UNIVERSAL RULES: apply to all AI agents (Antigravity, Jules, Copilot, etc.) -->

* All Python commands MUST use `pixi run -e dev` or `just`. Never bare `python`, `pip`, `uv`, `uvx`, or `poetry`. Never create or rely on `.venv/`.
* Never run `act` in agent sandboxes; `act` is for local workstation Docker runs only.
* Commit signing:
    * **Local agents (Antigravity) & human developers:** Commits MUST be GPG/SSH signed (`git commit -S`). Stop and escalate if signing fails.
    * **Cloud sandbox agents (Jules):** Creates PR feature branches without `-S`. Commits are signed upon PR merge via GitHub Web-Flow.
* Pre-commit gates (run before ANY commit to `src/taxon_vision/` or `config/`):

```bash
pixi run -e dev ruff check . && pixi run -e dev mypy src scripts
pixi run -e dev python scripts/audit_license_compliance.py
pixi run -e dev python scripts/validate_okf.py
```

<!-- JULES-SPECIFIC: session routing and token guardrail instructions. -->

## Jules: Mandatory First Step

Open `.agents/index.md` and follow its decision matrix before touching any file.
Stop loading additional files once you have a role assignment.

## Jules: Hard Routing Rules

1. **Route first.** Read `.agents/index.md` decision matrix before any action.
2. **Path-scoped roles.** Load only the role file for the paths you are modifying.
3. **Commit protocol.** Push PR feature branch directly without `git commit -S`.
4. **Restricted directories.** Do NOT open `.agents/memory/` unless the user prompt uses the words `historical` or `canon`.
5. **Do NOT crawl `.agents/` blindly** or pre-load all role files.
