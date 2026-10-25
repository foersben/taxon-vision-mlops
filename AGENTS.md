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
pixi run -e dev python -m taxon_vision.pipelines.audit_license
pixi run -e dev python scripts/governance/validate_okf.py
pixi run -e dev python scripts/governance/verify_hardware_invariants.py
```

## Fast Module & Test Routing Map

Read ONLY the files in your active task row:

| Domain | Primary Source Files (`src/taxon_vision/`) | Targeted Test (`tests/`) | Role & Rule |
| :--- | :--- | :--- | :--- |
| **Config & Types** | `config.py`, `params.yaml`, `config/default_config.yaml` | `unit/test_config.py` | `@01`, `00-python` |
| **Data & S3** | `data/{s3_streamer,inat_client,license_filter}.py` | `unit/test_data_components.py` | `@03`, `05-license` |
| **Pipelines** | `pipelines/{ingest,audit_license,extract_features,train,export_onnx,calibrate,evaluate}.py` | `unit/test_pipelines.py` | `@03`, `01-mlops` |
| **Models & Loss** | `models/{factory,head,loss,trainer}.py` | `unit/test_models_components.py` | `@04`, `01-mlops` |
| **Training & HPO** | `models/training/{runner,loops,embeddings,tuning,tracking,checkpoint,types}.py` | `unit/test_tuning.py`, `unit/test_tracking.py` | `@04`, `01-mlops` |
| **ONNX & CAM** | `inference/{onnx_exporter,onnx_engine,quantizer,cam,calibration,cli}.py` | `unit/test_onnx_exporter.py`, `unit/test_quantizer.py` | `@05`, `02-latency` |
| **OOD & Conformal** | `uncertainty/{ood_detector,conformal}.py` | `unit/test_ood_energy.py`, `unit/test_conformal_bounds.py` | `@06`, `06-conformal` |
| **Active Learning** | `active_learning/{query,triage_queue}.py` | `unit/test_active_learning.py` | `@07`, `06-conformal` |
| **Monitoring** | `monitoring/{telemetry,drift}.py` | `unit/test_monitoring.py` | `@08`, `02-latency` |
| **FastAPI & HTMX** | `service/{api,inference,schemas,storage}.py`, `service/routes/*.py` | `integration/test_api_endpoints.py`, `integration/test_views_and_routes.py` | `@09`, `02-latency` |
| **Hardware & Infra** | `deploy/`, `Justfile`, `docs/latex/` | `invariants/test_hardware_invariants.py` | `@11`, `07-hardware` |

## Token-Optimal Execution Protocol

* **Read Scope Guardrail:** Read ONLY the files mapped to your active task row. Never scan `openwiki/.claims/`, `pixi.lock`, `dvc.lock`, or `*.pdf`.
* **Fast Targeted Verify:** During iteration, execute fast test sweeps suppressing global coverage:
    `pixi run -e ci-dev pytest <targeted_test_file> -q -o "addopts="`
* **Full Quality Gate:** Run the comprehensive test suite and gates once upon completing code changes:
    `pixi run -e ci-dev ruff check . && pixi run -e ci-dev mypy src scripts && pixi run -e ci-dev python scripts/governance/validate_okf.py && pixi run -e ci-dev python -m taxon_vision.pipelines.audit_license`

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

<!-- OPENWIKI:START -->

## OpenWiki

This repository has a generated `openwiki/` evidence index. It is optional just-in-time context, not required startup reading.

* Do not enumerate, preload, or search wikis at task start. Use retrieval when the user asks for it, when unfamiliar architecture or dependency behavior materially affects the task, or when source inspection leaves an important uncertainty. Stop once the question is grounded.
* When those conditions apply and OpenWiki retrieval tools are available, use `openwiki_search` for just-in-time context and `openwiki_read` for the relevant complete sections. If search returns `workspace_required`, ask which listed workspace to use and retry with its ID.
* Use `openwiki_list_workspaces` or `openwiki_list_wikis` when workspace membership itself needs to be discovered.
* If the retrieval tools are unavailable, read `openwiki/quickstart.md` and follow its links to the relevant pages.
* Treat source code and tests as authoritative. A brief's unknowns and review items are verification gaps, not automatic requirements.
* Prefer the narrowest quiet validation that proves the changed behavior. Preserve complete failure output.

The scheduled OpenWiki GitHub Actions workflow refreshes the repository wiki. Do not hand-edit generated OpenWiki pages unless explicitly asked; prefer updating source code/docs and letting OpenWiki regenerate.

<!-- OPENWIKI:END -->
