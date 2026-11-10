# TaxonVision-MLOps Master Agent Router

<!-- UNIVERSAL RULES: apply to all AI agents -->

* All Python commands MUST use `pixi run -e dev` or `just`. Never bare `python`, `pip`, `uv`, `uvx`, or `poetry`. Never create or rely on `.venv/`.
* Never run `act` in agent sandboxes; `act` is for local workstation Docker runs only.
* Commit signing: Commits MUST be GPG/SSH signed (`git commit -S`). Stop and escalate if signing fails.
* Pre-commit gates (run before ANY commit to `src/taxon_vision/` or `config/`):

```bash
pixi run -e dev ruff check . && pixi run -e dev mypy src scripts
pixi run -e dev python -m taxon_vision.pipelines.audit_license
pixi run -e dev python scripts/governance/validate_okf.py
pixi run -e dev python scripts/governance/verify_hardware_invariants.py
```

## Core Architecture Constraints

* **Dual-Target Toolchain:** All development and testing execute via `pixi run -e dev` or `just`. CI runs on CPU via `ci-dev`.
* **Zero-Allocation Inference:** ONNX Runtime inference hot paths must avoid dynamic memory allocation and Python object boxing.
* **Strict Open Licensing:** Only observations under open licenses (CC0, CC-BY, CC-BY-NC) are admitted into training datasets. Attribution metadata must be retained.
* **Conformal Coverage Guarantee:** Split Conformal Prediction must mathematically guarantee user-specified coverage ($1 - \alpha$). Observations exceeding uncertainty bounds or empty prediction sets route to human triage.
* **Epistemic Diamond DAG:** 7 canonical pipeline stages in `src/taxon_vision/pipelines/` governed by root `params.yaml`: `raw_manifest.parquet` -> `dataset_manifest.parquet` -> 4 split tensors (`train_embeddings.pt`, `val_embeddings.pt`, `cal_embeddings.pt`, `test_embeddings.pt`) -> `head.pt` -> `model_int8.onnx` -> `calibrate` & `eval`.
* **Hardware Invariants:** Hardware specifications must strictly adhere to Rule 07 (`.agents/rules/07-hardware-invariants.md`). State only verified parameters (Intel i7-14700K, RTX 5070 Ti 16GB, 128GB DDR5, 2TB NVMe BTRFS zstd:3 non-CoW).
* **Clean Code & Invariants:** Code must conform to Rule 08 (`.agents/rules/08-clean-code-and-antipatterns.md`), enforcing 4-part architectural docstrings, typed parameter models (no tuple blindness), async event-loop non-blocking I/O, and RAII lifecycle management.
* **Unified Web Service:** FastAPI powers both the machine REST API and the server-rendered Jinja2/HTMX operator dashboard.
* **Editorial Idiom & British English:** All generated documentation, docstrings, commit messages, and reports must adhere to Rule 04 (`.agents/rules/04-markdown-formatting.md`), employing British English orthography (`optimise`, `analyse`, `behaviour`, `modelling`, `artefact`) and an analytical, pragmatic German-native idiom (structured causal transitions, smell-oriented optimality auditing, and hyphen-delimited qualification clauses without colloquial marketing fluff).

## Fast Module & Test Routing Map

Read ONLY the files in your active task row:

| Domain | Primary Source Files (`src/taxon_vision/`) | Targeted Test (`tests/`) | Role & Rule |
| :--- | :--- | :--- | :--- |
| **Config & Types** | `config.py`, `params.yaml`, `config/default_config.yaml` | `unit/test_config.py` | `@01`, `00-python` |
| **Data & S3** | `data/{s3_streamer,inat_client,license_filter}.py` | `unit/test_data_components.py` | `@03`, `05-license` |
| **Pipelines** | `pipelines/{ingest,audit_license,extract_features,train,export_onnx,calibrate,evaluate}.py` | `unit/test_pipelines.py` | `@03`, `01-mlops` |
| **Models & Loss** | `models/{factory,head,loss,trainer}.py` | `unit/test_models_components.py` | `@04`, `01-mlops` |
| **Training & HPO** | `models/training/{runner,loops,embeddings,tuning,tracking,checkpoint,types}.py` | `unit/test_tuning.py`, `unit/test_tracking.py` | `@04`, `01-mlops`, `08-clean-code` |
| **ONNX & CAM** | `inference/{onnx_exporter,onnx_engine,quantizer,cam,calibration,cli}.py` | `unit/test_onnx_exporter.py`, `unit/test_quantizer.py` | `@05`, `02-latency` |
| **OOD & Conformal** | `uncertainty/{ood_detector,conformal}.py` | `unit/test_ood_energy.py`, `unit/test_conformal_bounds.py` | `@06`, `06-conformal` |
| **Active Learning** | `active_learning/{query,triage_queue}.py` | `unit/test_active_learning.py` | `@07`, `06-conformal` |
| **Monitoring** | `monitoring/{telemetry,drift}.py` | `unit/test_monitoring.py` | `@08`, `02-latency`, `08-clean-code` |
| **FastAPI & HTMX** | `service/{api,inference,schemas,storage}.py`, `service/routes/*.py` | `integration/test_api_endpoints.py`, `integration/test_views_and_routes.py` | `@09`, `02-latency`, `08-clean-code` |
| **Hardware & Infra** | `deploy/`, `Justfile`, `docs/latex/` | `invariants/test_hardware_invariants.py` | `@11`, `07-hardware` |

## Token-Optimal Execution Protocol

* **Read Scope Guardrail:** Read ONLY the files mapped to your active task row. Never scan `openwiki/.claims/`, `pixi.lock`, `dvc.lock`, or `*.pdf`.
* **Fast Targeted Verify:** During iteration, execute fast test sweeps suppressing global coverage:
    `pixi run -e ci-dev pytest <targeted_test_file> -q -o "addopts="`
* **Full Quality Gate:** Run the comprehensive test suite and gates once upon completing code changes:
    `pixi run -e ci-dev ruff check . && pixi run -e ci-dev mypy src scripts && pixi run -e ci-dev python scripts/governance/validate_okf.py && pixi run -e ci-dev python -m taxon_vision.pipelines.audit_license`

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
