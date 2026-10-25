# ==============================================================================
# TaxonVision-MLOps Automation Center
# ==============================================================================
set shell := ["bash", "-uc"]

# Show available recipes and descriptions
default:
	@just --list

# ── Setup & Bootstrapping ───────────────────────────────────────────────────

# Bootstrap dev environment with Pixi and pre-commit hooks (pass --scratch for Jupyter)
[group("setup")]
setup mode="":
	@if [ "{{mode}}" = "--scratch" ]; then \
		just _setup-core; \
		just _setup-scratch; \
	elif [ -n "{{mode}}" ]; then \
		echo "Unknown setup flag: {{mode}}. Supported: --scratch"; \
		exit 1; \
	else \
		just _setup-core; \
	fi

# Bootstrap scratch experimentation workspace with isolated git repo and kernel
[group("setup")]
setup-scratch:
	@just setup --scratch

[private]
_setup-core:
	pixi install -e dev
	pixi run --frozen -e dev pre-commit install

[private]
_setup-scratch:
	@echo "==> Configuring scratch experimentation repository..."
	@mkdir -p scratch/notebooks
	@if [ ! -d "scratch/.git" ]; then \
		git -C scratch init; \
		printf "# TaxonVision Scratch & Experimentation\n\nLocal isolated workspace for interactive notebooks.\n" > scratch/README.md; \
		printf ".ipynb_checkpoints/\n__pycache__/\n*.pyc\n" > scratch/.gitignore; \
		echo "*.ipynb filter=nbstripout" > scratch/.gitattributes; \
		git -C scratch config filter.nbstripout.clean "pixi run -e dev nbstripout"; \
		git -C scratch config filter.nbstripout.smudge cat; \
	fi
	@pixi run --frozen -e dev python -m ipykernel install --user --name=taxon-scratch --display-name="TaxonVision (Scratch)"
	@echo "==> Scratch environment ready at ./scratch. Launch with: just lab"

# Enter activated Pixi development shell
[group("setup")]
init:
	pixi shell --manifest-path ./pyproject.toml -e dev

# ── Quality & Gates ─────────────────────────────────────────────────────────

# Run Ruff linting, formatting, and MyPy strict type checking
[group("quality")]
lint:
	pixi run --frozen -e dev ruff check --fix .
	pixi run --frozen -e dev ruff format .
	pixi run --frozen -e dev mypy src scripts

# Run all pre-commit quality hooks across the repository
[group("quality")]
check:
	pixi run --frozen -e dev pre-commit run --all-files

# Format all Python code with Ruff
[group("quality")]
format:
	pixi run --frozen -e dev ruff format .

# Check cyclomatic and cognitive complexity metrics with complexipy
[group("quality")]
complexity:
	pixi run --frozen -e dev complexipy src/ --failed

# Lint Markdown documents with markdownlint-cli
[group("quality")]
lint-markdown:
	markdownlint --config config/lint/.markdownlint.yaml --fix "docs/**/*.md" "README.md" ".agents/**/*.md"

# Validate Open Knowledge Format (OKF v0.2) frontmatter and link topology
[group("quality")]
validate-okf:
	pixi run --frozen -e dev python scripts/governance/validate_okf.py

# Verify and fix Markdown Rule 04 formatting invariants
[group("quality")]
lint-md:
	pixi run --frozen -e dev python scripts/governance/verify_markdown_rules.py --fix

# Verify Rule 07 Hardware Invariants across documentation and LaTeX
[group("quality")]
check-hardware:
	pixi run --frozen -e dev python scripts/governance/verify_hardware_invariants.py

# ── Testing ─────────────────────────────────────────────────────────────────

# Run complete test suite with coverage reporting
[group("testing")]
test:
	pixi run --frozen -e dev pytest

# Run unit tests only
[group("testing")]
test-unit:
	pixi run --frozen -e dev pytest tests/unit/

# Run integration tests only
[group("testing")]
test-integration:
	pixi run --frozen -e dev pytest tests/integration/

# Run conformal prediction mathematical coverage invariant assertions
[group("testing")]
test-conformal:
	pixi run --frozen -e dev pytest tests/invariants/test_conformal_coverage_invariant.py -v

# Run local CI pass without Docker/GPU (options: all, quality, tests, docs)
[group("testing")]
ci job="all":
	./scripts/ci/local_ci.sh {{job}}

# Run local CI test pass (alias for 'ci tests')
[group("testing")]
ci-test:
	./scripts/ci/local_ci.sh tests

# Fast static contract & syntax validation (< 3 seconds, zero GPU/network required)
[group("quality")]
verify-contracts:
	@echo ">>> 1. Checking shell script syntax..."
	bash -n scripts/ci/jenkins_gpu_train.sh
	bash -n scripts/ci/verify_deployment.sh
	bash -n scripts/ci/local_ci.sh
	@echo ">>> 2. Validating DVC DAG structure & dry-run execution plan..."
	pixi run --frozen -e ci-dev dvc dag
	pixi run --frozen -e ci-dev dvc repro --dry
	@if command -v kubectl >/dev/null 2>&1; then \
		echo ">>> 3. Validating Kubernetes manifests with kubectl..."; \
		kubectl apply -f deploy/k8s/ --dry-run=client; \
	else \
		echo ">>> 3. kubectl not found on workstation (skipping client-side K8s manifest validation)."; \
	fi

# Render visual DVC pipeline Mermaid graph to inspect stage boundaries
[group("mlops")]
visualize-dag:
	pixi run --frozen -e ci-dev dvc dag --mermaid

# Micro-canary end-to-end pipeline verification (tests ONNX export, INT8 quantization, and conformal calibration offline)
[group("testing")]
test-canary:
	@echo ">>> Running offline micro-canary verification..."
	TAXON_ALLOW_UNTRAINED_FALLBACK=true pixi run --frozen -e ci-dev pytest tests/integration/ tests/invariants/ -v -o "addopts="

# ── MLOps & Pipelines ───────────────────────────────────────────────────────

# Stream sample image batch from S3 into local cache for inspection
[group("mlops")]
fetch-sample:
	pixi run --frozen -e dev python -m taxon_vision.data.s3_streamer --limit 100 --out data/sample/

# Ingest citizen-science observations from iNaturalist to Parquet manifest (Stage 1)
[group("mlops")]
ingest-data:
	pixi run --frozen -e dev python -m taxon_vision.pipelines.ingest

# Audit open licenses and filter compliant observations to dataset manifest (Stage 2)
[group("mlops")]
audit-license:
	pixi run --frozen -e dev python -m taxon_vision.pipelines.audit_license

# Stream representations from S3 open data into 4-way embedding split (Stage 3)
[group("mlops")]
extract-features:
	pixi run --frozen -e dev python -m taxon_vision.pipelines.extract_features

# Train baseline classification head with Class-Balanced Loss (Stage 4)
[group("mlops")]
train epochs="5" extractor="dinov3":
	@TOKEN=$(secret-tool lookup Title DagsHub 2>/dev/null || secret-tool lookup service dagshub 2>/dev/null || true); \
	if [ -n "$TOKEN" ]; then \
		export MLFLOW_TRACKING_USERNAME="foersben"; \
		export MLFLOW_TRACKING_PASSWORD="$TOKEN"; \
	fi; \
	pixi run --frozen -e dev python -m taxon_vision.pipelines.train --epochs {{epochs}} --extractor {{extractor}}

# Export trained classification head to ONNX computational graph (Stage 5)
[group("mlops")]
export-onnx:
	pixi run --frozen -e dev python -m taxon_vision.pipelines.export_onnx

# Calibrate Split Conformal Prediction quantiles and OOD energy thresholds (Stage 6)
[group("mlops")]
calibrate:
	pixi run --frozen -e dev python -m taxon_vision.pipelines.calibrate

# Evaluate Pareto trade-offs and holdout test set with conformal bounds (Stage 7)
[group("mlops")]
pareto:
	pixi run --frozen -e dev python -m taxon_vision.pipelines.evaluate

# Run CLI prediction script on an observation image
[group("mlops")]
predict image="data/sample/406185885_taxon_47120.jpg":
	pixi run --frozen -e dev python -m taxon_vision.inference.cli {{image}}

# Download verified open-license sample images for all 10 catalog taxa from iNaturalist
[group("mlops")]
download-samples per_taxon="1":
	pixi run --frozen -e dev python scripts/data/download_samples.py --per-taxon {{per_taxon}}

# Push tracked datasets & models to DagsHub or local remote
[group("mlops")]
dvc-push remote="origin":
	@TOKEN=$(secret-tool lookup Title DagsHub 2>/dev/null || secret-tool lookup service dagshub 2>/dev/null); \
	if [ -z "$TOKEN" ] && [ "{{remote}}" = "origin" ]; then \
		echo "❌ Error: KeePassXC is locked or DagsHub token not found in Secret Service." >&2; \
		exit 1; \
	fi; \
	AWS_ACCESS_KEY_ID="$TOKEN" AWS_SECRET_ACCESS_KEY="$TOKEN" \
	pixi run --frozen -e dev dvc push -r {{remote}}

# Pull tracked datasets & models from DagsHub or local remote
[group("mlops")]
dvc-pull remote="origin":
	@TOKEN=$(secret-tool lookup Title DagsHub 2>/dev/null || secret-tool lookup service dagshub 2>/dev/null); \
	if [ -z "$TOKEN" ] && [ "{{remote}}" = "origin" ]; then \
		echo "❌ Error: KeePassXC is locked or DagsHub token not found in Secret Service." >&2; \
		exit 1; \
	fi; \
	AWS_ACCESS_KEY_ID="$TOKEN" AWS_SECRET_ACCESS_KEY="$TOKEN" \
	pixi run --frozen -e dev dvc pull -r {{remote}}

# Display DVC pipeline Directed Acyclic Graph
[group("mlops")]
dvc-dag:
	PAGER=cat pixi run --frozen -e dev dvc dag

# Reproduce DVC pipeline Directed Acyclic Graph
[group("mlops")]
dvc-repro:
	pixi run --frozen -e dev dvc repro

# ── Serving & Web UI ────────────────────────────────────────────────────────

# Start FastAPI inference and HTMX dashboard server with live reload
[group("app")]
run-api:
	pixi run --frozen -e dev uvicorn taxon_vision.service.api:app --host 0.0.0.0 --port 8000 --reload

# Start web service (alias for run-api)
[group("app")]
run:
	@just run-api

# Validate live Kubernetes deployment health, endpoints, and inference
[group("app")]
verify-deployment namespace="taxon-vision":
	@bash scripts/ci/verify_deployment.sh {{namespace}}

# ── Documentation ───────────────────────────────────────────────────────────

# Build Zensical documentation site
[group("docs")]
docs:
	pixi run --frozen -e dev python scripts/governance/visualize_okf.py
	pixi run --frozen -e dev zensical build

# Build and serve Zensical documentation locally on port 9000
[group("docs")]
serve:
	pixi run --frozen -e dev python scripts/governance/visualize_okf.py
	pixi run --frozen -e dev zensical build
	pixi run --frozen -e dev zensical serve -a localhost:9000

# Generate interactive D3/SVG knowledge graph visualization
[group("docs")]
visualize-okf:
	pixi run --frozen -e dev python scripts/governance/visualize_okf.py

# Update OpenWiki documentation and reconcile grounded claims
[group("docs")]
wiki-update:
    @echo "Fetching credentials from KeePassXC via Secret Service..."
    OPENWIKI_PROVIDER="gemini" \
    OPENWIKI_MODEL_ID="gemini-3.8-flash" \
    GOOGLE_API_KEY=$(secret-tool lookup service openwiki key google_api_key) \
    LANGCHAIN_API_KEY=$(secret-tool lookup service openwiki key langsmith_api_key) \
    LANGSMITH_API_KEY=$(secret-tool lookup service openwiki key langsmith_api_key) \
    LANGCHAIN_TRACING_V2="true" \
    openwiki code --update

# Validate OKF v0.2 structural integrity for .agents/ and openwiki/
[group("docs")]
wiki-validate:
	pixi run --frozen -e dev python scripts/governance/validate_okf.py

# Launch local interactive visualizer for the OKF knowledge graph
[group("docs")]
wiki-viz:
	pixi run --frozen -e dev python scripts/governance/visualize_okf.py



# ── Utilities ───────────────────────────────────────────────────────────────

# Launch JupyterLab within the isolated scratch experimentation workspace
[group("utils")]
lab:
	pixi run --frozen -e dev jupyter lab --notebook-dir=scratch --ip=127.0.0.1 --port=8888

# Clean build artifacts, caches, and ephemeral files
[group("utils")]
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf .cache site build dist .pytest_cache .mypy_cache .ruff_cache htmlcov .pixi .venv uv.lock
