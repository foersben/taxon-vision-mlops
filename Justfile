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

# Validate Open Knowledge Format (OKF v0.2) frontmatter and link topology
[group("quality")]
validate-okf:
	pixi run --frozen -e dev python scripts/validate_okf.py

# Verify and fix Markdown Rule 04 formatting invariants
[group("quality")]
lint-md:
	pixi run --frozen -e dev python scripts/verify_markdown_rules.py --fix

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
	./scripts/local_ci.sh {{job}}

# Run local CI test pass (alias for 'ci tests')
[group("testing")]
ci-test:
	./scripts/local_ci.sh tests

# ── MLOps & Pipelines ───────────────────────────────────────────────────────

# Stream sample image batch from S3 into local cache for inspection
[group("mlops")]
fetch-sample:
	pixi run --frozen -e dev python -m taxon_vision.data.s3_streamer --limit 100 --out data/sample/

# Ingest citizen-science observations from iNaturalist to Parquet manifest
[group("mlops")]
ingest-data:
	pixi run --frozen -e dev python scripts/ingest_data.py

# Train baseline classification head with Class-Balanced Loss
[group("mlops")]
train-baseline extractor="bioclip-2":
	pixi run --frozen -e dev python -m taxon_vision.models.trainer --extractor {{extractor}} --epochs 5

# Run CLI training script (Phase 1 Baseline)
[group("mlops")]
train epochs="5" extractor="mobilenetv4_conv_small":
	pixi run --frozen -e dev python -m taxon_vision.models.trainer --epochs {{epochs}} --extractor {{extractor}}

# Run CLI prediction script on an observation image
[group("mlops")]
predict image="data/sample/406185885_taxon_47120.jpg":
	pixi run --frozen -e dev python -m taxon_vision.inference.cli {{image}}



# Export trained PyTorch model to ONNX computational graph
[group("mlops")]
export-onnx:
	pixi run --frozen -e dev python -m taxon_vision.inference.onnx_exporter

# Evaluate Pareto trade-offs (Accuracy vs Latency vs Cost) across backbones
[group("mlops")]
pareto:
	pixi run --frozen -e dev python scripts/benchmark_pareto.py

# Calibrate Split Conformal Prediction quantiles on holdout validation data
[group("mlops")]
calibrate:
	pixi run --frozen -e dev python scripts/calibrate_conformal.py --alpha 0.05

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
	@bash scripts/verify_deployment.sh {{namespace}}

# ── Documentation ───────────────────────────────────────────────────────────

# Build Zensical documentation site
[group("docs")]
docs:
	pixi run --frozen -e dev zensical build

# Build and serve Zensical documentation locally on port 9000
[group("docs")]
serve:
	pixi run --frozen -e dev zensical build
	pixi run --frozen -e dev zensical serve -a localhost:9000

# Generate interactive D3/SVG knowledge graph visualization
[group("docs")]
visualize-okf:
	pixi run --frozen -e dev python scripts/visualize_okf.py

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
