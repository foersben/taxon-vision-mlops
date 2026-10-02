# ==============================================================================
# TaxonVision-MLOps Automation Center
# ==============================================================================
set shell := ["bash", "-uc"]

default:
	@just --list

# ── Setup & Bootstrapping ───────────────────────────────────────────────────

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

[group("setup")]
init:
	pixi shell --manifest-path ./pyproject.toml -e dev

# ── Quality & Gates ─────────────────────────────────────────────────────────

[group("quality")]
lint:
	pixi run --frozen -e dev ruff check --fix .
	pixi run --frozen -e dev ruff format .
	pixi run --frozen -e dev mypy src scripts

[group("quality")]
check:
	pixi run --frozen -e dev pre-commit run --all-files

[group("quality")]
format:
	pixi run --frozen -e dev ruff format .

[group("quality")]
complexity:
	pixi run --frozen -e dev complexipy src/ --failed

[group("quality")]
validate-okf:
	pixi run --frozen -e dev python scripts/validate_okf.py

[group("quality")]
lint-md:
	pixi run --frozen -e dev python scripts/verify_markdown_rules.py --fix

# ── Testing ─────────────────────────────────────────────────────────────────

[group("testing")]
test:
	pixi run --frozen -e dev pytest

[group("testing")]
test-unit:
	pixi run --frozen -e dev pytest tests/unit/

[group("testing")]
test-integration:
	pixi run --frozen -e dev pytest tests/integration/

[group("testing")]
test-conformal:
	pixi run --frozen -e dev pytest tests/invariants/test_conformal_coverage_invariant.py -v

[group("testing")]
ci-test:
	./scripts/local_ci.sh tests

# ── MLOps & Pipelines ───────────────────────────────────────────────────────

[group("mlops")]
fetch-sample:
	pixi run --frozen -e dev python -m taxon_vision.data.s3_streamer --limit 100 --out data/sample/

[group("mlops")]
train-baseline extractor="bioclip-2":
	pixi run --frozen -e dev python -m taxon_vision.models.trainer --extractor {{extractor}} --epochs 5

[group("mlops")]
export-onnx:
	pixi run --frozen -e dev python -m taxon_vision.inference.onnx_exporter

[group("mlops")]
pareto:
	pixi run --frozen -e dev python scripts/benchmark_pareto.py

[group("mlops")]
calibrate:
	pixi run --frozen -e dev python scripts/calibrate_conformal.py --alpha 0.05

# ── Serving & Web UI ────────────────────────────────────────────────────────

[group("app")]
run-api:
	pixi run --frozen -e dev uvicorn taxon_vision.service.api:app --host 0.0.0.0 --port 8000 --reload

[group("app")]
run:
	@just run-api

# ── Documentation ───────────────────────────────────────────────────────────

[group("docs")]
docs:
	pixi run --frozen -e dev zensical build

[group("docs")]
serve:
	pixi run --frozen -e dev zensical build
	pixi run --frozen -e dev zensical serve -a localhost:9000

[group("docs")]
visualize-okf:
	pixi run --frozen -e dev python scripts/visualize_okf.py

# ── Utilities ───────────────────────────────────────────────────────────────

[group("utils")]
lab:
	pixi run --frozen -e dev jupyter lab --notebook-dir=scratch --ip=127.0.0.1 --port=8888

[group("utils")]
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf .cache site build dist .pytest_cache .mypy_cache .ruff_cache htmlcov .pixi
