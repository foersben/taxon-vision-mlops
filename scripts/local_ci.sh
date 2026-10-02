#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Benjamin Förster
# SPDX-License-Identifier: MIT
#
# Runs TaxonVision continuous integration passes locally on the native host interpreter
# without container overhead (Docker/act). Replicates CI quality, test, and doc builds.

set -euo pipefail

usage() {
  cat <<'EOF'
Usage: ./scripts/local_ci.sh [quality|tests|docs|all]

Runs the main TaxonVision CI commands on the current local interpreter.

Examples:
  ./scripts/local_ci.sh quality
  ./scripts/local_ci.sh tests
  ./scripts/local_ci.sh docs
  ./scripts/local_ci.sh all

Environment variables:
  TAXON_SKIP_SYNC=1  Skip 'pixi install' before running checks.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

job="${1:-all}"

run_sync() {
  if [[ "${TAXON_SKIP_SYNC:-0}" == "1" ]]; then
    echo ">>> Skipping dependency sync because TAXON_SKIP_SYNC=1"
    return
  fi

  echo ">>> Syncing dependencies with Pixi"
  pixi install -e dev
}

run_quality() {
  echo ">>> Running Ruff lint"
  pixi run --frozen -e dev ruff check .
  echo ">>> Running Ruff format check"
  pixi run --frozen -e dev ruff format --check .
  echo ">>> Running Mypy"
  pixi run --frozen -e dev mypy src scripts
}

run_tests() {
  echo ">>> Pass 1: Unit Tests & Coverage"
  pixi run --frozen -e dev pytest tests/unit/ --cov=src/taxon_vision --cov-fail-under=80

  echo ">>> Pass 2: Integration & Invariants"
  pixi run --frozen -e dev pytest tests/integration/ tests/invariants/ -x -q -o "addopts="

  echo ">>> Open License Compliance Audit"
  pixi run --frozen -e dev python scripts/audit_license_compliance.py
}

run_docs() {
  echo ">>> Validating OKF Frontmatter and Links"
  pixi run --frozen -e dev python scripts/validate_okf.py
  echo ">>> Building docs in strict mode"
  pixi run --frozen -e dev zensical build --strict
}

case "$job" in
  quality)
    run_sync
    run_quality
    ;;
  tests)
    run_sync
    run_tests
    ;;
  docs)
    run_sync
    run_docs
    ;;
  all)
    run_sync
    run_quality
    run_tests
    run_docs
    ;;
  *)
    echo "Unknown job: $job" >&2
    usage >&2
    exit 1
    ;;
esac
