#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Benjamin Förster
# SPDX-License-Identifier: MIT
#
# Rehearses GitHub Actions CI/CD workflows locally using nektos/act with Docker or Podman.

set -euo pipefail

usage() {
  cat <<'EOF'
Usage: ./scripts/run_ci_with_act.sh [--dryrun] [--job <job>]

Rehearse the GitHub Actions workflow locally with 'act'.

Examples:
  ./scripts/run_ci_with_act.sh --dryrun
  ./scripts/run_ci_with_act.sh --job quality-gate
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if ! command -v act >/dev/null 2>&1; then
  echo "The 'act' CLI is not installed or not on PATH." >&2
  exit 1
fi

act "$@"
