#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Benjamin Förster
# SPDX-License-Identifier: MIT
# Execute pre-commit tools inside the active Pixi environment without nested uv lock contention.

set -euo pipefail

if [ -n "${PIXI_ENVIRONMENT_NAME:-}" ]; then
  exec "$@"
fi

exec pixi run --frozen -e dev "$@"
