#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 Benjamin Förster
# SPDX-License-Identifier: MIT
"""CLI shim for Stage 2: License compliance audit & active gatekeeper.

Delegates directly to canonical implementation in taxon_vision.pipelines.audit_license.
"""

from __future__ import annotations

import sys

from taxon_vision.pipelines.audit_license import (
    ALLOWED_LICENSES,
    audit_license_cli,
    main,
    verify_observation_compliance,
)

__all__ = ["ALLOWED_LICENSES", "audit_license_cli", "main", "verify_observation_compliance"]

if __name__ == "__main__":
    sys.exit(audit_license_cli())
