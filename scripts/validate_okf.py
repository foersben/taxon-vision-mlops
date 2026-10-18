#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Benjamin Förster
# SPDX-License-Identifier: EUPL-1.2 OR LicenseRef-PHIDS-Commercial

"""Backward-compatible shim delegating to scripts/governance/validate_okf.py."""

import runpy
from pathlib import Path

if __name__ == "__main__":
    target = Path(__file__).resolve().parent / "governance" / "validate_okf.py"
    runpy.run_path(str(target), run_name="__main__")
