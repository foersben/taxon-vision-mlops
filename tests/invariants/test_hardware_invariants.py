# SPDX-FileCopyrightText: 2026 Benjamin Förster
# SPDX-License-Identifier: MIT
"""Scientific and architectural invariants verifying hardware representations and Rule 07 compliance."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "governance" / "verify_hardware_invariants.py"


def test_rule_07_hardware_invariants_pass() -> None:
    """Invariant: All Markdown, LaTeX, and TikZ files must adhere to Rule 07 hardware standards."""
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--root", str(ROOT)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, f"Hardware invariant violations detected:\n{result.stdout}\n{result.stderr}"


def test_rule_07_governance_file_exists_and_registered() -> None:
    """Invariant: Rule 07 must exist and be registered in agent routers."""
    rule_file = ROOT / ".agents" / "rules" / "07-hardware-invariants.md"
    assert rule_file.exists(), f"Missing rule file at {rule_file}"

    agents_md = ROOT / ".agents" / "AGENTS.md"
    assert agents_md.exists()
    content = agents_md.read_text(encoding="utf-8")
    assert "07-hardware-invariants.md" in content, "Rule 07 not registered in .agents/AGENTS.md"
    assert "Hardware Invariants" in content, "Hardware Invariants constraint missing from .agents/AGENTS.md"
