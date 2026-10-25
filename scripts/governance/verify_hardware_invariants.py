#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Benjamin Förster
# SPDX-License-Identifier: MIT
"""Hardware Invariant Governance Enforcer (Rule 07).

Scans Markdown documents, LaTeX sources, and TikZ diagrams across the
repository to ensure zero mentions of prohibited PC-builder trivia (e.g.
motherboard models, GPU sub-vendors, RAM slots, SSD controllers) and zero
unscientific claims regarding public cloud volatility.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import NamedTuple

IGNORE_DIR_PATTERNS = (".pixi", "site", ".cache", ".archive", "openwiki", ".git", "scratch")

# Allowed files that mention prohibited terms strictly as part of governance policy
EXEMPT_FILES = {
    Path(".agents/rules/07-hardware-invariants.md"),
    Path("scripts/governance/verify_hardware_invariants.py"),
    Path("tests/invariants/test_hardware_invariants.py"),
}


class Violation(NamedTuple):
    """Represents a hardware invariant violation found in a file."""

    file_path: Path
    line_number: int
    matched_term: str
    line_content: str
    remediation: str


PROHIBITED_RULES: list[tuple[re.Pattern[str], str]] = [
    (
        re.compile(r"\bASRock\b", re.IGNORECASE),
        "Omit motherboard brand. State CPU (Intel Core i7-14700K) directly.",
    ),
    (
        re.compile(r"\bB760M?\b", re.IGNORECASE),
        "Omit motherboard chipset. Architectural logic only requires CPU core counts.",
    ),
    (
        re.compile(r"\bZOTAC\b", re.IGNORECASE),
        "Omit GPU sub-vendor. State 'NVIDIA GeForce RTX 5070 Ti' directly.",
    ),
    (
        re.compile(r"\bFireCuda\b", re.IGNORECASE),
        "Omit SSD brand. State '2 TB PCIe Gen4 NVMe SSD' directly.",
    ),
    (
        re.compile(r"\bPhison\b", re.IGNORECASE),
        "Omit storage controller chip model.",
    ),
    (
        re.compile(r"\bPS5018\b", re.IGNORECASE),
        "Omit storage controller chip model.",
    ),
    (
        re.compile(r"\b2[,.]?550\s*TBW\b", re.IGNORECASE),
        "Omit consumer SSD endurance ratings.",
    ),
    (
        re.compile(r"\bM2_1\b", re.IGNORECASE),
        "Omit motherboard M.2 slot identifier.",
    ),
    (
        re.compile(r"\b(slots?\s+A2\s+and\s+B2|slot\s+A2|slot\s+B2)\b", re.IGNORECASE),
        "Omit RAM slot topology. State '128 GB DDR5' system memory.",
    ),
    (
        re.compile(r"\bCrucial\s+DDR5\b", re.IGNORECASE),
        "Omit RAM sub-vendor. State '128 GB DDR5' system memory.",
    ),
    (
        re.compile(r"\bvolatile\s+cloud\b", re.IGNORECASE),
        "Reframe cloud rationale around TCO, zero egress, and kernel sovereignty.",
    ),
    (
        re.compile(r"\bhypervisor\s+(?:virtualis|virtualiz)ation\s+overhead\b", re.IGNORECASE),
        "Reframe bare-metal deployment around unit economics and kernel-level resource control.",
    ),
    (
        re.compile(r"\bhypervisor\s+jitter\b", re.IGNORECASE),
        "Reframe bare-metal deployment around unit economics and zero-egress data transfer.",
    ),
]


def scan_file(file_path: Path, root_path: Path) -> list[Violation]:
    """Scan a single file against prohibited hardware patterns."""
    rel_path = file_path.relative_to(root_path)
    if rel_path in EXEMPT_FILES:
        return []

    violations: list[Violation] = []
    try:
        content = file_path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return []

    for idx, line in enumerate(content.splitlines(), start=1):
        for pattern, remediation in PROHIBITED_RULES:
            match = pattern.search(line)
            if match:
                violations.append(
                    Violation(
                        file_path=rel_path,
                        line_number=idx,
                        matched_term=match.group(0),
                        line_content=line.strip(),
                        remediation=remediation,
                    )
                )
    return violations


def collect_target_files(root_path: Path) -> list[Path]:
    """Collect Markdown, LaTeX, and TikZ files for invariant checking."""
    target_extensions = {".md", ".tex", ".tikz"}
    collected: list[Path] = []
    for path in root_path.rglob("*"):
        if not path.is_file() or path.suffix not in target_extensions:
            continue
        rel = path.relative_to(root_path)
        if any(ignored in rel.parts for ignored in IGNORE_DIR_PATTERNS):
            continue
        collected.append(path)
    return sorted(collected)


def run_check(root_path: Path) -> int:
    """Run hardware invariant check across repository files."""
    target_files = collect_target_files(root_path)
    total_violations: list[Violation] = []

    for file_path in target_files:
        total_violations.extend(scan_file(file_path, root_path))

    if not total_violations:
        print(f"✅ Rule 07 Hardware Invariants PASSED across {len(target_files)} scanned files.")
        return 0

    print("=" * 70)
    print(f"❌ RULE 07 VIOLATIONS DETECTED: {len(total_violations)} issue(s) found")
    print("=" * 70)
    for v in total_violations:
        print(f"\n[{v.file_path}:{v.line_number}] Prohibited token: '{v.matched_term}'")
        print(f"  Line: {v.line_content}")
        print(f"  Remediation: {v.remediation}")
    print("\n" + "=" * 70)
    return 1


def main() -> None:
    """CLI entrypoint for hardware invariant governance."""
    parser = argparse.ArgumentParser(description="Enforce Rule 07 Hardware Invariants.")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
        help="Root path of the repository.",
    )
    args = parser.parse_args()
    sys.exit(run_check(args.root))


if __name__ == "__main__":
    main()
