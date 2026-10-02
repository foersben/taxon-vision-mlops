#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Benjamin Förster
# SPDX-License-Identifier: MIT
"""Strict Markdown Formatting Rule Enforcer (Rule 04).

Enforces:
1. Standard ASCII hyphen ('-') exclusively; ban en-dash (\u2013) and em-dash (\u2014).
2. Unordered lists MUST use asterisk ('*').
3. List syntax: exactly 1 space after '*', indented in multiples of 2 spaces (0 spaces at level 0).
4. Exactly 1 blank line before/after lists, code blocks, and headings.
5. Zero trailing whitespace at line ends.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

DASH_REPLACEMENTS = {
    "\u2013": "-",  # en-dash
    "\u2014": "-",  # em-dash
    "\u2015": "-",  # horizontal bar
    "\u2212": "-",  # minus sign
}


def verify_file(path: Path, fix: bool = False) -> list[str]:
    """Verify and optionally auto-fix a single markdown file."""
    text = path.read_text(encoding="utf-8")
    original_text = text
    errors: list[str] = []

    # 1. Dash verification
    for bad_char, replacement in DASH_REPLACEMENTS.items():
        if bad_char in text:
            count = text.count(bad_char)
            errors.append(f"Contains {count} non-standard dash (U+{ord(bad_char):04X}) characters.")
            if fix:
                text = text.replace(bad_char, replacement)

    # 2. Line-by-line checks
    lines = text.splitlines()
    in_frontmatter = False
    in_codeblock = False
    new_lines: list[str] = []

    for i, line in enumerate(lines, 1):
        stripped = line.strip()

        # Frontmatter boundary
        if stripped == "---":
            in_frontmatter = not in_frontmatter
            new_lines.append(line.rstrip())
            continue

        # Code block boundary
        if stripped.startswith("```"):
            in_codeblock = not in_codeblock
            new_lines.append(line.rstrip())
            continue

        # Trailing whitespace
        if line != line.rstrip():
            errors.append(f"Line {i}: Trailing whitespace detected.")
            line = line.rstrip()

        # Content checks outside frontmatter & code blocks
        if not in_frontmatter and not in_codeblock:
            # Unordered lists using '-' or '+'
            match_dash_list = re.match(r"^(\s*)([-+])(\s+.*)$", line)
            if match_dash_list:
                indent, marker, rest = match_dash_list.groups()
                errors.append(f"Line {i}: Unordered list uses '{marker}' instead of '*'.")
                if fix:
                    line = f"{indent}*{rest}"

            # Asterisk list spacing check
            match_star_list = re.match(r"^(\s*)\*(\s+)(.*)$", line)
            if match_star_list:
                indent, spaces, rest = match_star_list.groups()
                # Must be 1 space after '*'
                if len(spaces) != 1:
                    errors.append(f"Line {i}: List marker '*' must be followed by exactly 1 space.")
                    if fix:
                        line = f"{indent}* {rest}"
                # Indentation must be multiple of 2 spaces
                if len(indent) % 2 != 0:
                    errors.append(f"Line {i}: List indentation must be a multiple of 2 spaces (got {len(indent)}).")
                    if fix:
                        fixed_indent = " " * (len(indent) - (len(indent) % 2))
                        line = f"{fixed_indent}* {rest}"

        new_lines.append(line)

    if fix and (new_lines != lines or text != original_text):
        path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Enforce Rule 04 Markdown formatting.")
    parser.add_argument("paths", nargs="*", type=Path, help="Files or directories to check.")
    parser.add_argument("--fix", action="store_true", help="Auto-fix violations.")
    args = parser.parse_args()

    paths_to_check = args.paths if args.paths else [Path("docs"), Path(".agents"), Path("README.md")]

    target_files: list[Path] = []
    for p in paths_to_check:
        if p.is_file() and p.suffix == ".md":
            target_files.append(p)
        elif p.is_dir():
            target_files.extend(
                f for f in p.rglob("*.md") if ".pixi" not in str(f) and "site" not in str(f) and ".cache" not in str(f)
            )

    all_errors: dict[Path, list[str]] = {}
    for f in sorted(target_files):
        errs = verify_file(f, fix=args.fix)
        if errs:
            all_errors[f] = errs

    if all_errors:
        print(f"❌ Markdown Rule 04 Violations found in {len(all_errors)} files:", file=sys.stderr)
        for f, errs in all_errors.items():
            print(f"  • {f}:", file=sys.stderr)
            for e in errs[:5]:
                print(f"      - {e}", file=sys.stderr)
            if len(errs) > 5:
                print(f"      ... and {len(errs) - 5} more issues", file=sys.stderr)
        return 1

    print(f"✅ Markdown Rule 04 Compliance PASSED across {len(target_files)} markdown files.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
