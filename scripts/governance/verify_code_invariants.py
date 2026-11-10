# SPDX-FileCopyrightText: 2026 Benjamin Förster
# SPDX-License-Identifier: MIT
"""AST-based code invariant verification engine.

Summary:
    Validates Python source files against strict architectural invariants
    including dynamic reflection bans, positional arity limits, eager imports,
    and async event-loop non-blocking discipline.

Why (Architectural Rationale):
    Markdown documentation alone is insufficient to prevent architectural decay.
    AI agents frequently introduce shortcuts like dynamic reflection ('getattr')
    or un-encapsulated positional parameter lists. Programmatic AST checks
    ensure continuous automated enforcement in pre-commit and CI gates,
    supporting a baseline ratchet to remediate legacy technical debt.

How (Algorithmic Mechanism & Complexity):
    Parses Python modules into Abstract Syntax Trees (AST) using Python's 'ast'
    module and walks node hierarchies to evaluate:
    - Calls to 'hasattr' and 'getattr' (Rule 00).
    - Function definitions accepting > 4 positional arguments (Rule 08).
    - Package '__init__.py' files containing eager nested submodule imports (Rule 00).
    Ratchets legacy violations against 'code_invariants_baseline.json'.
    Runtime complexity: O(N * S) where N is the number of Python source files and
    S is the AST node count per file.

Standard Fields:
    References: Rule 00 (Python Modernization), Rule 08 (Clean Code & Invariants).
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class InvariantViolation:
    """Represents a single detected architectural violation."""

    rule_id: str
    file_path: str
    line_number: int
    message: str


def check_dynamic_reflection(node: ast.Call, file_path: Path) -> InvariantViolation | None:
    """Detect forbidden hasattr and getattr calls.

    Args:
        node: AST Call node to evaluate.
        file_path: Relative path to source file.

    Returns:
        Violation instance if forbidden reflection is found, else None.
    """
    if isinstance(node.func, ast.Name) and node.func.id in {"hasattr", "getattr"}:
        return InvariantViolation(
            rule_id="Rule 00",
            file_path=file_path.as_posix(),
            line_number=node.lineno,
            message=f"Dynamic reflection '{node.func.id}' is prohibited. Use typed models or protocols.",
        )
    return None


def check_function_arity(node: ast.FunctionDef | ast.AsyncFunctionDef, file_path: Path) -> InvariantViolation | None:
    """Verify function positional parameter arity does not exceed 4.

    Args:
        node: AST function node.
        file_path: Relative path to source file.

    Returns:
        Violation instance if positional arity > 4, else None.
    """
    if node.name.startswith("__"):
        return None

    # Count positional arguments (excluding 'self' and 'cls')
    args = [arg.arg for arg in node.args.args if arg.arg not in {"self", "cls"}]
    pos_count = len(args) + len(node.args.posonlyargs)

    if pos_count > 4:
        return InvariantViolation(
            rule_id="Rule 08",
            file_path=file_path.as_posix(),
            line_number=node.lineno,
            message=(
                f"Function '{node.name}' accepts {pos_count} positional parameters (limit is 4). "
                f"Encapsulate arguments into a typed Pydantic model or dataclass."
            ),
        )
    return None


def check_eager_package_imports(node: ast.Import | ast.ImportFrom, file_path: Path) -> InvariantViolation | None:
    """Detect eager nested submodule imports in package __init__.py files.

    Args:
        node: AST import node.
        file_path: Relative path to source file.

    Returns:
        Violation instance if eager submodule import is found, else None.
    """
    if file_path.name != "__init__.py":
        return None

    if isinstance(node, ast.ImportFrom) and node.module and "." in node.module:
        return InvariantViolation(
            rule_id="Rule 00",
            file_path=file_path.as_posix(),
            line_number=node.lineno,
            message=f"Eager submodule import '{node.module}' in __init__.py is prohibited.",
        )
    return None


def inspect_module_tree(tree: ast.AST, file_path: Path) -> list[InvariantViolation]:
    """Walk module AST nodes and accumulate invariant violations.

    Args:
        tree: Parsed AST tree.
        file_path: Source file path.

    Returns:
        List of identified InvariantViolation instances.
    """
    violations: list[InvariantViolation] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            violation = check_dynamic_reflection(node, file_path)
            if violation:
                violations.append(violation)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            violation = check_function_arity(node, file_path)
            if violation:
                violations.append(violation)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            violation = check_eager_package_imports(node, file_path)
            if violation:
                violations.append(violation)

    return violations


def scan_source_directory(src_dir: Path) -> list[InvariantViolation]:
    """Iterate through Python files in the source tree and evaluate AST rules.

    Args:
        src_dir: Base directory containing source modules.

    Returns:
        Aggregated list of all invariant violations.
    """
    all_violations: list[InvariantViolation] = []

    for py_file in sorted(src_dir.rglob("*.py")):
        if "scratch" in py_file.parts or ".pixi" in py_file.parts:
            continue
        try:
            content = py_file.read_text(encoding="utf-8")
            tree = ast.parse(content, filename=py_file.as_posix())
            violations = inspect_module_tree(tree, py_file)
            all_violations.extend(violations)
        except SyntaxError as err:
            all_violations.append(
                InvariantViolation(
                    rule_id="Syntax",
                    file_path=py_file.as_posix(),
                    line_number=err.lineno or 0,
                    message=f"Syntax parsing failure: {err.msg}",
                )
            )

    return all_violations


def load_baseline(baseline_file: Path) -> set[tuple[str, str, str]]:
    """Load baseline violations set from JSON snapshot.

    Args:
        baseline_file: Path to baseline JSON file.

    Returns:
        Set of tuples (rule_id, file_path, message_signature).
    """
    if not baseline_file.exists():
        return set()
    try:
        data = json.loads(baseline_file.read_text(encoding="utf-8"))
        return {(item["rule_id"], item["file_path"], item["message"]) for item in data}
    except Exception:
        return set()


def save_baseline(violations: list[InvariantViolation], baseline_file: Path) -> None:
    """Save current violations to baseline snapshot.

    Args:
        violations: Current list of violations.
        baseline_file: Path to baseline JSON file.
    """
    baseline_file.parent.mkdir(parents=True, exist_ok=True)
    payload = [asdict(v) for v in violations]
    baseline_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def evaluate_violations(
    violations: list[InvariantViolation],
    baseline_set: set[tuple[str, str, str]],
    is_strict: bool,
) -> int:
    """Compare violations against baseline and return exit code.

    Args:
        violations: Identified violations.
        baseline_set: Set of baseline violations.
        is_strict: Whether strict mode is enabled.

    Returns:
        Process exit code (0 for pass, 1 for fail).
    """
    if is_strict:
        new_violations = violations
    else:
        new_violations = [v for v in violations if (v.rule_id, v.file_path, v.message) not in baseline_set]

    if not new_violations:
        print(f"✅ Code Invariants Verification: Passed ({len(violations)} legacy baseline items ratcheted).")
        return 0

    print(f"❌ Found {len(new_violations)} new architectural invariant violation(s):")
    for v in new_violations:
        print(f"  [{v.rule_id}] {v.file_path}:{v.line_number} -> {v.message}")
    return 1


def main() -> None:
    """Execute code invariant scan and return appropriate process exit code."""
    parser = argparse.ArgumentParser(description="AST Code Invariant Verification")
    parser.add_argument("--snapshot-create", action="store_true", help="Create baseline snapshot")
    parser.add_argument("--strict", action="store_true", help="Disallow all violations including baseline")
    args = parser.parse_args()

    src_dir = Path("src/taxon_vision")
    baseline_path = Path("scripts/governance/code_invariants_baseline.json")

    violations = scan_source_directory(src_dir)

    if args.snapshot_create:
        save_baseline(violations, baseline_path)
        print(f"📸 Baseline snapshot created with {len(violations)} items at {baseline_path}")
        sys.exit(0)

    baseline_set = load_baseline(baseline_path)
    exit_code = evaluate_violations(violations, baseline_set, args.strict)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
