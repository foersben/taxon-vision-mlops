# SPDX-FileCopyrightText: 2026 Benjamin Förster
# SPDX-License-Identifier: MIT
"""Classification head training pipeline entry point."""

from __future__ import annotations

import sys

from taxon_vision.models.training.runner import train_cli


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for training pipeline execution.

    Why:
        Enables seamless invocation as both an executable module (`python -m taxon_vision.pipelines.train`)
        and a programmatic entry point for integration tests or runner scripts.

    Args:
        argv: Optional list of command-line arguments (defaults to `sys.argv[1:]`).

    Returns:
        Integer exit code (0 for success, non-zero for error).
    """
    return train_cli(argv)


if __name__ == "__main__":
    sys.exit(main())
