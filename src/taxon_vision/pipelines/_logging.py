# SPDX-FileCopyrightText: 2026 Benjamin Förster
# SPDX-License-Identifier: MIT
"""Shared structured and human-readable logging bootstrap for DVC pipeline CLIs.

Summary:
    Configures root Python logging across all TaxonVision pipeline entrypoints on `sys.stdout`, automatically switching between human-readable timestamps locally and compact single-line NDJSON when running inside containerised Kubernetes pods.

Why (Architectural Rationale & Trade-offs):
    Previously, seven pipeline scripts invoked `logging.basicConfig(..., force=True)` targeting `sys.stderr` while three CLIs emitted `print()` statements to `sys.stdout`. Splitting records across `stdout` and `stderr` scrambles log ordering in DVC, Jenkins, and containerd CRI pipelines, while `force=True` strips active test `caplog` handlers. Centralising bootstrap into one helper ensures deterministic stream ordering, JSON Promtail compatibility when `TAXON_STRUCTURED_STDOUT=true`, and zero root-handler clobbering during unit tests.

How (Algorithmic Mechanism & Invariants):
    Checks `TAXON_STRUCTURED_STDOUT`, binds either `StructuredJsonFormatter` or a concise console formatter onto a single `sys.stdout` `StreamHandler` without discarding existing `LogCaptureHandler` instances, and optionally attaches a UTF-8 `FileHandler` at `DEBUG` severity.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any

from taxon_vision.service.structured_logging import StructuredJsonFormatter


def _build_stdout_formatter() -> logging.Formatter:
    """Build either an NDJSON formatter for container pods or a concise CLI formatter."""
    use_structured = os.getenv("TAXON_STRUCTURED_STDOUT", "false").strip().lower() in ("true", "1", "yes")
    if use_structured:
        return StructuredJsonFormatter()
    return logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%H:%M:%S")


def _find_stdout_handler(root: logging.Logger) -> logging.StreamHandler[Any] | None:
    """Locate an existing console StreamHandler whilst skipping FileHandlers and pytest LogCaptureHandlers."""
    for handler in list(root.handlers):
        if (
            isinstance(handler, logging.StreamHandler)
            and not isinstance(handler, logging.FileHandler)
            and type(handler).__name__ != "LogCaptureHandler"
        ):
            return handler
    return None


def _attach_debug_file_handler(root: logging.Logger, log_file: Path | str) -> None:
    """Attach a UTF-8 DEBUG FileHandler to ``root`` when not already present for ``log_file``."""
    file_path = Path(log_file)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    resolved = str(file_path.resolve())
    already_attached = any(isinstance(h, logging.FileHandler) and h.baseFilename == resolved for h in root.handlers)
    if already_attached:
        return
    file_handler = logging.FileHandler(file_path, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s:%(lineno)d: %(message)s"))
    root.addHandler(file_handler)


def configure_pipeline_logging(level: int = logging.INFO, log_file: Path | str | None = None) -> None:
    """Configure unified stdout pipeline logging and an optional debug log file.

    Args:
        level: Base severity threshold applied to root and console loggers.
        log_file: Optional destination file path persisting `DEBUG`-level log output.
    """
    root = logging.getLogger()
    root.setLevel(min(level, logging.DEBUG) if log_file else level)

    console_handler = _find_stdout_handler(root)
    if console_handler is None:
        console_handler = logging.StreamHandler(sys.stdout)
        root.addHandler(console_handler)
    else:
        console_handler.setStream(sys.stdout)

    console_handler.setLevel(level)
    console_handler.setFormatter(_build_stdout_formatter())

    if log_file:
        _attach_debug_file_handler(root, log_file)
