# SPDX-FileCopyrightText: 2026 Benjamin Förster
# SPDX-License-Identifier: MIT
"""Structured JSON Logging Infrastructure for Grafana Loki Observability.

Summary:
    Implements an RFC3339-compliant JSON log formatter and file handler configuration to streamline log shipping via Promtail to Grafana Loki.

Why:
    Unstructured, human-oriented plain-text log lines require fragile regular expressions in LogQL, increasing CPU consumption and query latency during incident triage. Emitting newline-delimited JSON (NDJSON) enables zero-regex streaming parsing in Grafana Loki (`| json`), facilitates direct correlation with Prometheus latency histograms via trace IDs, and guarantees deterministic retention without Elasticsearch-level memory overhead.

How:
    Subclasses `logging.Formatter` to serialize standard `LogRecord` fields and extra context attributes into JSON objects. Configures a `RotatingFileHandler` writing to a designated volume path (`data/logs/app.log`) while retaining standard console logging for terminal inspection.

Standard Fields:
    timestamp: ISO 8601 / RFC 3339 UTC string with millisecond precision.
    level: Log severity string (INFO, WARNING, ERROR, CRITICAL).
    logger: Python logger hierarchy name.
    message: Formatted log message text.
    exception: Optional multiline traceback string if exception info is present.
"""

from __future__ import annotations

import datetime
import json
import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Final

_BUILTIN_RECORD_ATTRS: Final[frozenset[str]] = frozenset(
    {
        "name",
        "msg",
        "args",
        "levelname",
        "levelno",
        "pathname",
        "filename",
        "module",
        "exc_info",
        "exc_text",
        "stack_info",
        "lineno",
        "funcName",
        "created",
        "msecs",
        "relativeCreated",
        "thread",
        "threadName",
        "processName",
        "process",
        "message",
        "taskName",
    }
)


class StructuredJsonFormatter(logging.Formatter):
    """Formats Python logging records as compact, single-line JSON objects.

    Why:
        Unstructured multiline tracebacks and plain log lines fragment inside Promtail container log pipelines; single-line JSON ensures Grafana Loki parses severity, logger module, UTC timestamp, and correlation metadata deterministically without custom regex scrapers.

    How:
        Formats `record.created` as UTC ISO-8601, extracts standard message/exception fields alongside non-builtin `extra={...}` keys without dynamic attribute reflection (`hasattr`/`getattr`), and serialises cleanly to UTF-8 JSON.
    """

    def format(self, record: logging.LogRecord) -> str:
        """Format the specified record as newline-delimited JSON.

        Why:
            Emits uniform machine-parseable log lines compatible with both local rotating JSON logs and containerd CRI stdout log collectors.

        How:
            Constructs a typed dictionary from standard `LogRecord` attributes, attaches formatted exception info when present, merges user-supplied extras, and serialises via `json.dumps`.

        Args:
            record: Logging record instance emitted by an application logger.

        Returns:
            Single-line JSON string representing the structured log entry.
        """
        created_dt = datetime.datetime.fromtimestamp(record.created, tz=datetime.UTC)
        payload: dict[str, object] = {
            "timestamp": created_dt.isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        # Include custom domain fields passed via extra={...} without using dynamic reflection
        for key, value in record.__dict__.items():
            if key not in _BUILTIN_RECORD_ATTRS:
                payload[key] = value

        return json.dumps(payload, default=str)


def _ensure_structured_stdout_handler(root_logger: logging.Logger) -> None:
    """Configure root stdout `StreamHandler` instances to emit compact JSON when `TAXON_STRUCTURED_STDOUT` is enabled.

    Why:
        Inside Kubernetes pods, Promtail tails container stdout from `/var/log/pods` via containerd CRI rather than reading application-private files; switching stdout formatting in-place activates Loki ingestion without duplicating file mounts.

    How:
        Checks `TAXON_STRUCTURED_STDOUT`, applies `StructuredJsonFormatter` to any existing non-file `StreamHandler` on `root_logger`, or attaches a new `sys.stdout` handler if none exists.

    Args:
        root_logger: Python root logger instance to configure.
    """
    use_structured_stdout = os.getenv("TAXON_STRUCTURED_STDOUT", "false").strip().lower() in ("true", "1", "yes")
    if not use_structured_stdout:
        return

    formatter = StructuredJsonFormatter()
    updated_existing = False
    for existing in root_logger.handlers:
        if isinstance(existing, logging.StreamHandler) and not isinstance(existing, RotatingFileHandler):
            existing.setFormatter(formatter)
            existing.setLevel(logging.INFO)
            updated_existing = True

    if not updated_existing:
        stdout_handler = logging.StreamHandler(sys.stdout)
        stdout_handler.setFormatter(formatter)
        stdout_handler.setLevel(logging.INFO)
        root_logger.addHandler(stdout_handler)


def configure_structured_logging(
    log_dir: Path | str | None = None,
    log_filename: str = "app.log",
    max_bytes: int = 10 * 1024 * 1024,
    backup_count: int = 3,
) -> None:
    """Configure root logger to mirror events into a structured JSON file and optional JSON stdout stream.

    Why:
        Enables simultaneous human-readable terminal output locally and machine-parseable JSON streams for Promtail log shipping across both host volumes and Kubernetes containerd CRI pod logs.

    How:
        Resolves `log_dir`, switches root stdout handlers to `StructuredJsonFormatter` when `TAXON_STRUCTURED_STDOUT` is set, and idempotently attaches a size-bounded `RotatingFileHandler`.

    Args:
        log_dir: Directory where log files are persisted (defaults to `TAXON_LOG_DIR` or `'data/logs'`).
        log_filename: Log file name inside `log_dir`.
        max_bytes: Maximum size in bytes before rotating (default: 10 MB).
        backup_count: Number of rotated log archives to retain (default: 3).
    """
    target_dir: str | Path = log_dir if log_dir is not None else os.getenv("TAXON_LOG_DIR", "data/logs")
    resolved_dir = Path(target_dir)
    resolved_dir.mkdir(parents=True, exist_ok=True)
    log_path = resolved_dir / log_filename

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    _ensure_structured_stdout_handler(root_logger)

    for existing_handler in root_logger.handlers:
        if isinstance(existing_handler, RotatingFileHandler) and str(existing_handler.baseFilename) == str(
            log_path.resolve()
        ):
            return

    file_handler = RotatingFileHandler(
        log_path,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    file_handler.setFormatter(StructuredJsonFormatter())
    file_handler.setLevel(logging.INFO)
    root_logger.addHandler(file_handler)
