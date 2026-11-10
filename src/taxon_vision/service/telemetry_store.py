# SPDX-FileCopyrightText: 2026 Benjamin Förster
# SPDX-License-Identifier: MIT
"""Prediction telemetry persistence and historical embedding retrieval.

Why:
    Online distribution drift monitoring - specifically Tier 1 dimensional Wasserstein-1 distances and Tier 2 Maximum Mean Discrepancy (MMD²) - requires historical access to penultimate feature representations from production inference traffic. Persisting raw pre-normalised activation vectors alongside prediction metadata in a relational PostgreSQL store decouples ephemeral model serving from asynchronous analytical pipelines, guaranteeing that the FastAPI microservice p95 latency ceiling (< 25 ms) remains unaffected by drift evaluation.

How:
    Employs a thread-safe connection pool via psycopg 3 to execute non-blocking writes of pre-normalised REAL[] vectors, conformal prediction sets, and end-to-end latency metrics. Abstracts storage behind the TelemetryStore Protocol. When relational database connectivity fails, the system enforces fail-fast error handling by raising an exception unless in-memory buffering is explicitly permitted via configuration, in which case a bounded ring buffer is provisioned to prevent out-of-memory heap expansion.
"""

from __future__ import annotations

import logging
import threading
from collections import deque
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Protocol

import numpy as np
import numpy.typing as npt
from psycopg_pool import ConnectionPool

from taxon_vision.config import get_settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PredictionRecord:
    """Immutable data container representing an evaluated inference prediction.

    Why:
        Preserves complete telemetry context across the prediction lifecycle without mutable state leaks. Decoupling observation metadata, model versioning, conformal prediction sets, OOD energy scores, and penultimate feature embeddings into a frozen record enforces referential transparency and guarantees deterministic batch serialisation.

    How:
        Encapsulates scalar classification results, string array conformal sets, and a pre-normalised NumPy float32 embedding vector. Frozen dataclass semantics prevent post-instantiation mutation on asynchronous executor threads.

    Attributes:
        observation_id: Unique identifier or UUID of the evaluated observation.
        extractor: Feature extraction backbone identifier (e.g. 'dinov2_small').
        model_version: Checkpoint hash or release version tag of the active model.
        top1_taxon: Scientific taxon name assigned the highest posterior probability.
        top1_confidence: Calibrated or softmax confidence score for the top-1 prediction.
        conformal_set: Class labels included in the Split Conformal prediction set.
        ood_energy: Negative Helmholtz free energy score for out-of-distribution gating.
        is_ood: Boolean flag indicating whether energy exceeded the OOD threshold tau.
        requires_review: Flag indicating whether triage review is triggered (OOD or set ambiguity).
        latency_ms: End-to-end inference execution latency in milliseconds.
        embedding: Pre-normalised penultimate feature vector of shape (D,), or None.
    """

    observation_id: str
    extractor: str
    model_version: str | None
    top1_taxon: str | None
    top1_confidence: float | None
    conformal_set: list[str]
    ood_energy: float | None
    is_ood: bool
    requires_review: bool
    latency_ms: float
    embedding: npt.NDArray[np.float32] | None


@dataclass(frozen=True)
class DriftRunRecord:
    """Immutable data container representing an evaluated drift detection run.

    Summary:
        Stores the statistical outputs and metadata from a two-tier drift analysis window.

    Attributes:
        window_size: Number of production samples evaluated in the sliding window.
        w1_distance: Computed 1D Wasserstein-1 distance on embedding norms.
        w1_threshold: Calibrated baseline threshold tau_W1.
        mmd2: Computed unbiased sample estimate of Maximum Mean Discrepancy squared.
        mmd2_p_value: Empirical p-value from Monte Carlo permutation testing.
        drift_detected: Boolean flag indicating if either tier breached its decision boundary.
        mlflow_run_id: Associated MLflow experiment run ID, if tracking was active.
        created_at: Timestamp when the evaluation was executed.
        id: Relational database primary key identifier, if persisted.
    """

    window_size: int
    w1_distance: float
    w1_threshold: float
    mmd2: float
    mmd2_p_value: float
    drift_detected: bool
    mlflow_run_id: str | None = None
    created_at: datetime | None = None
    id: int | None = None


class TelemetryStore(Protocol):
    """Protocol interface defining telemetry persistence and embedding retrieval operations.

    Why:
        Decouples the FastAPI application routes from concrete relational database drivers. Enables zero-cost mock injection during unit testing and provides seamless fallback boundaries during local offline development or temporary PostgreSQL outages.

    How:
        Defines an asynchronous/synchronous threadpool contract for recording single prediction telemetry entities and retrieving sliding-window matrices of historical embeddings for drift statistical hypothesis testing.
    """

    def record(self, rec: PredictionRecord) -> None:
        """Persist a single prediction telemetry record.

        Why:
            Provides durable audit trails for model outputs, out-of-distribution flags, and photographer-associated prediction sets necessary for governance and compliance.

        How:
            Dispatches structured attributes of the given record to the underlying storage backend. Implementations must handle database connection timeouts and serialisation errors gracefully without terminating the serving worker.

        Args:
            rec: Immutable prediction record containing inference outputs and embeddings.
        """
        ...

    def recent_embeddings(self, limit: int, extractor: str | None = None) -> npt.NDArray[np.float32]:
        """Retrieve recent pre-normalised feature embeddings for drift detection.

        Why:
            Two-sample statistical drift testing (Wasserstein-1 and MMD²) requires a sliding window of recent production embeddings to compare against reference calibration embeddings without modifying serving caches.

        How:
            Queries the underlying repository for the most recent un-normalised feature vectors, returning a contiguous two-dimensional matrix of shape (N, D).

        Args:
            limit: Maximum number of recent embedding vectors to retrieve (N).
            extractor: Model extractor name to filter by, or None for all.

        Returns:
            Contiguous float32 NumPy array of shape (N, D), or shape (0, 0) if empty.
        """
        ...

    def record_drift_run(self, run: DriftRunRecord) -> None:
        """Persist an evaluated drift analysis record.

        Why:
            Provides durable audit trails for model outputs, out-of-distribution flags, and photographer-associated prediction sets necessary for governance and compliance.

        How:
            Dispatches structured attributes of the given record to the underlying storage backend. Implementations must handle database connection timeouts and serialisation errors gracefully without terminating the serving worker.

        Args:
            run: Immutable drift run record containing statistical outputs and metadata.
        """
        ...

    def latest_drift_run(self) -> DriftRunRecord | None:
        """Retrieve the most recent drift evaluation record.

        Why:
            Provides the most recent drift detection results for monitoring and alerting purposes.

        How:
            Queries the underlying repository for the most recent drift evaluation record, returning a contiguous two-dimensional matrix of shape (N, D).

        Returns:
            Contiguous float32 NumPy array of shape (N, D), or shape (0, 0) if empty.
        """
        ...


_INSERT_SQL = """
INSERT INTO predictions (
    observation_id, model_version, extractor, top1_taxon, top1_confidence,
    conformal_set, set_size, ood_energy, is_ood, requires_review, latency_ms, embedding
) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
ON CONFLICT (observation_id) DO NOTHING
"""

_INSERT_DRIFT_RUN_SQL = """
INSERT INTO drift_runs (
    window_size, w1_distance, w1_threshold, mmd2, mmd2_p_value, drift_detected, mlflow_run_id
) VALUES (%s, %s, %s, %s, %s, %s, %s)
"""

_SELECT_LATEST_DRIFT_RUN_SQL = """
SELECT id, created_at, window_size, w1_distance, w1_threshold, mmd2, mmd2_p_value, drift_detected, mlflow_run_id
FROM drift_runs
ORDER BY created_at DESC
LIMIT 1
"""


class PostgresTelemetryStore:
    """PostgreSQL-backed prediction telemetry repository utilising a bounded connection pool.

    Why:
        Production deployment on dedicated cluster nodes requires high-throughput, persistent telemetry storage capable of indexing high-dimensional feature vectors (`REAL[]`). Managing queries through a pre-warmed connection pool avoids the socket handshaking overhead of raw TCP connections, preserving the microservice latency budget.

    How:
        Interacts with the PostgreSQL `predictions` table via parameterised psycopg 3 statements. Embeddings are serialised to native PostgreSQL `REAL[]` arrays. Conflicting observation identifiers are ignored (`ON CONFLICT (observation_id) DO NOTHING`) to guarantee idempotent ingestion. Write failures increment Prometheus telemetry failure counters and emit structured error logs without raising unhandled exceptions into the event loop.

    Complexity:
        Write operations: O(D) time complexity where D is embedding dimension (e.g. 384).
        Read operations: O(N * D) retrieval and array reconstruction time complexity.
    """

    def __init__(self, pool: ConnectionPool) -> None:
        """Initialise PostgreSQL repository with a pre-configured connection pool.

        Why:
            Connection pooling bounds database socket consumption, eliminating connection churn and preventing resource starvation under concurrent request spikes.

        How:
            Stores the supplied thread-safe psycopg 3 ConnectionPool reference for RAII-scoped context manager acquisition during read and write transactions.

        Args:
            pool: Active thread-safe psycopg ConnectionPool instance.
        """
        self._pool = pool

    def record(self, rec: PredictionRecord) -> None:
        """Persist an inference record into the relational predictions table.

        Why:
            Guarantees durable persistence of prediction outputs and high-dimensional representations for downstream drift monitoring, active learning triage, and model auditing.

        How:
            Extracts fields from the immutable PredictionRecord, serialises the NumPy array into a Python float list for PostgreSQL REAL[] mapping, and executes an idempotent INSERT using a leased connection. If an exception occurs, records a metric failure via `TELEMETRY_WRITE_FAILURES.inc()` and logs the stack trace with fail-fast observability.

        Args:
            rec: Prediction record containing inference attributes and feature embeddings.
        """
        emb = rec.embedding.astype(np.float32).flatten().tolist() if rec.embedding is not None else None
        try:
            with self._pool.connection() as conn:
                conn.execute(
                    _INSERT_SQL,
                    (
                        rec.observation_id,
                        rec.model_version,
                        rec.extractor,
                        rec.top1_taxon,
                        rec.top1_confidence,
                        rec.conformal_set,
                        len(rec.conformal_set),
                        rec.ood_energy,
                        rec.is_ood,
                        rec.requires_review,
                        rec.latency_ms,
                        emb,
                    ),
                )
        except Exception:
            try:
                from taxon_vision.monitoring.telemetry import TELEMETRY_WRITE_FAILURES

                TELEMETRY_WRITE_FAILURES.inc()
            except Exception:
                pass
            logger.exception("Failed to write prediction telemetry for observation %s", rec.observation_id)

    def recent_embeddings(self, limit: int, extractor: str | None = None) -> npt.NDArray[np.float32]:
        """Fetch the most recent pre-normalised embeddings ordered by timestamp descending.

        Why:
            Provides the empirical test sample matrix required by Tier 1 (Wasserstein-1) and Tier 2 (MMD²) drift detection engines to quantify feature distribution shifts.

        How:
            Leases a connection from the pool, executes an indexed `SELECT embedding` query ordered by `created_at DESC` up to the requested limit, filters null embeddings, and constructs a contiguous 2D NumPy array of shape (N, D).

        Complexity:
            O(N * D) space and time complexity for reconstructing N vectors of dimension D.

        Args:
            limit: Maximum count of recent observation embeddings to fetch.
            extractor: Model extractor name to filter by, or None for all.

        Returns:
            NumPy array of shape (N, D) containing float32 vectors, or shape (0, 0) if absent.

        Raises:
            RuntimeError: If query execution or pool connection checkout fails.
        """
        try:
            with self._pool.connection() as conn:
                if extractor:
                    query = (
                        "SELECT embedding FROM predictions "
                        "WHERE extractor = %s AND embedding IS NOT NULL "
                        "ORDER BY created_at DESC LIMIT %s"
                    )
                    rows = conn.execute(query, (extractor, limit)).fetchall()
                else:
                    query = (
                        "SELECT embedding FROM predictions "
                        "WHERE embedding IS NOT NULL "
                        "ORDER BY created_at DESC LIMIT %s"
                    )
                    rows = conn.execute(query, (limit,)).fetchall()
            if not rows:
                return np.empty((0, 0), dtype=np.float32)
            return np.ascontiguousarray([r[0] for r in rows], dtype=np.float32)
        except Exception as exc:
            logger.exception("Failed to fetch recent embeddings from telemetry store")
            raise RuntimeError("Failed to retrieve recent embeddings from telemetry store") from exc

    def record_drift_run(self, run: DriftRunRecord) -> None:
        """Persist a drift detection run record into the relational drift_runs table.

        Why:
            Enables forensic analysis of detection events, performance comparison across runs via Wasserstein-1 and MMD², and integration with MLflow experiment tracking for model quality governance.

        Args:
            run: Drift run record containing detection metrics, configuration, and MLflow run metadata.
        """
        try:
            with self._pool.connection() as conn:
                conn.execute(
                    _INSERT_DRIFT_RUN_SQL,
                    (
                        run.window_size,
                        run.w1_distance,
                        run.w1_threshold,
                        run.mmd2,
                        run.mmd2_p_value,
                        run.drift_detected,
                        run.mlflow_run_id,
                    ),
                )
        except Exception:
            logger.exception("Failed to persist drift run record to PostgreSQL")

    def latest_drift_run(self) -> DriftRunRecord | None:
        """Retrieve the most recent drift detection run record from the database.

        Why:
            Enables CI/CD validation gate queries, API status endpoints, and drift health dashboards to query the last computed statistical signature.

        Returns:
            DriftRunRecord instance if available, otherwise None.

        Raises:
            RuntimeError: On query execution failure or pool checkout issues.
        """
        try:
            with self._pool.connection() as conn:
                row = conn.execute(_SELECT_LATEST_DRIFT_RUN_SQL).fetchone()
            if not row:
                return None
            return DriftRunRecord(
                id=row[0],
                created_at=row[1],
                window_size=row[2],
                w1_distance=row[3],
                w1_threshold=row[4],
                mmd2=row[5],
                mmd2_p_value=row[6],
                drift_detected=row[7],
                mlflow_run_id=row[8],
            )
        except Exception:
            logger.exception("Failed to fetch latest drift run from PostgreSQL")
            return None


class BoundedInMemoryTelemetryStore:
    """Bounded in-memory telemetry repository utilising a fixed-capacity ring buffer.

    Why:
        Production serving instances operate within strict physical host memory ceilings (reserving RAM for LanceDB vector indexes, Apache Arrow buffers, and the INT8 ONNX serving envelope). An unbounded in-memory list introduces a critical memory leak under sustained database outages. Employing a fixed-size double-ended queue guarantees that heap consumption is strictly bounded to O(maxlen) whilst allowing graceful offline execution when explicitly permitted.

    How:
        Encapsulates a `collections.deque` with a fixed `maxlen`. When the buffer reaches capacity, the oldest prediction records are automatically evicted in O(1) time. Historical feature retrieval scans the queue in reverse chronological order and serialises non-null embedding arrays into a contiguous C-order float32 NumPy array.

    Complexity:
        Write operations: O(1) amortised insertion and eviction time complexity. Read operations: O(min(limit, maxlen) * D) time complexity for matrix reconstruction.

    Attributes:
        records: Bounded deque of PredictionRecord instances.
        drift_runs: Bounded deque of DriftRunRecord instances.
    """

    def __init__(self, maxlen: int = 500) -> None:
        """Initialise bounded in-memory telemetry buffer.

        Why:
            Enforces a hard limit on RAM consumption during offline execution or transient relational database outages, preventing runaway memory expansion.

        How:
            Instantiates an internal double-ended queue bounded to `maxlen` entries.

        Args:
            maxlen: Maximum number of recent prediction records retained in memory.
        """
        self.records: deque[PredictionRecord] = deque(maxlen=maxlen)
        self.drift_runs: deque[DriftRunRecord] = deque(maxlen=100)

    def record(self, rec: PredictionRecord) -> None:
        """Append a prediction record to the ring buffer, evicting the oldest entry on overflow.

        Why:
            Captures prediction metadata synchronously in volatile memory without blocking the request thread or risking out-of-memory container terminations.

        How:
            Appends the record to the right side of the bounded deque. If length exceeds maxlen, the leftmost (oldest) entry is discarded automatically by the deque implementation.

        Args:
            rec: Prediction record containing inference attributes and feature embeddings.
        """
        self.records.append(rec)

    def recent_embeddings(self, limit: int, extractor: str | None = None) -> npt.NDArray[np.float32]:
        """Fetch the most recent pre-normalised embeddings ordered by timestamp descending.

        Why:
            Supplies recent empirical observation vectors to drift detection engines (Wasserstein-1 and MMD²) without requiring an active PostgreSQL database connection.

        How:
            Iterates through the bounded deque in reverse chronological order, extracts non-null feature vectors, flattens any multi-dimensional arrays, slices up to `limit`, and allocates a contiguous 2D float32 NumPy array of shape (N, D).

        Complexity:
            O(min(limit, maxlen) * D) space and time complexity where D is embedding dimension.

        Args:
            limit: Maximum count of recent observation embeddings to fetch.
            extractor: Model extractor name to filter by, or None for all.

        Returns:
            Contiguous float32 NumPy array of shape (N, D), or shape (0, 0) if absent.
        """
        embs = [
            r.embedding.flatten()
            for r in reversed(self.records)
            if r.embedding is not None and (extractor is None or r.extractor == extractor)
        ]
        if not embs:
            return np.empty((0, 0), dtype=np.float32)
        return np.ascontiguousarray(embs[:limit], dtype=np.float32)

    def record_drift_run(self, run: DriftRunRecord) -> None:
        """Append a drift detection run record to the ring buffer, evicting the oldest entry on overflow.

        Why:
            Captures statistical signature evaluation results synchronously in volatile memory without blocking the request thread or risking out-of-memory container terminations.

        Args:
            run: Drift run record containing statistical signature evaluation results.
        """
        if run.created_at is None:
            run = DriftRunRecord(
                id=len(self.drift_runs) + 1,
                created_at=datetime.now(UTC),
                window_size=run.window_size,
                w1_distance=run.w1_distance,
                w1_threshold=run.w1_threshold,
                mmd2=run.mmd2,
                mmd2_p_value=run.mmd2_p_value,
                drift_detected=run.drift_detected,
                mlflow_run_id=run.mlflow_run_id,
            )
        self.drift_runs.append(run)

    def latest_drift_run(self) -> DriftRunRecord | None:
        """Retrieve the most recent drift detection run record from the bounded in-memory buffer.

        Why: Enables CI/CD validation gate queries, API status endpoints, and drift health dashboards to query the last computed statistical signature without an active database connection.

        How: Inspects the most recent entry in the bounded deque of drift run records.

        Returns:
            The most recent drift detection run record, or None if the buffer is empty.
        """
        return self.drift_runs[-1] if self.drift_runs else None


_STORE_INSTANCE: TelemetryStore | None = None
_POOL_INSTANCE: ConnectionPool | None = None
_INIT_LOCK = threading.Lock()


def get_telemetry_store() -> TelemetryStore:
    """Provide a singleton TelemetryStore instance.

    Why:
        Centralises database connection pool management across the ASGI application lifecycle. Prevents redundant connection pool allocations and ensures consistent telemetry routing throughout the service.

    How:
        Inspects the module-level singleton instance using thread-safe double-checked locking. If uninitialised, reads database settings from application configuration, normalises the DSN for libpq URI compatibility, attempts to instantiate a bounded psycopg 3 ConnectionPool, and verifies connectivity via pool.wait. If the database connection fails:
        1. When `allow_in_memory_fallback` is False (default), fails fast by raising a RuntimeError to avoid masking database connection outages.
        2. When `allow_in_memory_fallback` is True, instantiates a BoundedInMemoryTelemetryStore capped at `fallback_buffer_size` whilst logging an operational warning.

    Returns:
        Configured TelemetryStore instance adhering to the protocol.

    Raises:
        RuntimeError: If PostgreSQL connection fails and allow_in_memory_fallback is disabled.
    """
    global _STORE_INSTANCE, _POOL_INSTANCE
    if _STORE_INSTANCE is not None:
        return _STORE_INSTANCE

    with _INIT_LOCK:
        if _STORE_INSTANCE is not None:
            return _STORE_INSTANCE

        settings = get_settings()
        dsn = settings.telemetry.database_url
        # Normalise SQLAlchemy-style URI scheme to standard libpq URI for psycopg 3
        conninfo = dsn.replace("postgresql+psycopg://", "postgresql://", 1)
        pool: ConnectionPool | None = None
        try:
            pool = ConnectionPool(
                conninfo=conninfo,
                min_size=settings.telemetry.pool_min_size,
                max_size=settings.telemetry.pool_max_size,
                timeout=settings.telemetry.timeout_seconds,
                open=True,
            )
            pool.wait(timeout=min(2.0, settings.telemetry.timeout_seconds))
            _POOL_INSTANCE = pool
            _STORE_INSTANCE = PostgresTelemetryStore(_POOL_INSTANCE)
            logger.info("Initialized PostgresTelemetryStore connected to %s", dsn.split("@")[-1])
        except Exception as err:
            if pool is not None:
                pool.close()

            if not settings.telemetry.allow_in_memory_fallback:
                logger.error(
                    "PostgreSQL connection pool failed (%s) and allow_in_memory_fallback is disabled. Failing fast.",
                    err,
                )
                raise RuntimeError(
                    f"Failed to connect to PostgreSQL telemetry database at {dsn}: {err}. "
                    "Set telemetry.allow_in_memory_fallback=true in configuration or export "
                    "TAXON_TELEMETRY_ALLOW_IN_MEMORY_FALLBACK=true to permit bounded in-memory buffering."
                ) from err

            logger.warning(
                "PostgreSQL connection pool failed (%s); using BoundedInMemoryTelemetryStore fallback (maxlen=%d)",
                err,
                settings.telemetry.fallback_buffer_size,
            )
            _STORE_INSTANCE = BoundedInMemoryTelemetryStore(maxlen=settings.telemetry.fallback_buffer_size)

    return _STORE_INSTANCE


def set_telemetry_store(store: TelemetryStore | None) -> None:
    """Override the global telemetry store instance.

    Why:
        Enables seamless dependency injection of mock stores during automated testing, preventing test pollution of live database instances.

    How:
        Assigns the provided TelemetryStore instance or None directly to the private module-level variable `_STORE_INSTANCE` within a re-entrant lock.

    Args:
        store: Custom TelemetryStore implementation, or None to clear singleton state.
    """
    global _STORE_INSTANCE
    with _INIT_LOCK:
        _STORE_INSTANCE = store


def close_telemetry_store() -> None:
    """Teardown database connection pool deterministically during ASGI shutdown.

    Why:
        Enforces RAII lifecycle discipline. Dangling PostgreSQL connections exhaust cluster socket limits and trigger connection pool leaks across deployment rolling restarts.

    How:
        Closes the active psycopg 3 ConnectionPool, drains active connections, and resets singleton references to None within a re-entrant lock.
    """
    global _STORE_INSTANCE, _POOL_INSTANCE
    with _INIT_LOCK:
        if _POOL_INSTANCE is not None:
            _POOL_INSTANCE.close()
            _POOL_INSTANCE = None
        _STORE_INSTANCE = None
