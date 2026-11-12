# SPDX-FileCopyrightText: 2026 Benjamin Förster
# SPDX-License-Identifier: MIT
"""Hourly prediction drift monitor DAG for automated distribution shift evaluation.

Summary:
    Orchestrates scheduled hourly evaluations of incoming prediction telemetry against
    the canonical baseline embedding manifold, automatically triggering retraining on detected drift.

Why:
    Continuous inference in open ecological environments experiences seasonal, illumination,
    and sensor degradation shifts. To ensure statistical exchangeability required for conformal
    prediction coverage guarantees (1 - alpha), live prediction vectors must be evaluated
    periodically via two-tier hypothesis tests (Wasserstein-1 and RBF MMD²).

How:
    Dispatches an authenticated HTTP POST request to the internal `/api/v1/drift/evaluate` endpoint
    via `HttpHook(http_conn_id="taxon_app")`. Evaluates the returned JSON payload with a short-circuit
    operator; if `drift_detected` is True, dispatches `TriggerDagRunOperator` to start `model_retraining`.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

from airflow.providers.http.hooks.http import HttpHook

try:
    from airflow.providers.standard.operators.trigger_dagrun import TriggerDagRunOperator
except ImportError:
    from airflow.operators.trigger_dagrun import TriggerDagRunOperator  # type: ignore[no-redef]

try:
    from airflow.sdk import dag, task
except ImportError:
    from airflow.decorators import dag, task  # type: ignore[no-redef]


@dag(
    dag_id="prediction_drift_monitor",
    schedule="@hourly",
    start_date=datetime(2026, 11, 1),
    catchup=False,
    max_active_runs=1,
    default_args={
        "retries": 2,
        "retry_delay": timedelta(minutes=5),
    },
    tags=["taxon-vision", "drift", "monitoring"],
)
def prediction_drift_monitor() -> None:
    """Hourly DAG orchestrating statistical drift evaluation and retraining dispatch."""

    @task(multiple_outputs=False)
    def evaluate_live_drift() -> dict[str, Any]:
        """Trigger on-demand drift evaluation via the internal microservice endpoint.

        Returns:
            Structured dictionary representation of `DriftEvaluationResponse`.
        """
        hook = HttpHook(method="POST", http_conn_id="taxon_app")
        conn = hook.get_connection("taxon_app")
        auth_token = conn.password or ""
        headers: dict[str, str] = {}
        if auth_token:
            headers["Authorization"] = f"Bearer {auth_token}"

        response = hook.run(
            endpoint="/api/v1/drift/evaluate",
            headers=headers,
            extra_options={"timeout": 300},
        )
        verdict: dict[str, Any] = response.json()
        return verdict

    @task.short_circuit
    def is_drift_detected(verdict: dict[str, Any]) -> bool:
        """Evaluate whether distribution shift requires model retraining.

        Args:
            verdict: Telemetry drift evaluation response containing `drift_detected`.

        Returns:
            True if drift was detected and downstream retraining should trigger; False otherwise.
        """
        return bool(verdict.get("drift_detected", False))

    trigger_retraining = TriggerDagRunOperator(
        task_id="trigger_retraining",
        trigger_dag_id="model_retraining",
        conf={"trigger_source": "prediction_drift_monitor"},
    )

    verdict_payload = evaluate_live_drift()
    is_drift_detected(verdict_payload) >> trigger_retraining


prediction_drift_monitor()
