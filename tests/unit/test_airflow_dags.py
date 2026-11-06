# SPDX-FileCopyrightText: 2026 Benjamin Förster
# SPDX-License-Identifier: MIT
"""Unit tests verifying Apache Airflow DAG declarations and Docker Compose integration."""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest
import yaml

# Ensure root workspace is in sys.path for scripts import
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def test_airflow_dags_directory_and_files_exist() -> None:
    """Verify that the Airflow DAG directory and required DAG files exist on disk."""
    dags_dir = Path("airflow/dags")
    assert dags_dir.is_dir(), "airflow/dags directory must exist"

    drift_dag = dags_dir / "prediction_drift_monitor.py"
    retrain_dag = dags_dir / "model_retraining.py"

    assert drift_dag.is_file(), "prediction_drift_monitor.py must exist in airflow/dags"
    assert retrain_dag.is_file(), "model_retraining.py must exist in airflow/dags"


def test_prediction_drift_monitor_dag_ast_structure() -> None:
    """Statically verify the AST structure, tasks, and target endpoint of prediction_drift_monitor."""
    dag_path = Path("airflow/dags/prediction_drift_monitor.py")
    source = dag_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(dag_path))

    function_names = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]

    assert "prediction_drift_monitor" in function_names
    assert "evaluate_live_drift" in function_names
    assert "is_drift_detected" in function_names

    assert "/api/v1/drift/evaluate" in source
    assert "taxon_app" in source
    assert "TriggerDagRunOperator" in source
    assert 'trigger_dag_id="model_retraining"' in source
    assert 'schedule="@hourly"' in source or "schedule='@hourly'" in source


def test_model_retraining_dag_ast_structure() -> None:
    """Statically verify the AST structure, tasks, and sequence of model_retraining."""
    dag_path = Path("airflow/dags/model_retraining.py")
    source = dag_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(dag_path))

    function_names = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]

    assert "model_retraining" in function_names
    assert "merge_human_feedback" in function_names
    assert "trigger_jenkins_build" in function_names
    assert "wait_for_jenkins_build" in function_names
    assert "report_promotion_outcome" in function_names

    assert "schedule=None" in source
    assert "max_active_runs=1" in source
    assert "jenkins" in source


def test_docker_compose_airflow_services() -> None:
    """Verify that docker-compose.yml defines all required Airflow 3 services and configurations."""
    compose_path = Path("docker-compose.yml")
    assert compose_path.is_file()

    with compose_path.open("r", encoding="utf-8") as f:
        compose_data = yaml.safe_load(f)

    services = compose_data.get("services", {})

    expected_services = [
        "airflow-init",
        "airflow-apiserver",
        "airflow-scheduler",
        "airflow-dag-processor",
    ]
    for svc in expected_services:
        assert svc in services, f"Service {svc} must be defined in docker-compose.yml"

    apiserver = services["airflow-apiserver"]
    ports = apiserver.get("ports", [])
    assert any("8090:8080" in str(p) for p in ports), "airflow-apiserver must map host port 8090 to 8080"

    scheduler = services["airflow-scheduler"]
    assert scheduler.get("command") == "scheduler"

    dag_processor = services["airflow-dag-processor"]
    assert dag_processor.get("command") == "dag-processor"

    # Verify database connection string references shared postgres
    airflow_common = compose_data.get("x-airflow-common", {})
    env = airflow_common.get("environment", {})
    db_conn = env.get("AIRFLOW__DATABASE__SQL_ALCHEMY_CONN", "")
    assert "postgresql+psycopg2://airflow:taxon_airflow_pass@postgres:5432/airflow" in db_conn
    assert env.get("AIRFLOW__CORE__EXECUTOR") == "LocalExecutor"


def test_visualize_airflow_dag_cli(capsys: pytest.CaptureFixture[str]) -> None:
    """Verify that the visualize_airflow_dag CLI executes and formats topologies cleanly."""
    from scripts.maintenance.visualize_airflow_dag import main

    # Test all DAGs rendering
    exit_code_all = main(["visualize_airflow_dag.py"])
    captured_all = capsys.readouterr()
    assert exit_code_all == 0
    assert "prediction_drift_monitor" in captured_all.out
    assert "model_retraining" in captured_all.out

    # Test single DAG rendering
    exit_code_single = main(["visualize_airflow_dag.py", "prediction_drift_monitor"])
    captured_single = capsys.readouterr()
    assert exit_code_single == 0
    assert "prediction_drift_monitor" in captured_single.out
    assert "evaluate_live_drift" in captured_single.out

    # Test unknown DAG error
    exit_code_invalid = main(["visualize_airflow_dag.py", "invalid_dag_id"])
    captured_invalid = capsys.readouterr()
    assert exit_code_invalid == 1
    assert "Error: DAG 'invalid_dag_id' not found" in captured_invalid.err
