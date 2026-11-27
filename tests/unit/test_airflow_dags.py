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


def test_airflow_k8s_mounts_cover_all_model_retraining_imports() -> None:
    """Ensure Airflow mounts the active_learning package directory and Jenkins syncs every module in it."""
    dag_path = Path("airflow/dags/model_retraining.py")
    tree = ast.parse(dag_path.read_text(encoding="utf-8"), filename=str(dag_path))
    airflow_docs = list(yaml.safe_load_all(Path("deploy/k8s/airflow.yaml").read_text(encoding="utf-8")))
    jenkinsfile = Path("Jenkinsfile").read_text(encoding="utf-8")
    deploy_script = Path("scripts/ci/deploy_k8s.sh").read_text(encoding="utf-8")
    sync_script = Path("scripts/ci/sync_airflow_configmaps.sh").read_text(encoding="utf-8")

    imported_modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("taxon_vision.active_learning."):
            imported_modules.add(f"{node.module.rsplit('.', maxsplit=1)[-1]}.py")

    assert imported_modules, "Expected at least one taxon_vision.active_learning.* import in model_retraining.py"
    for filename in sorted(imported_modules):
        assert (Path("src/taxon_vision/active_learning") / filename).is_file()

    assert "scripts/ci/deploy_k8s.sh" in jenkinsfile
    assert "scripts/ci/sync_airflow_configmaps.sh" in deploy_script
    assert "airflow-active-learning" in sync_script
    assert "src/taxon_vision/active_learning/*.py" in sync_script

    deployments = [d for d in airflow_docs if isinstance(d, dict) and d.get("kind") == "Deployment"]
    assert len(deployments) == 2
    for dep in deployments:
        pod_spec = dep["spec"]["template"]["spec"]
        assert any(v.get("configMap", {}).get("name") == "airflow-active-learning" for v in pod_spec["volumes"])
        for container in pod_spec["containers"]:
            assert any(
                m.get("name") == "airflow-active-learning"
                and m.get("mountPath") == "/opt/airflow/src/taxon_vision/active_learning"
                and "subPath" not in m
                for m in container.get("volumeMounts", [])
            )


def test_api_and_airflow_shared_flagged_volume_permissions() -> None:
    """Verify api-deployment and Airflow coordinate shared group 10001 + setgid access without world-writable 777."""
    api_docs = list(yaml.safe_load_all(Path("deploy/k8s/base/api-deployment.yaml").read_text(encoding="utf-8")))
    airflow_docs = list(yaml.safe_load_all(Path("deploy/k8s/airflow.yaml").read_text(encoding="utf-8")))

    api_deploy = next(d for d in api_docs if isinstance(d, dict) and d.get("kind") == "Deployment")
    api_volumes = api_deploy["spec"]["template"]["spec"]["volumes"]
    assert any(v.get("hostPath", {}).get("path") == "/var/lib/taxon-vision/flagged_observations" for v in api_volumes)

    init_containers = api_deploy["spec"]["template"]["spec"]["initContainers"]
    init_cmd = " ".join(init_containers[0]["command"])
    assert "chmod -R g+rwX /app/data/flagged_observations" in init_cmd
    assert "chmod g+s /app/data/flagged_observations" in init_cmd
    assert "a+rwX" not in init_cmd

    scheduler_deploy = next(
        d
        for d in airflow_docs
        if isinstance(d, dict)
        and d.get("kind") == "Deployment"
        and d.get("metadata", {}).get("name") == "airflow-scheduler"
    )
    sched_spec = scheduler_deploy["spec"]["template"]["spec"]
    assert 10001 in sched_spec["securityContext"].get("supplementalGroups", [])
    assert any(
        v.get("hostPath", {}).get("path") == "/var/lib/taxon-vision/flagged_observations"
        for v in sched_spec.get("volumes", [])
    )


def test_docker_image_includes_passport_reports_not_dockerignored() -> None:
    """Verify serving Dockerfile copies tracked holdout reports and .dockerignore does not exclude them."""
    dockerfile = Path("deploy/docker/Dockerfile").read_text(encoding="utf-8")
    dockerignore_lines = {
        line.strip().rstrip("/")
        for line in Path(".dockerignore").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    }

    assert "reports" not in dockerignore_lines
    required_reports = (
        "reports/pareto_evaluation.json",
        "reports/train_metrics.json",
        "reports/conformal_metrics.json",
        "reports/data_drift.json",
    )
    for rel_path in required_reports:
        assert Path(rel_path).is_file(), f"Tracked report missing: {rel_path}"
        assert rel_path in dockerfile, f"Dockerfile must copy {rel_path} for ModelPassport"


def test_jenkins_skip_code_gates_wired_for_airflow_retrains() -> None:
    """Verify SKIP_CODE_GATES skips PR gates for Airflow/timer retrains whilst keeping manual RETRAIN_MODEL gated."""
    jenkinsfile = Path("Jenkinsfile").read_text(encoding="utf-8")
    retrain_dag = Path("airflow/dags/model_retraining.py").read_text(encoding="utf-8")

    assert "booleanParam(name: 'SKIP_CODE_GATES', defaultValue: false" in jenkinsfile
    assert jenkinsfile.count("expression { params.SKIP_CODE_GATES == true }") == 4
    assert jenkinsfile.count("triggeredBy 'TimerTrigger'") >= 5
    assert "SKIP_CODE_GATES=true" in retrain_dag


def test_airflow_private_helper_ignored_by_dagbag_and_synced_to_k8s() -> None:
    """Verify _jenkins_mlflow_client.py is excluded by .airflowignore whilst explicitly synced to K3s ConfigMaps."""
    airflowignore = Path("airflow/dags/.airflowignore").read_text(encoding="utf-8")
    sync_script = Path("scripts/ci/sync_airflow_configmaps.sh").read_text(encoding="utf-8")

    assert "^_.*\\.py$" in airflowignore
    assert ".airflowignore" in sync_script
    assert Path("airflow/dags/_jenkins_mlflow_client.py").is_file()
