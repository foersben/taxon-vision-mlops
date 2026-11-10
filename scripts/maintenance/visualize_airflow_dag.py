# SPDX-FileCopyrightText: 2026 Benjamin Förster
# SPDX-License-Identifier: MIT
"""Offline visualizer for Apache Airflow DAG topologies.

Summary:
    Parses Airflow DAG source files using Python's abstract syntax tree (AST) to render
    clean, zero-dependency terminal tree diagrams without requiring active container runtimes.

Why:
    Running live Airflow CLI commands (`airflow tasks list ... --tree`) requires active
    scheduler containers and PostgreSQL connections. During fast iterative development,
    developers and CI runners require an instant, zero-residue inspection tool that runs
    in milliseconds without starting background containers or consuming memory.

How:
    Extracts AST function definitions, TaskFlow decorators, parameters, and downstream
    operator dependencies from `airflow/dags/*.py`, formatting them into structured ASCII
    box-and-branch diagrams matching `dvc dag` ergonomics.
"""

from __future__ import annotations

import ast
import sys
from dataclasses import dataclass, field
from pathlib import Path

DAGS_DIR = Path("airflow/dags")


@dataclass(frozen=True)
class DagTaskNode:
    """Represents a single task node within an Airflow DAG topology."""

    name: str
    kind: str
    description: str


@dataclass
class DagTopology:
    """Encapsulates metadata and task hierarchy of an Airflow DAG."""

    dag_id: str
    file_path: Path
    schedule: str = "None"
    catchup: bool = False
    max_active_runs: int = 1
    tags: list[str] = field(default_factory=list)
    nodes: list[DagTaskNode] = field(default_factory=list)


def _extract_dag_metadata(decorator: ast.Call, dag_meta: DagTopology) -> None:
    """Extract keyword arguments from the @dag decorator call.

    Args:
        decorator: AST Call node corresponding to @dag.
        dag_meta: Mutable DagTopology instance to populate.
    """
    for kw in decorator.keywords:
        if kw.arg == "dag_id" and isinstance(kw.value, ast.Constant):
            dag_meta.dag_id = str(kw.value.value)
        elif kw.arg == "schedule" and isinstance(kw.value, ast.Constant):
            dag_meta.schedule = str(kw.value.value)
        elif kw.arg == "catchup" and isinstance(kw.value, ast.Constant):
            dag_meta.catchup = bool(kw.value.value)
        elif kw.arg == "max_active_runs" and isinstance(kw.value, ast.Constant):
            if isinstance(kw.value.value, (int, str)):
                dag_meta.max_active_runs = int(kw.value.value)
        elif kw.arg == "tags" and isinstance(kw.value, ast.List):
            dag_meta.tags = [str(el.value) for el in kw.value.elts if isinstance(el, ast.Constant)]


def _determine_node_kind(node: ast.FunctionDef) -> tuple[str, str]:
    """Determine the task decorator kind and summary description.

    Args:
        node: AST FunctionDef node.

    Returns:
        Tuple of (kind_label, description).
    """
    kind = "Task"
    for dec in node.decorator_list:
        if isinstance(dec, ast.Attribute) and dec.attr == "short_circuit":
            kind = "Short-Circuit"
            break

    doc = ast.get_docstring(node)
    desc = doc.split("\n")[0].strip() if doc else "Task execution unit"
    return kind, desc


def _parse_dag_file(file_path: Path) -> DagTopology | None:
    """Parse a single Airflow DAG Python file into a structured DagTopology.

    Args:
        file_path: Path to the Python DAG script.

    Returns:
        Populated DagTopology, or None if no valid @dag definition is found.
    """
    try:
        tree = ast.parse(file_path.read_text(encoding="utf-8"), filename=str(file_path))
    except (SyntaxError, UnicodeDecodeError):
        return None

    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            for dec in node.decorator_list:
                if isinstance(dec, ast.Call) and getattr(dec.func, "id", None) == "dag":
                    topology = DagTopology(dag_id=node.name, file_path=file_path)
                    _extract_dag_metadata(dec, topology)

                    # Extract nested tasks and operators
                    for subnode in node.body:
                        if isinstance(subnode, ast.FunctionDef):
                            kind, desc = _determine_node_kind(subnode)
                            topology.nodes.append(DagTaskNode(name=subnode.name, kind=kind, description=desc))
                        elif isinstance(subnode, ast.Assign):
                            for target in subnode.targets:
                                if isinstance(target, ast.Name) and isinstance(subnode.value, ast.Call):
                                    call_id = getattr(subnode.value.func, "id", "")
                                    if "Operator" in call_id:
                                        topology.nodes.append(
                                            DagTaskNode(
                                                name=target.id,
                                                kind="Operator",
                                                description=f"Instantiates {call_id}",
                                            )
                                        )
                    return topology
    return None


def _render_dag(topology: DagTopology) -> None:
    """Print an ASCII visual representation of an Airflow DAG.

    Args:
        topology: Structured DAG topology to render.
    """
    banner = f" DAG: {topology.dag_id} "
    width = 78
    print("┌" + "─" * (width - 2) + "┐")
    print(f"│{banner:<{width - 2}}│")
    sched_info = (
        f" Schedule: {topology.schedule} | Catchup: {topology.catchup} | Max Active Runs: {topology.max_active_runs}"
    )
    print(f"│{sched_info:<{width - 2}}│")
    tag_info = f" Tags: {', '.join(topology.tags) if topology.tags else 'None'}"
    print(f"│{tag_info:<{width - 2}}│")
    file_info = f" File: {topology.file_path}"
    print(f"│{file_info:<{width - 2}}│")
    print("└" + "─" * (width - 2) + "┘")

    if not topology.nodes:
        print("  └── (No tasks defined)\n")
        return

    for idx, task_node in enumerate(topology.nodes):
        is_last = idx == len(topology.nodes) - 1
        branch = "└──" if is_last else "├──"
        pipe = "   " if is_last else "│  "
        print(f"  {branch} [{task_node.kind}] {task_node.name}")
        print(f"  {pipe}   └─ {task_node.description}")
    print()


def main(argv: list[str]) -> int:
    """Entry point for DAG visualization CLI.

    Args:
        argv: Command line argument list.

    Returns:
        Integer exit code.
    """
    filter_dag = argv[1].strip() if len(argv) > 1 else ""

    if not DAGS_DIR.is_dir():
        print(f"Error: Airflow DAGs directory '{DAGS_DIR}' not found.", file=sys.stderr)
        return 1

    dag_files = sorted(DAGS_DIR.glob("*.py"))
    topologies: list[DagTopology] = []

    for f in dag_files:
        parsed = _parse_dag_file(f)
        if parsed is not None:
            topologies.append(parsed)

    if not topologies:
        print(f"No valid Airflow DAGs found in {DAGS_DIR}.", file=sys.stderr)
        return 1

    if filter_dag:
        matched = [t for t in topologies if t.dag_id == filter_dag]
        if not matched:
            available = ", ".join(t.dag_id for t in topologies)
            print(f"Error: DAG '{filter_dag}' not found.", file=sys.stderr)
            print(f"Available DAGs: {available}", file=sys.stderr)
            return 1
        for t in matched:
            _render_dag(t)
        return 0

    print(f"TaxonVision Airflow Topology ({len(topologies)} registered workflows):\n")
    for t in topologies:
        _render_dag(t)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
