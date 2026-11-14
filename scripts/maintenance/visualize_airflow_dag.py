# SPDX-FileCopyrightText: 2026 Benjamin Förster
# SPDX-License-Identifier: MIT
"""Offline visualizer for Apache Airflow DAG topologies.

Summary:
    Parses Airflow DAG source files using Python's abstract syntax tree (AST) to render clean, zero-dependency terminal tree diagrams without requiring active container runtimes.

Why:
    Running live Airflow CLI commands (`airflow tasks list ... --tree`) requires active scheduler containers and PostgreSQL connections. During fast iterative development, developers and CI runners require an instant, zero-residue inspection tool that runs in milliseconds without starting background containers or consuming memory.

How:
    Extracts AST function definitions, TaskFlow decorators, parameters, and downstream operator dependencies from `airflow/dags/*.py`, formatting them into structured ASCII box-and-branch diagrams matching `dvc dag` ergonomics.
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


def _apply_dag_kwarg(kw: ast.keyword, dag_meta: DagTopology) -> None:
    """Apply a single @dag keyword argument onto the topology metadata."""
    if kw.arg == "tags" and isinstance(kw.value, ast.List):
        dag_meta.tags = [str(el.value) for el in kw.value.elts if isinstance(el, ast.Constant)]
        return
    if not isinstance(kw.value, ast.Constant):
        return
    if kw.arg == "dag_id":
        dag_meta.dag_id = str(kw.value.value)
    elif kw.arg == "schedule":
        dag_meta.schedule = str(kw.value.value)
    elif kw.arg == "catchup":
        dag_meta.catchup = bool(kw.value.value)
    elif kw.arg == "max_active_runs" and isinstance(kw.value.value, (int, str)):
        dag_meta.max_active_runs = int(kw.value.value)


def _extract_dag_metadata(decorator: ast.Call, dag_meta: DagTopology) -> None:
    """Extract keyword arguments from the @dag decorator call.

    Args:
        decorator: AST Call node corresponding to @dag.
        dag_meta: Mutable DagTopology instance to populate.
    """
    for kw in decorator.keywords:
        _apply_dag_kwarg(kw, dag_meta)


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


def _find_dag_function(tree: ast.Module) -> tuple[ast.FunctionDef, ast.Call] | None:
    """Locate the first top-level function decorated with @dag(...)."""
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef):
            continue
        for dec in node.decorator_list:
            if isinstance(dec, ast.Call) and isinstance(dec.func, ast.Name) and dec.func.id == "dag":
                return node, dec
    return None


def _extract_operator_nodes(subnode: ast.Assign) -> list[DagTaskNode]:
    """Extract Airflow Operator instantiations from an assignment statement."""
    if not (isinstance(subnode.value, ast.Call) and isinstance(subnode.value.func, ast.Name)):
        return []
    call_id = subnode.value.func.id
    if "Operator" not in call_id:
        return []
    return [
        DagTaskNode(name=target.id, kind="Operator", description=f"Instantiates {call_id}")
        for target in subnode.targets
        if isinstance(target, ast.Name)
    ]


def _extract_dag_nodes(node: ast.FunctionDef) -> list[DagTaskNode]:
    """Extract nested task functions and operator assignments from a @dag body."""
    nodes: list[DagTaskNode] = []
    for subnode in node.body:
        if isinstance(subnode, ast.FunctionDef):
            kind, desc = _determine_node_kind(subnode)
            nodes.append(DagTaskNode(name=subnode.name, kind=kind, description=desc))
        elif isinstance(subnode, ast.Assign):
            nodes.extend(_extract_operator_nodes(subnode))
    return nodes


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

    dag_match = _find_dag_function(tree)
    if dag_match is None:
        return None
    node, dec = dag_match
    topology = DagTopology(dag_id=node.name, file_path=file_path)
    _extract_dag_metadata(dec, topology)
    topology.nodes.extend(_extract_dag_nodes(node))
    return topology


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


def _collect_dag_topologies(dags_dir: Path) -> list[DagTopology]:
    """Parse all Python DAG definitions inside the target directory."""
    topologies: list[DagTopology] = []
    for f in sorted(dags_dir.glob("*.py")):
        parsed = _parse_dag_file(f)
        if parsed is not None:
            topologies.append(parsed)
    return topologies


def _render_filtered_dag(topologies: list[DagTopology], filter_dag: str) -> int:
    """Render a single matching DAG topology or report available IDs."""
    matched = [t for t in topologies if t.dag_id == filter_dag]
    if not matched:
        available = ", ".join(t.dag_id for t in topologies)
        print(f"Error: DAG '{filter_dag}' not found.", file=sys.stderr)
        print(f"Available DAGs: {available}", file=sys.stderr)
        return 1
    for t in matched:
        _render_dag(t)
    return 0


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

    topologies = _collect_dag_topologies(DAGS_DIR)
    if not topologies:
        print(f"No valid Airflow DAGs found in {DAGS_DIR}.", file=sys.stderr)
        return 1

    if filter_dag:
        return _render_filtered_dag(topologies, filter_dag)

    print(f"TaxonVision Airflow Topology ({len(topologies)} registered workflows):\n")
    for t in topologies:
        _render_dag(t)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
