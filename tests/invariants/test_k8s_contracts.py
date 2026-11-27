# SPDX-FileCopyrightText: 2026 Benjamin Förster
# SPDX-License-Identifier: MIT
"""Invariant tests enforcing Kubernetes deployment contracts across deploy/k8s/ and Jenkinsfile."""

from __future__ import annotations

import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
K8S_DIR = ROOT / "deploy" / "k8s"

_ALLOWED_HOSTPATH_ROOTS = (
    "/var/lib/taxon-vision/",
    "/var/log/pods",
    "/var/lib/rancher/k3s/agent/containerd/",
)


def _iter_k8s_docs() -> list[tuple[Path, dict]]:
    """Load all YAML documents under deploy/k8s/."""
    docs: list[tuple[Path, dict]] = []
    for path in sorted(K8S_DIR.rglob("*.yaml")):
        for doc in yaml.safe_load_all(path.read_text(encoding="utf-8")):
            if isinstance(doc, dict):
                docs.append((path.relative_to(ROOT), doc))
    return docs


def test_no_latest_image_tags_in_k8s_or_jenkins() -> None:
    """Ensure all Kubernetes workloads and Jenkins pod templates pin explicit immutable image tags."""
    offenders: list[str] = []
    for path in sorted(K8S_DIR.rglob("*.yaml")):
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if re.match(r"^\s*image:\s*\S+:latest\s*$", line):
                offenders.append(f"{path.relative_to(ROOT)}:{lineno}: {line.strip()}")

    jenkins_text = (ROOT / "Jenkinsfile").read_text(encoding="utf-8")
    for lineno, line in enumerate(jenkins_text.splitlines(), 1):
        if re.match(r"^\s*image:\s*\S+:latest\s*$", line):
            offenders.append(f"Jenkinsfile:{lineno}: {line.strip()}")

    assert not offenders, f"Mutable ':latest' image tags found: {offenders}"


def test_nodeports_are_unique_across_services() -> None:
    """Ensure no two Kubernetes Services claim the same NodePort."""
    seen: dict[int, str] = {}
    for rel_path, doc in _iter_k8s_docs():
        if doc.get("kind") != "Service" or "base" in rel_path.parts:
            continue
        for port in doc.get("spec", {}).get("ports", []):
            node_port = port.get("nodePort")
            if node_port is None:
                continue
            owner = f"{rel_path}:{doc.get('metadata', {}).get('name')}"
            assert node_port not in seen, f"NodePort {node_port} collision between {seen[node_port]} and {owner}"
            seen[node_port] = owner


def test_non_root_security_contexts_and_hostpath_roots() -> None:
    """Verify mandatory non-root UIDs (10001/50000/70) and canonical hostPath prefixes."""
    expected_uids = {
        "api-deployment.yaml": 10001,
        "airflow.yaml": 50000,
        "postgres.yaml": 70,
    }
    for rel_path, doc in _iter_k8s_docs():
        if doc.get("kind") not in {"Deployment", "DaemonSet", "Pod"}:
            continue
        pod_spec = doc["spec"]["template"]["spec"] if "template" in doc["spec"] else doc["spec"]

        if rel_path.name in expected_uids:
            uid = pod_spec.get("securityContext", {}).get("runAsUser")
            assert uid == expected_uids[rel_path.name], (
                f"{rel_path} expected runAsUser={expected_uids[rel_path.name]}, got {uid}"
            )

        for vol in pod_spec.get("volumes", []):
            hp = vol.get("hostPath", {}).get("path")
            if hp is not None:
                assert hp.startswith(_ALLOWED_HOSTPATH_ROOTS), f"Unexpected hostPath {hp} in {rel_path}"
