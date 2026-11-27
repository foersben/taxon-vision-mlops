# SPDX-FileCopyrightText: 2026 Benjamin Förster
# SPDX-License-Identifier: MIT
"""Unit tests for monitoring, representation drift, and MMD hypothesis testing."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import ortho_group

from taxon_vision.monitoring.drift import (
    MMDResult,
    compute_mmd2_permutation,
    compute_wasserstein_drift,
)


def test_compute_wasserstein_drift() -> None:
    """Verify 1D Wasserstein approximation computes positive drift distance under shift."""
    ref = np.random.randn(100, 16)
    curr = np.random.randn(100, 16) + 2.0
    drift = compute_wasserstein_drift(ref, curr)
    assert drift > 0.0


def test_compute_wasserstein_drift_unnormalized_vs_normalized_paradox() -> None:
    """Verify that unnormalized activations detect magnitude shift, whereas unit vectors collapse to 0."""
    rng = np.random.default_rng(42)
    # Reference activations centered with scale 1.0, current activations shifted with scale 3.0
    ref_raw = rng.normal(loc=0.0, scale=1.0, size=(100, 32))
    curr_raw = rng.normal(loc=0.0, scale=3.0, size=(100, 32))

    # Pre-normalized activations preserve norm differences
    drift_unnorm = compute_wasserstein_drift(ref_raw, curr_raw)
    assert drift_unnorm > 1.0

    # If vectors were unit-normalized prior to drift calculation (the L2-norm paradox):
    ref_unit = ref_raw / np.linalg.norm(ref_raw, axis=-1, keepdims=True)
    curr_unit = curr_raw / np.linalg.norm(curr_raw, axis=-1, keepdims=True)
    drift_unit = compute_wasserstein_drift(ref_unit, curr_unit)

    # Unit vectors collapse to static 1.0 vs 1.0 comparison (drift ~ 0.0)
    assert np.isclose(drift_unit, 0.0, atol=1e-6)


def test_compute_mmd2_same_distribution() -> None:
    """Verify that independent draws from the same distribution yield p-value > 0.05."""
    rng = np.random.default_rng(42)
    ref = rng.standard_normal(size=(120, 16))
    curr = rng.standard_normal(size=(120, 16))

    result = compute_mmd2_permutation(ref, curr, max_samples=200, n_permutations=200, seed=42)
    assert isinstance(result, MMDResult)
    assert result.p_value > 0.05
    assert result.n_reference == 120
    assert result.n_current == 120
    assert result.bandwidth > 0.0


def test_compute_mmd2_mean_shift() -> None:
    """Verify that multivariate mean displacement triggers rejection of null hypothesis with p < 0.01."""
    rng = np.random.default_rng(42)
    ref = rng.standard_normal(size=(120, 16))
    curr = rng.standard_normal(size=(120, 16)) + 0.5

    result = compute_mmd2_permutation(ref, curr, max_samples=200, n_permutations=200, seed=42)
    assert result.p_value < 0.01
    assert result.mmd2 > 0.0


def test_compute_mmd2_rotation_blind_spot() -> None:
    """Verify that orthogonal manifold rotations are detected by MMD while invisible to 1D norm W1."""
    rng = np.random.default_rng(42)
    # Construct anisotropic covariance matrix (elongated in first dimension)
    cov = np.diag([10.0, 1.0, 0.1] + [1.0] * 13)
    ref = rng.multivariate_normal(np.zeros(16), cov, size=150)

    # Apply random orthogonal rotation matrix R in O(16)
    rot_matrix = ortho_group.rvs(16, random_state=42)
    curr_rotated = ref @ rot_matrix.T

    # Tier 1: Norms are invariant under orthogonal rotation (||Rx||_2 == ||x||_2)
    w1_dist = compute_wasserstein_drift(ref, curr_rotated)
    assert np.isclose(w1_dist, 0.0, atol=1e-6)

    # Tier 2: MMD detects the angular distribution displacement
    mmd_res = compute_mmd2_permutation(ref, curr_rotated, max_samples=200, n_permutations=200, seed=42)
    assert mmd_res.p_value < 0.01
    assert mmd_res.mmd2 > 0.0


def test_compute_mmd2_deterministic_seed() -> None:
    """Verify that identical inputs and seeds yield identical bitwise MMD results."""
    rng = np.random.default_rng(42)
    ref = rng.standard_normal(size=(80, 8))
    curr = rng.standard_normal(size=(80, 8)) + 0.2

    res1 = compute_mmd2_permutation(ref, curr, max_samples=50, n_permutations=100, seed=123)
    res2 = compute_mmd2_permutation(ref, curr, max_samples=50, n_permutations=100, seed=123)

    assert res1.mmd2 == res2.mmd2
    assert res1.p_value == res2.p_value
    assert res1.bandwidth == res2.bandwidth


def test_compute_mmd2_input_validation() -> None:
    """Verify that populations with fewer than 2 samples raise ValueError."""
    ref = np.ones((1, 8))
    curr = np.ones((10, 8))

    with pytest.raises(ValueError, match="at least two samples per group"):
        compute_mmd2_permutation(ref, curr)

    with pytest.raises(ValueError, match="at least two samples per group"):
        compute_mmd2_permutation(curr, ref)


def test_generate_evidently_drift_report_persists_workspace(tmp_path) -> None:
    """Persist Evidently snapshots into the configured UI workspace alongside standalone HTML output."""
    import numpy as np

    from taxon_vision.monitoring import drift_runner

    rng = np.random.default_rng(42)
    ref = rng.normal(0.0, 1.0, size=(30, 8)).astype(np.float32)
    cur = rng.normal(0.2, 1.0, size=(25, 8)).astype(np.float32)

    ws_dir = tmp_path / "evidently_ws"
    html_out = tmp_path / "reports" / "drift.html"
    drift_runner.generate_evidently_drift_report(
        ref,
        cur,
        html_out,
        workspace_dir=ws_dir,
        project_name="TaxonVision Drift Monitor",
    )

    assert html_out.is_file()
    assert ws_dir.is_dir()
    assert any(ws_dir.rglob("*.json"))


def test_structured_stdout_logging_handler_idempotent(monkeypatch: pytest.MonkeyPatch) -> None:
    """Configure stdout handlers with StructuredJsonFormatter when TAXON_STRUCTURED_STDOUT=true without duplicates."""
    import logging

    from taxon_vision.service.structured_logging import StructuredJsonFormatter, _ensure_structured_stdout_handler

    monkeypatch.setenv("TAXON_STRUCTURED_STDOUT", "true")
    isolated_logger = logging.getLogger("taxon_vision.test_stdout_isolation")
    isolated_logger.handlers.clear()
    _ensure_structured_stdout_handler(isolated_logger)
    _ensure_structured_stdout_handler(isolated_logger)
    assert len(isolated_logger.handlers) == 1
    assert isinstance(isolated_logger.handlers[0].formatter, StructuredJsonFormatter)


def _load_k8s_docs(relative_path: str) -> list[dict]:
    """Load non-empty YAML documents from a workspace-relative Kubernetes manifest path."""
    from pathlib import Path

    import yaml

    manifest_path = Path(__file__).resolve().parents[2] / relative_path
    return [doc for doc in yaml.safe_load_all(manifest_path.read_text(encoding="utf-8")) if isinstance(doc, dict)]


def _find_k8s_doc(docs: list[dict], kind: str, name: str) -> dict:
    """Locate a single Kubernetes manifest document by kind and metadata name."""
    matches = [doc for doc in docs if doc.get("kind") == kind and doc.get("metadata", {}).get("name") == name]
    assert len(matches) == 1, f"Expected exactly one {kind}/{name}, found {len(matches)}"
    return matches[0]


def _assert_prometheus_sd_job(prom_config_doc: dict, serving_doc: dict) -> None:
    """Verify per-pod Kubernetes SD relabelling matches the serving Deployment pod labels and port name."""
    import yaml

    prom_yml = yaml.safe_load(prom_config_doc["data"]["prometheus.yml"])
    serving_jobs = [job for job in prom_yml["scrape_configs"] if job.get("job_name") == "taxon-vision-serving"]
    assert len(serving_jobs) == 1
    job = serving_jobs[0]

    assert "static_configs" not in job
    assert job["kubernetes_sd_configs"] == [{"role": "pod", "namespaces": {"names": ["taxon-vision"]}}]

    pod_template = serving_doc["spec"]["template"]
    expected_app_label = pod_template["metadata"]["labels"]["app.kubernetes.io/name"]
    expected_port_name = pod_template["spec"]["containers"][0]["ports"][0]["name"]

    keep_rules = {
        tuple(rule["source_labels"]): rule["regex"] for rule in job["relabel_configs"] if rule.get("action") == "keep"
    }
    assert keep_rules[("__meta_kubernetes_pod_label_app_kubernetes_io_name",)] == expected_app_label
    assert keep_rules[("__meta_kubernetes_pod_container_port_name",)] == expected_port_name
    assert keep_rules[("__meta_kubernetes_pod_phase",)] == "Running"

    target_labels = {
        tuple(rule["source_labels"]): rule["target_label"] for rule in job["relabel_configs"] if "target_label" in rule
    }
    assert target_labels[("__meta_kubernetes_pod_name",)] == "pod"
    assert target_labels[("__meta_kubernetes_namespace",)] == "namespace"


def _assert_prometheus_rbac(docs: list[dict]) -> None:
    """Verify namespace-scoped read-only RBAC for Prometheus pod discovery without ClusterRole escalation."""
    assert not [doc for doc in docs if doc.get("kind") in {"ClusterRole", "ClusterRoleBinding"}]

    sa = _find_k8s_doc(docs, "ServiceAccount", "prometheus")
    assert sa["metadata"]["namespace"] == "taxon-vision"

    role = _find_k8s_doc(docs, "Role", "prometheus-pod-sd")
    assert role["metadata"]["namespace"] == "taxon-vision"
    assert role["rules"] == [{"apiGroups": [""], "resources": ["pods"], "verbs": ["get", "list", "watch"]}]

    binding = _find_k8s_doc(docs, "RoleBinding", "prometheus-pod-sd")
    assert binding["metadata"]["namespace"] == "taxon-vision"
    assert binding["roleRef"] == {
        "apiGroup": "rbac.authorization.k8s.io",
        "kind": "Role",
        "name": "prometheus-pod-sd",
    }
    assert binding["subjects"] == [{"kind": "ServiceAccount", "name": "prometheus", "namespace": "taxon-vision"}]

    prom_deploy = _find_k8s_doc(docs, "Deployment", "prometheus")
    assert prom_deploy["spec"]["template"]["spec"]["serviceAccountName"] == "prometheus"


def test_prometheus_k8s_sd_matches_serving_pod_manifest_and_rbac() -> None:
    """Ensure Prometheus discovers every serving pod directly with least-privilege namespace-scoped RBAC."""
    monitoring_docs = _load_k8s_docs("deploy/k8s/monitoring-prometheus-grafana.yaml")
    serving_docs = _load_k8s_docs("deploy/k8s/base/api-deployment.yaml")

    prom_config_doc = _find_k8s_doc(monitoring_docs, "ConfigMap", "prometheus-config")
    serving_doc = _find_k8s_doc(serving_docs, "Deployment", "taxon-vision-serving")

    _assert_prometheus_sd_job(prom_config_doc, serving_doc)
    _assert_prometheus_rbac(monitoring_docs)

    for deploy_name in ("prometheus", "grafana"):
        dep = _find_k8s_doc(monitoring_docs, "Deployment", deploy_name)
        assert dep["spec"]["strategy"]["type"] == "Recreate"
        pod_spec = dep["spec"]["template"]["spec"]
        for container in [*pod_spec.get("initContainers", []), *pod_spec.get("containers", [])]:
            assert not str(container["image"]).endswith(":latest")


_KIND_TO_RBAC_RESOURCE: dict[str, tuple[str, str]] = {
    "ConfigMap": ("", "configmaps"),
    "Secret": ("", "secrets"),
    "Service": ("", "services"),
    "ServiceAccount": ("", "serviceaccounts"),
    "PersistentVolumeClaim": ("", "persistentvolumeclaims"),
    "Deployment": ("apps", "deployments"),
    "DaemonSet": ("apps", "daemonsets"),
    "StatefulSet": ("apps", "statefulsets"),
    "Role": ("rbac.authorization.k8s.io", "roles"),
    "RoleBinding": ("rbac.authorization.k8s.io", "rolebindings"),
    "IngressRoute": ("traefik.io", "ingressroutes"),
    "TraefikService": ("traefik.io", "traefikservices"),
    "Middleware": ("traefik.io", "middlewares"),
}


def _collect_role_permissions(role_doc: dict) -> dict[tuple[str, str], set[str]]:
    """Expand a Kubernetes Role spec into an (apiGroup, resource) -> verbs lookup table."""
    granted: dict[tuple[str, str], set[str]] = {}
    for rule in role_doc.get("rules", []):
        verbs = set(rule.get("verbs", []))
        for group in rule.get("apiGroups", []):
            for resource in rule.get("resources", []):
                granted.setdefault((group, resource), set()).update(verbs)
    return granted


def test_jenkins_deployer_rbac_covers_applied_manifests() -> None:
    """Ensure Jenkins deployer RBAC covers every Kubernetes resource kind applied in Jenkinsfile."""
    import re
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    jenkinsfile_text = (root / "Jenkinsfile").read_text(encoding="utf-8")
    applied_manifests = sorted(set(re.findall(r"kubectl apply -f (deploy/k8s/[^\s]+)", jenkinsfile_text)))
    assert "deploy/k8s/monitoring-prometheus-grafana.yaml" in applied_manifests

    rbac_docs = _load_k8s_docs("deploy/k8s/jenkins-agent-rbac.yaml")
    prod_roles = [
        doc
        for doc in rbac_docs
        if doc.get("kind") == "Role"
        and doc.get("metadata", {}).get("name") == "jenkins-deployer-role"
        and doc.get("metadata", {}).get("namespace") == "taxon-vision"
    ]
    assert len(prod_roles) == 1
    granted = _collect_role_permissions(prod_roles[0])

    required_verbs = {"get", "create", "patch", "update"}
    for manifest_rel in applied_manifests:
        for doc in _load_k8s_docs(manifest_rel):
            kind = str(doc["kind"])
            assert kind in _KIND_TO_RBAC_RESOURCE, f"Unmapped kind {kind} in {manifest_rel}"
            key = _KIND_TO_RBAC_RESOURCE[kind]
            assert required_verbs.issubset(granted.get(key, set())), (
                f"Role jenkins-deployer-role lacks {required_verbs} on {key} needed by {manifest_rel}"
            )
