#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Benjamin Förster
# SPDX-License-Identifier: MIT
# Synchronise Airflow 3 DAGs, ignore rules, and active_learning helpers into Kubernetes ConfigMaps.

set -euo pipefail

NAMESPACE="${1:-taxon-vision}"

sync_configmap_files() {
    local cm_name="$1"
    shift
    local -a args=()
    local f
    for f in "$@"; do
        [ -f "$f" ] && args+=("--from-file=${f}")
    done
    kubectl create configmap "${cm_name}" "${args[@]}" -n "${NAMESPACE}" --dry-run=client -o yaml | kubectl apply -f -
}

dag_files=(airflow/dags/*.py)
[ -f "airflow/dags/.airflowignore" ] && dag_files+=("airflow/dags/.airflowignore")

sync_configmap_files "airflow-dags" "${dag_files[@]}"
sync_configmap_files "airflow-active-learning" src/taxon_vision/active_learning/*.py
kubectl create configmap airflow-taxon-deps \
    --from-file=taxa_catalog.yaml=config/taxa_catalog.yaml \
    -n "${NAMESPACE}" --dry-run=client -o yaml | kubectl apply -f -
