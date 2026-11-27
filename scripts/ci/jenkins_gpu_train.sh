#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Benjamin Förster
# SPDX-License-Identifier: MIT
# GPU training, evaluation, and artefact persistence stage executed inside Jenkins K3s agent pods.

set -euo pipefail

export PATH="/root/.pixi/bin:$PATH"
export PYTHONUNBUFFERED=1
export TAXON_OPTUNA_STORAGE_URI="${TAXON_OPTUNA_STORAGE_URI:-sqlite:////workspace/optuna/optuna_studies.db}"

USE_LOCAL_SEAWEEDFS=false
CATALOG_EXPANDED=false
PIXI_ENV="dev"

# shellcheck source=scripts/ci/hydrate_and_stage_checkpoints.sh
source "$(dirname "${BASH_SOURCE[0]}")/hydrate_and_stage_checkpoints.sh"

trap 'cleanup_dvc_local "dev"' EXIT

setup_toolchain() {
    echo ">>> Setting up GPU training pipeline..."
    which git >/dev/null 2>&1 || pixi global install git
    which kubectl >/dev/null 2>&1 || pixi global install kubernetes-client
    git config --global --add safe.directory "*"
    pixi install --frozen -e dev
}

prime_host_caches() {
    mkdir -p data/embeddings data/manifests data/reference models/checkpoints

    if [ -f "/var/lib/taxon-vision/embeddings/train_embeddings.pt" ]; then
        echo ">>> Found cached features on host tier; priming data/embeddings/*.pt..."
        cp -f /var/lib/taxon-vision/embeddings/*.pt data/embeddings/ 2>/dev/null || true
    fi
    if [ -f "/var/lib/taxon-vision/manifests/raw_manifest.parquet" ]; then
        echo ">>> Found cached manifests on host tier; priming data/manifests/*.parquet..."
        cp -f /var/lib/taxon-vision/manifests/*.parquet data/manifests/ 2>/dev/null || true
    fi
    if [ -f "/var/lib/taxon-vision/reference/reference_train_embeddings.pt" ]; then
        echo ">>> Found cached reference on host tier; priming data/reference/*.pt..."
        cp -f /var/lib/taxon-vision/reference/*.pt data/reference/ 2>/dev/null || true
    fi
    if [ -f "/var/lib/taxon-vision/checkpoints/head.pt" ]; then
        echo ">>> Found cached checkpoints on host tier; priming models/checkpoints/*..."
        cp -f /var/lib/taxon-vision/checkpoints/* models/checkpoints/ 2>/dev/null || true
    fi
}

hydrate_missing_dvc_artifacts() {
    local -a missing_artifacts=()
    local artifact pulled=false

    for artifact in \
        "data/manifests/raw_manifest.parquet" \
        "data/manifests/dataset_manifest.parquet" \
        "data/embeddings/train_embeddings.pt" \
        "data/embeddings/val_embeddings.pt" \
        "data/embeddings/cal_embeddings.pt" \
        "data/embeddings/test_embeddings.pt" \
        "data/reference/reference_train_embeddings.pt" \
        "models/checkpoints/head.pt" \
        "models/checkpoints/model_int8.onnx"; do
        [ ! -f "${artifact}" ] && missing_artifacts+=("${artifact}")
    done

    if [ ${#missing_artifacts[@]} -gt 0 ]; then
        echo ">>> Missing artifacts in workspace: ${missing_artifacts[*]}"
        if [ "${USE_LOCAL_SEAWEEDFS}" = "true" ] && kubectl get service seaweedfs -n taxon-vision >/dev/null 2>&1; then
            echo ">>> Attempting hydration from in-cluster SeaweedFS (local-seaweedfs)..."
            if env -u AWS_ACCESS_KEY_ID -u AWS_SECRET_ACCESS_KEY pixi run --frozen -e dev dvc pull -r local-seaweedfs "${missing_artifacts[@]}"; then
                pulled=true
                echo ">>> Successfully hydrated missing artifacts from local-seaweedfs."
            fi
        fi

        if [ "${pulled}" != "true" ]; then
            echo ">>> Attempting hydration from DagsHub cloud archive (origin)..."
            pixi run --frozen -e dev dvc pull -r origin "${missing_artifacts[@]}" || \
            pixi run --frozen -e dev dvc pull "${missing_artifacts[@]}" || \
            echo ">>> Warning: Could not pull all missing artifacts from DVC remotes."
        fi
    fi

    if [ ! -f "data/manifests/raw_manifest.parquet" ]; then
        echo ">>> raw_manifest.parquet missing from all caches and remotes; forcing INGEST_DATA=true."
        export INGEST_DATA=true
    fi

    if [ ! -f "data/reference/reference_train_embeddings.pt" ] && [ -f "data/embeddings/train_embeddings.pt" ]; then
        echo ">>> reference_train_embeddings.pt missing; bootstrapping reference baseline from train_embeddings.pt..."
        mkdir -p data/reference
        cp -f data/embeddings/train_embeddings.pt data/reference/reference_train_embeddings.pt
    fi
}

apply_runtime_params_and_expansions() {
    echo ">>> Applying parameter overrides to params.yaml..."
    pixi run --frozen -e dev python -c '
import os, yaml
from pathlib import Path

p = Path("params.yaml")
data = yaml.safe_load(p.read_text())
changed = False

if os.environ.get("TUNE_HYPERPARAMETERS") == "true":
    data.setdefault("training", {})["tune"] = True
    changed = True

extractor = os.environ.get("EXTRACTOR", "")
if extractor and extractor != data.get("model", {}).get("extractor"):
    data.setdefault("model", {})["extractor"] = extractor
    changed = True

if changed:
    p.write_text(yaml.safe_dump(data, sort_keys=False))
'

    local expansions_file="/var/lib/taxon-vision/manifests/catalog_expansions.yaml"
    if [ -s "${expansions_file}" ] && grep -q "inat_taxon_id:" "${expansions_file}"; then
        echo ">>> Applying pending Outcome B catalogue expansions from ${expansions_file}..."
        pixi run --frozen -e dev python -m taxon_vision.active_learning.feedback \
            --apply-expansions "${expansions_file}" \
            --catalog-path config/taxa_catalog.yaml \
            --params-path params.yaml \
            --label-studio-config-path deploy/label_studio/project_config.xml
        CATALOG_EXPANDED=true
        export INGEST_DATA=true
    fi
}

run_dvc_pipeline() {
    echo ">>> Executing unified 7-stage DVC pipeline..."
    if [ "${INGEST_DATA:-false}" = "true" ]; then
        echo ">>> INGEST_DATA is true: Unfreezing ingestion stage and fetching fresh data..."
        trap 'echo ">>> Ingestion failed! Restoring dvc.yaml frozen state..."; git checkout -- dvc.yaml 2>/dev/null || true' ERR
        pixi run --frozen -e dev dvc unfreeze ingestion
        pixi run --frozen -e dev dvc repro ingestion
        pixi run --frozen -e dev dvc freeze ingestion
        trap - ERR
    else
        echo ">>> Using existing cached raw_manifest.parquet (ingestion stage remains frozen)."
        pixi run --frozen -e dev dvc commit -f ingestion >/dev/null 2>&1 || true
    fi

    pixi run --frozen -e dev dvc repro

    if [ "${CATALOG_EXPANDED}" = "true" ] && [ -n "${DAGSHUB_TOKEN:-}" ]; then
        echo ">>> Synchronising expanded Label Studio XML configuration to DagsHub..."
        pixi run --frozen -e dev python -c "from taxon_vision.active_learning.label_studio_sync import DagsHubLabelStudioClient; DagsHubLabelStudioClient().push_project_label_config('deploy/label_studio/project_config.xml')" || echo "Warning: DagsHub Label Studio XML push skipped."
    fi
}

persist_host_volumes_and_remotes() {
    if [ -d "/var/lib/taxon-vision/embeddings" ] && [ -f "data/embeddings/train_embeddings.pt" ]; then
        cp -f data/embeddings/*.pt /var/lib/taxon-vision/embeddings/ 2>/dev/null || true
    fi

    if [ -d "/var/lib/taxon-vision/reference" ]; then
        if [ -f "data/reference/reference_train_embeddings.pt" ]; then
            cp -f data/reference/*.pt /var/lib/taxon-vision/reference/ 2>/dev/null || true
        fi
        atomic_stage_file "data/embeddings/cal_embeddings.pt" "/var/lib/taxon-vision/reference"
    fi

    if [ -d "/var/lib/taxon-vision/manifests" ] && [ -f "data/manifests/raw_manifest.parquet" ]; then
        for manfile in data/manifests/*.parquet; do
            atomic_stage_file "${manfile}" "/var/lib/taxon-vision/manifests"
        done
    fi

    if [ -d "/var/lib/taxon-vision/checkpoints" ] && [ -f "models/checkpoints/head.pt" ]; then
        echo ">>> Atomically staging checkpoints and co-versioned configs to /var/lib/taxon-vision/checkpoints/..."
        for ckpt in models/checkpoints/* config/taxa_catalog.yaml config/conformal_config.yaml dvc.lock; do
            atomic_stage_file "${ckpt}" "/var/lib/taxon-vision/checkpoints"
        done
    fi

    if [ "${USE_LOCAL_SEAWEEDFS}" = "true" ] && kubectl get service seaweedfs -n taxon-vision >/dev/null 2>&1; then
        echo ">>> Synchronizing Epistemic Diamond artifacts to SeaweedFS (local-seaweedfs)..."
        env -u AWS_ACCESS_KEY_ID -u AWS_SECRET_ACCESS_KEY pixi run --frozen -e dev dvc push -r local-seaweedfs || echo "SeaweedFS push skipped."
    fi

    echo ">>> Pushing all Epistemic Diamond artifacts to DagsHub (origin)..."
    pixi run --frozen -e dev dvc push -r origin
    echo ">>> Pipeline artefacts synced to host volume, SeaweedFS, and DagsHub."
}

main() {
    setup_toolchain
    configure_seaweedfs_remote
    prime_host_caches
    hydrate_missing_dvc_artifacts
    apply_runtime_params_and_expansions
    run_dvc_pipeline
    persist_host_volumes_and_remotes
}

main
