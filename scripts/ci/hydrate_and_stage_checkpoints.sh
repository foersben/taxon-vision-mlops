#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Benjamin Förster
# SPDX-License-Identifier: MIT
# Shared DVC SeaweedFS/DagsHub hydration and atomic host-volume staging helpers.
# Executed standalone by deploy_k8s.sh or sourced by jenkins_gpu_train.sh.

set -euo pipefail

PIXI_ENV="${PIXI_ENV:-ci-dev}"

cleanup_dvc_local() {
    local env_name="${1:-${PIXI_ENV}}"
    pixi run --frozen -e "${env_name}" dvc config --local --unset remote.local-seaweedfs.access_key_id >/dev/null 2>&1 || true
    pixi run --frozen -e "${env_name}" dvc config --local --unset remote.local-seaweedfs.secret_access_key >/dev/null 2>&1 || true
}

# Copy one file atomically (.tmp + chmod a+r + mv) so UID 10001 serving pods never observe partial writes
atomic_stage_file() {
    local src="$1"
    local dest_dir="$2"
    [ -f "${src}" ] || return 0

    local fname tmp_path
    fname=$(basename "${src}")
    tmp_path="${dest_dir}/.${fname}.tmp"
    cp -f "${src}" "${tmp_path}"
    chmod a+r "${tmp_path}"
    mv -f "${tmp_path}" "${dest_dir}/${fname}"
}

configure_seaweedfs_remote() {
    local env_name="${1:-${PIXI_ENV}}"
    local k8s_key k8s_secret
    USE_LOCAL_SEAWEEDFS=false

    k8s_key="$(kubectl get secret seaweedfs-s3-secret -n taxon-vision -o jsonpath='{.data.AWS_ACCESS_KEY_ID}' 2>/dev/null | base64 -d 2>/dev/null || true)"
    k8s_secret="$(kubectl get secret seaweedfs-s3-secret -n taxon-vision -o jsonpath='{.data.AWS_SECRET_ACCESS_KEY}' 2>/dev/null | base64 -d 2>/dev/null || true)"
    if [ -n "${k8s_key}" ] && [ -n "${k8s_secret}" ]; then
        SEAWEEDFS_ACCESS_KEY_ID="${k8s_key}"
        SEAWEEDFS_SECRET_ACCESS_KEY="${k8s_secret}"
    fi

    if [ -n "${SEAWEEDFS_ACCESS_KEY_ID:-}" ] && [ -n "${SEAWEEDFS_SECRET_ACCESS_KEY:-}" ]; then
        pixi run --frozen -e "${env_name}" dvc config --local remote.local-seaweedfs.access_key_id "${SEAWEEDFS_ACCESS_KEY_ID}"
        pixi run --frozen -e "${env_name}" dvc config --local remote.local-seaweedfs.secret_access_key "${SEAWEEDFS_SECRET_ACCESS_KEY}"
        USE_LOCAL_SEAWEEDFS=true
    else
        echo ">>> Warning: SEAWEEDFS_ACCESS_KEY_ID/SECRET_ACCESS_KEY unset; skipping in-cluster SeaweedFS DVC sync."
    fi
}

pull_dvc_artifacts_with_fallback() {
    local env_name="$1"
    shift
    local pulled=false

    if [ "${USE_LOCAL_SEAWEEDFS:-false}" = "true" ] && kubectl get service seaweedfs -n taxon-vision >/dev/null 2>&1; then
        echo ">>> Attempting hydration from in-cluster SeaweedFS (local-seaweedfs)..."
        if env -u AWS_ACCESS_KEY_ID -u AWS_SECRET_ACCESS_KEY pixi run --frozen -e "${env_name}" dvc pull --force -r local-seaweedfs "$@"; then
            pulled=true
            echo ">>> Successfully hydrated artifacts from local-seaweedfs."
        fi
    fi

    if [ "${pulled}" != "true" ]; then
        echo ">>> Pulling artifacts from DagsHub archive (origin)..."
        pixi run --frozen -e "${env_name}" dvc pull --force -r origin "$@" || \
        pixi run --frozen -e "${env_name}" dvc pull --force "$@" || \
        echo ">>> Warning: Remote artifacts not found in DVC; falling back to existing cache."
    fi
}

stage_serving_checkpoints() {
    if [ -d "/var/lib/taxon-vision/checkpoints" ] && [ -f "models/checkpoints/head.pt" ]; then
        echo ">>> Atomically staging checkpoints and co-versioned configs to host volume /var/lib/taxon-vision/checkpoints/..."
        for ckpt in models/checkpoints/* config/taxa_catalog.yaml config/conformal_config.yaml; do
            atomic_stage_file "${ckpt}" "/var/lib/taxon-vision/checkpoints"
        done
    fi

    if [ -d "/var/lib/taxon-vision/reference" ] && [ -f "data/embeddings/cal_embeddings.pt" ]; then
        echo ">>> Atomically staging cal_embeddings.pt to host volume /var/lib/taxon-vision/reference/..."
        atomic_stage_file "data/embeddings/cal_embeddings.pt" "/var/lib/taxon-vision/reference"
    fi
}

main() {
    export PATH="/root/.pixi/bin:${PATH}"
    trap 'cleanup_dvc_local "${PIXI_ENV}"' EXIT

    if [ -f "/var/lib/taxon-vision/checkpoints/head.pt" ] && \
       [ -f "/var/lib/taxon-vision/checkpoints/model_int8.onnx" ] && \
       [ -f "/var/lib/taxon-vision/reference/cal_embeddings.pt" ]; then
        echo ">>> Host volume /var/lib/taxon-vision/checkpoints/ already holds active checkpoints; preserving host checkpoints."
        return 0
    fi

    echo ">>> Hydrating missing model checkpoints and reference embeddings from DVC remotes..."
    configure_seaweedfs_remote "${PIXI_ENV}"
    pull_dvc_artifacts_with_fallback "${PIXI_ENV}" \
        models/checkpoints/head.pt \
        models/checkpoints/model_int8.onnx \
        data/embeddings/cal_embeddings.pt
    stage_serving_checkpoints
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
