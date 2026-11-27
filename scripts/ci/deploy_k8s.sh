#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Benjamin Förster
# SPDX-License-Identifier: MIT
# Zero-downtime Kubernetes continuous deployment orchestrator invoked by Jenkinsfile.

set -euo pipefail

export PATH="/root/.pixi/bin:${PATH}"

BRANCH_NAME="${1:?Usage: deploy_k8s.sh <branch_name>}"
GHCR_IMAGE="ghcr.io/foersben/taxon-vision-mlops"

echo ">>> Executing zero-downtime rolling deployment to Kubernetes cluster..."
kubectl version --client
git fetch origin "${BRANCH_NAME}" && git reset --hard "origin/${BRANCH_NAME}" || true

bash scripts/ci/hydrate_and_stage_checkpoints.sh
bash scripts/ci/sync_airflow_configmaps.sh taxon-vision

kubectl apply -k deploy
kubectl rollout restart deployment/airflow-apiserver deployment/airflow-scheduler -n taxon-vision
kubectl rollout status deployment/airflow-apiserver -n taxon-vision --timeout=180s
kubectl rollout status deployment/airflow-scheduler -n taxon-vision --timeout=180s

if [ "${BRANCH_NAME}" = "main" ]; then
    TARGET_ENV="production"
    TARGET_NS="taxon-vision"
else
    TARGET_ENV="staging"
    TARGET_NS="taxon-vision-staging"
fi

echo ">>> Deploying ${TARGET_ENV} overlay to namespace ${TARGET_NS} via Kustomize..."
kubectl apply -k "deploy/k8s/overlays/${TARGET_ENV}"

SHORT_SHA="$(git rev-parse --short=7 HEAD)"
if bash scripts/ci/wait_for_ghcr_image.sh "${SHORT_SHA}"; then
    kubectl set image deployment/taxon-vision-serving "api-service=${GHCR_IMAGE}:sha-${SHORT_SHA}" -n "${TARGET_NS}"
else
    echo ">>> Using branch overlay image tag and restarting deployment to load updated host-mounted artefacts..."
    kubectl rollout restart deployment/taxon-vision-serving -n "${TARGET_NS}"
fi

kubectl rollout status deployment/taxon-vision-serving -n "${TARGET_NS}" --timeout=600s
bash scripts/ci/verify_deployment.sh "${TARGET_NS}"
echo ">>> Deployment successfully verified for ${TARGET_ENV}."
