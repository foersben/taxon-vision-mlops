#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Benjamin Förster
# SPDX-License-Identifier: MIT
set -euo pipefail

export PATH="/root/.pixi/bin:$PATH"
export PYTHONUNBUFFERED=1
export TAXON_OPTUNA_STORAGE_URI="${TAXON_OPTUNA_STORAGE_URI:-sqlite:////workspace/optuna/optuna_studies.db}"

which git >/dev/null 2>&1 || pixi global install git
which kubectl >/dev/null 2>&1 || pixi global install kubernetes-client
git config --global --add safe.directory "*"

echo ">>> Setting up GPU training pipeline..."
pixi install --frozen -e dev

# Configure local SeaweedFS credentials for DVC cache
pixi run --frozen -e dev dvc config --local remote.local-seaweedfs.access_key_id taxonadmin
pixi run --frozen -e dev dvc config --local remote.local-seaweedfs.secret_access_key taxonadmin

# Prime cached embeddings and manifests from host volume tier if available
mkdir -p data/embeddings data/manifests
if [ ! -f "data/embeddings/train_embeddings.pt" ]; then
  if [ -f "/var/lib/taxon-vision/embeddings/train_embeddings.pt" ]; then
    echo ">>> Found cached features on host tier; priming data/embeddings/*.pt..."
    cp -f /var/lib/taxon-vision/embeddings/*.pt data/embeddings/ 2>/dev/null || true
  elif kubectl get service seaweedfs -n taxon-vision >/dev/null 2>&1; then
    echo ">>> Attempting feature cache retrieval from in-cluster SeaweedFS (local-seaweedfs)..."
    pixi run --frozen -e dev dvc pull -r local-seaweedfs \
      data/manifests/raw_manifest.parquet \
      data/manifests/dataset_manifest.parquet \
      data/embeddings/train_embeddings.pt \
      data/embeddings/val_embeddings.pt \
      data/embeddings/cal_embeddings.pt \
      data/embeddings/test_embeddings.pt \
      models/checkpoints/head.pt \
      models/checkpoints/model_int8.onnx || echo "Cache pull from local-seaweedfs skipped; will generate via DVC pipeline."
  fi
fi

# Execute unified 7-stage DVC pipeline
echo ">>> Executing unified 7-stage DVC pipeline..."
EXTRA_DVC_OPTS=""
if [ "${TUNE_HYPERPARAMETERS:-false}" = "true" ]; then
  echo ">>> Bayesian optimization enabled: running with training.tune=true..."
  EXTRA_DVC_OPTS="-S training.tune=true"
fi
if [ -n "${EXTRACTOR:-}" ] && [ "${EXTRACTOR}" != "dinov3" ]; then
  EXTRA_DVC_OPTS="${EXTRA_DVC_OPTS} -S model.extractor=${EXTRACTOR}"
fi

pixi run --frozen -e dev dvc repro ${EXTRA_DVC_OPTS}

# Persist generated features to host volume cache
if [ -d "/var/lib/taxon-vision/embeddings" ] && [ -f "data/embeddings/train_embeddings.pt" ]; then
  cp -f data/embeddings/*.pt /var/lib/taxon-vision/embeddings/ 2>/dev/null || true
fi

# Push embedding cache and manifests to local SeaweedFS if reachable
if kubectl get service seaweedfs -n taxon-vision >/dev/null 2>&1; then
  echo ">>> Synchronizing Epistemic Diamond artifacts to in-cluster SeaweedFS (local-seaweedfs)..."
  pixi run --frozen -e dev dvc push -r local-seaweedfs \
    data/manifests/raw_manifest.parquet \
    data/manifests/dataset_manifest.parquet \
    data/embeddings/train_embeddings.pt \
    data/embeddings/val_embeddings.pt \
    data/embeddings/cal_embeddings.pt \
    data/embeddings/test_embeddings.pt \
    models/checkpoints/head.pt \
    models/checkpoints/model_int8.onnx || echo "DVC push to local-seaweedfs skipped."
fi

# Push release checkpoints, INT8 models, and manifests to DagsHub S3 remote (origin)
echo ">>> Pushing release checkpoints and manifests to DagsHub S3 remote (origin)..."
pixi run --frozen -e dev dvc push -r origin \
  data/manifests/raw_manifest.parquet \
  data/manifests/dataset_manifest.parquet \
  models/checkpoints/head.pt \
  models/checkpoints/model_int8.onnx

# Stage and commit updated DVC lockfile and evaluation reports to Git
echo ">>> Committing updated pipeline lockfile and evaluation reports to Git..."
git add dvc.lock config/conformal_config.yaml docs/assets/pareto_results.json reports/*.json
if ! git diff --staged --quiet; then
  git commit -m "chore(pipeline): update dvc.lock and evaluation reports [skip ci]" || echo "Git commit skipped."
  git push origin HEAD:"${BRANCH_NAME:-develop}" || echo "Git push skipped."
else
  echo "No pipeline artifact changes to commit."
fi
