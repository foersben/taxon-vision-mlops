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
if [ "${TUNE_HYPERPARAMETERS:-false}" = "true" ]; then
  echo ">>> Bayesian optimization enabled: updating training.tune=true in params.yaml..."
  sed -i -E "s/^([[:space:]]*tune:[[:space:]]*).*/\1true/" params.yaml
fi
if [ -n "${EXTRACTOR:-}" ] && [ "${EXTRACTOR}" != "dinov3" ]; then
  echo ">>> Updating model.extractor=${EXTRACTOR} in params.yaml..."
  sed -i -E "s/^([[:space:]]*extractor:[[:space:]]*).*/\1\"${EXTRACTOR}\"/" params.yaml
fi

# Check if live ingestion from iNaturalist is requested
if [ "${INGEST_DATA:-false}" = "true" ]; then
  echo ">>> INGEST_DATA is true: Unfreezing ingestion stage and fetching fresh data..."

  # If any command fails during live ingestion, restore dvc.yaml's frozen state
  trap 'echo ">>> Ingestion failed! Restoring dvc.yaml frozen state..."; git checkout -- dvc.yaml 2>/dev/null || true' ERR

  pixi run --frozen -e dev dvc unfreeze ingestion
  pixi run --frozen -e dev dvc repro ingestion
  pixi run --frozen -e dev dvc freeze ingestion

  # Clear the trap once ingestion and re-freezing succeed
  trap - ERR
else
  echo ">>> Using existing cached raw_manifest.parquet (ingestion stage remains frozen)."
fi

pixi run --frozen -e dev dvc repro

# Persist generated features to host volume cache
if [ -d "/var/lib/taxon-vision/embeddings" ] && [ -f "data/embeddings/train_embeddings.pt" ]; then
  cp -f data/embeddings/*.pt /var/lib/taxon-vision/embeddings/ 2>/dev/null || true
fi

# Persist generated checkpoints to host volume cache
if [ -d "/var/lib/taxon-vision/checkpoints" ] && [ -f "models/checkpoints/head.pt" ]; then
  echo ">>> Atomically staging checkpoints to host volume /var/lib/taxon-vision/checkpoints/..."
  for ckpt in models/checkpoints/*; do
    fname=$(basename "$ckpt")
    cp -f "$ckpt" "/var/lib/taxon-vision/checkpoints/.${fname}.tmp"
    chmod a+r "/var/lib/taxon-vision/checkpoints/.${fname}.tmp"
    mv -f "/var/lib/taxon-vision/checkpoints/.${fname}.tmp" "/var/lib/taxon-vision/checkpoints/${fname}"
  done
fi

# 1. Sync to in-cluster SeaweedFS (Tier 1 Edge Cache)
if kubectl get service seaweedfs -n taxon-vision >/dev/null 2>&1; then
  echo ">>> Synchronizing Epistemic Diamond artifacts to SeaweedFS (local-seaweedfs)..."
  pixi run --frozen -e dev dvc push -r local-seaweedfs || echo "SeaweedFS push skipped."
fi

# 2. Push all artifacts to DagsHub S3 (Tier 2 Cloud Archive)
echo ">>> Pushing all Epistemic Diamond artifacts to DagsHub (origin)..."
pixi run --frozen -e dev dvc push -r origin

# Stage and commit updated DVC lockfile and evaluation reports to Git
echo ">>> Committing updated pipeline lockfile and evaluation reports to Git..."
git add params.yaml dvc.lock config/conformal_config.yaml docs/assets/pareto_results.json reports/*.json
if ! git diff --staged --quiet; then
  git commit -m "chore(pipeline): update dvc.lock and evaluation reports [skip ci]" || echo "Git commit skipped."
  git push origin HEAD:"${BRANCH_NAME:-develop}" || echo "Git push skipped."
else
  echo "No pipeline artifact changes to commit."
fi
