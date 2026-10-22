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

# 1. Prime cached features, manifests, and checkpoints from host volume tier if available
mkdir -p data/embeddings data/manifests models/checkpoints

if [ -f "/var/lib/taxon-vision/embeddings/train_embeddings.pt" ]; then
  echo ">>> Found cached features on host tier; priming data/embeddings/*.pt..."
  cp -f /var/lib/taxon-vision/embeddings/*.pt data/embeddings/ 2>/dev/null || true
fi

if [ -f "/var/lib/taxon-vision/manifests/raw_manifest.parquet" ]; then
  echo ">>> Found cached manifests on host tier; priming data/manifests/*.parquet..."
  cp -f /var/lib/taxon-vision/manifests/*.parquet data/manifests/ 2>/dev/null || true
fi

if [ -f "/var/lib/taxon-vision/checkpoints/head.pt" ]; then
  echo ">>> Found cached checkpoints on host tier; priming models/checkpoints/*..."
  cp -f /var/lib/taxon-vision/checkpoints/* models/checkpoints/ 2>/dev/null || true
fi

# 2. Check for missing Epistemic Diamond artifacts and hydrate from DVC remotes
# Priority: Tier 1 (local-seaweedfs edge cache) -> Tier 2 (DagsHub origin cloud archive)
MISSING_ARTIFACTS=()
for artifact in \
  "data/manifests/raw_manifest.parquet" \
  "data/manifests/dataset_manifest.parquet" \
  "data/embeddings/train_embeddings.pt" \
  "data/embeddings/val_embeddings.pt" \
  "data/embeddings/cal_embeddings.pt" \
  "data/embeddings/test_embeddings.pt" \
  "models/checkpoints/head.pt" \
  "models/checkpoints/model_int8.onnx"; do
  if [ ! -f "$artifact" ]; then
    MISSING_ARTIFACTS+=("$artifact")
  fi
done

if [ ${#MISSING_ARTIFACTS[@]} -gt 0 ]; then
  echo ">>> Missing artifacts in workspace: ${MISSING_ARTIFACTS[*]}"
  PULLED=false
  if kubectl get service seaweedfs -n taxon-vision >/dev/null 2>&1; then
    echo ">>> Attempting hydration from in-cluster SeaweedFS (local-seaweedfs)..."
    if pixi run --frozen -e dev dvc pull -r local-seaweedfs "${MISSING_ARTIFACTS[@]}"; then
      PULLED=true
      echo ">>> Successfully hydrated missing artifacts from local-seaweedfs."
    fi
  fi

  if [ "$PULLED" != "true" ]; then
    echo ">>> Attempting hydration from DagsHub cloud archive (origin)..."
    pixi run --frozen -e dev dvc pull -r origin "${MISSING_ARTIFACTS[@]}" || \
    pixi run --frozen -e dev dvc pull "${MISSING_ARTIFACTS[@]}" || \
    echo ">>> Warning: Could not pull all missing artifacts from DVC remotes."
  fi
fi

# 3. Fail-safe: If raw_manifest.parquet is still missing, force INGEST_DATA=true to avoid frozen stage failure
if [ ! -f "data/manifests/raw_manifest.parquet" ]; then
  echo ">>> raw_manifest.parquet is missing from all caches and remotes; forcing INGEST_DATA=true to unfreeze and regenerate."
  export INGEST_DATA=true
fi

# Execute unified 7-stage DVC pipeline
echo ">>> Executing unified 7-stage DVC pipeline..."
# Safely apply parameter overrides via PyYAML if explicitly requested by Jenkins parameters
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

# Persist generated manifests to host volume cache
if [ -d "/var/lib/taxon-vision/manifests" ] && [ -f "data/manifests/raw_manifest.parquet" ]; then
  cp -f data/manifests/*.parquet /var/lib/taxon-vision/manifests/ 2>/dev/null || true
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
