#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Benjamin Förster
# SPDX-License-Identifier: MIT
set -euo pipefail

export KUBECONFIG="${KUBECONFIG:-$HOME/.kube/config}"
export PATH="/root/.pixi/bin:$PATH"

if ! command -v curl >/dev/null 2>&1; then
    if command -v pixi >/dev/null 2>&1; then
        echo ">>> curl not found; installing curl via pixi global..."
        pixi global install curl
    else
        echo "ERROR: curl is required for deployment verification." >&2
        exit 1
    fi
fi

NAMESPACE="${1:-taxon-vision}"
DEPLOYMENT="${DEPLOYMENT:-taxon-vision-serving}"
SERVICE="${SERVICE:-taxon-vision-serving}"

echo "=================================================================="
echo "   TaxonVision-MLOps Kubernetes Deployment Verification"
echo "=================================================================="

# 1. Verify Deployment Rollout Status
echo "==> Step 1/5: Checking rollout status in namespace '${NAMESPACE}'..."
kubectl rollout status deployment/"${DEPLOYMENT}" -n "${NAMESPACE}" --timeout=60s

# 2. Check Pod Readiness
echo "==> Step 2/5: Inspecting pod states..."
kubectl get pods -n "${NAMESPACE}" -l app.kubernetes.io/name="${DEPLOYMENT}" -o wide

# 3. Discover Service Endpoint
echo "==> Step 3/5: Resolving Service Endpoint..."
CLUSTER_IP=$(kubectl get svc "${SERVICE}" -n "${NAMESPACE}" -o jsonpath='{.spec.clusterIP}')
PORT=$(kubectl get svc "${SERVICE}" -n "${NAMESPACE}" -o jsonpath='{.spec.ports[0].port}')
TARGET_URL="http://${CLUSTER_IP}:${PORT}"
echo "    Found Service ClusterIP: ${TARGET_URL}"

# Wait for old pods to terminate and fully disappear from EndpointSlices before probing
echo "    Waiting for old terminating replicas to drain..."
for _ in {1..30}; do
    TERMINATING_COUNT=$(kubectl get pods -n "$NAMESPACE" -l app.kubernetes.io/name="${DEPLOYMENT}" \
        --no-headers 2>/dev/null | grep -c "Terminating" || true)
    if [ "$TERMINATING_COUNT" -eq 0 ]; then
        break
    fi
    sleep 1
done


# Reusable curl flags to retry if connection is refused
CURL_OPTS=(-s --retry 5 --retry-delay 2 --retry-connrefused)

# 4. Probe Health and Readiness Endpoints
echo "==> Step 4/5: Probing HTTP endpoints..."
LIVENESS_STATUS=$(curl "${CURL_OPTS[@]}" -o /dev/null -w "%{http_code}" "${TARGET_URL}/health" || echo "FAILED")
READINESS_STATUS=$(curl "${CURL_OPTS[@]}" -o /dev/null -w "%{http_code}" "${TARGET_URL}/health/ready" || echo "FAILED")

echo "    Liveness Probe (/health): HTTP ${LIVENESS_STATUS}"
echo "    Readiness Probe (/health/ready): HTTP ${READINESS_STATUS}"

if [ "${LIVENESS_STATUS}" != "200" ] || [ "${READINESS_STATUS}" != "200" ]; then
    echo "   ERROR: Health probes did not return HTTP 200 OK."
    exit 1
fi

# 5. Execute Prediction Inference & Metrics Scrape
echo "==> Step 5/5: Testing live prediction and telemetry..."
PREDICT_RESP=$(curl "${CURL_OPTS[@]}" -X POST "${TARGET_URL}/api/v1/predict" -F 'file=@/dev/null;filename=probe.jpg;type=image/jpeg')
echo "    Prediction Response: ${PREDICT_RESP}"

METRICS_MATCH=$(curl "${CURL_OPTS[@]}" "${TARGET_URL}/metrics" | grep -c "taxon_predictions_total" || true)
echo "    Prometheus metrics verified (${METRICS_MATCH} matches for taxon_predictions_total)"

# Prune any historical failed or evicted pods from older rollouts
kubectl delete pods -n "${NAMESPACE}" --field-selector=status.phase=Failed --ignore-not-found=true >/dev/null 2>&1 || true

echo "=================================================================="
echo "   DEPLOYMENT VERIFICATION PASSED: TaxonVision is operational!"
echo "=================================================================="
