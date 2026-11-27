#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Benjamin Förster
# SPDX-License-Identifier: MIT
# Post-deploy verification for the TaxonVision serving Deployment in Kubernetes.

set -euo pipefail

export KUBECONFIG="${KUBECONFIG:-$HOME/.kube/config}"
export PATH="/root/.pixi/bin:$PATH"

NAMESPACE="${1:-taxon-vision}"
DEPLOYMENT="${DEPLOYMENT:-taxon-vision-serving}"
SERVICE="${SERVICE:-taxon-vision-serving}"

# Minimal 1x1 RGB PNG (69 bytes, base64-encoded) for side-effect-free ONNX smoke probes
readonly PROBE_PNG_BASE64="iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGNgaPgPAAIDAYAkYfWXAAAAAElFTkSuQmCC"
readonly CURL_OPTS=(-s --retry 5 --retry-delay 2 --retry-connrefused)

PORT_FORWARD_PID=""
PROBE_FILE=""
PREDICT_BODY_FILE=""
TARGET_URL=""

cleanup() {
    rm -f "${PROBE_FILE}" "${PREDICT_BODY_FILE}" 2>/dev/null || true
    if [ -n "${PORT_FORWARD_PID}" ]; then
        echo "    Stopping temporary kubectl port-forward (PID ${PORT_FORWARD_PID})..."
        kill "${PORT_FORWARD_PID}" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

# Strip Pixi's LD_LIBRARY_PATH so system curl never collides with Conda OpenSSL/zlib libs
curl_clean() {
    env -u LD_LIBRARY_PATH curl "$@"
}

ensure_curl() {
    if command -v curl >/dev/null 2>&1; then
        return
    fi
    if command -v pixi >/dev/null 2>&1; then
        echo ">>> curl not found; installing curl via pixi global..."
        pixi global install curl
        return
    fi
    echo "ERROR: curl is required for deployment verification." >&2
    exit 1
}

is_url_reachable() {
    curl_clean -s --connect-timeout 1 -o /dev/null "$1" 2>/dev/null
}

resolve_target_url() {
    local cluster_ip port node_port node_ip lan_url local_port=18000
    cluster_ip=$(kubectl get svc "${SERVICE}" -n "${NAMESPACE}" -o jsonpath='{.spec.clusterIP}')
    port=$(kubectl get svc "${SERVICE}" -n "${NAMESPACE}" -o jsonpath='{.spec.ports[0].port}')

    # Tier 1: In-cluster routing (e.g. from a Jenkins pod inside K3s)
    if is_url_reachable "http://${cluster_ip}:${port}/health"; then
        TARGET_URL="http://${cluster_ip}:${port}"
        echo "    Route: ClusterIP (${TARGET_URL})"
        return
    fi

    # Tier 2: LAN NodePort routing (e.g. from a workstation on the same LAN)
    node_port=$(kubectl get svc "${SERVICE}" -n "${NAMESPACE}" -o jsonpath='{.spec.ports[0].nodePort}' 2>/dev/null || true)
    node_ip=$(kubectl get nodes -o jsonpath='{.items[0].status.addresses[?(@.type=="InternalIP")].address}' 2>/dev/null || true)
    if [ -n "${node_port}" ] && [ "${node_port}" != "<no value>" ] && [ -n "${node_ip}" ]; then
        lan_url="http://${node_ip}:${node_port}"
        if is_url_reachable "${lan_url}/health"; then
            TARGET_URL="${lan_url}"
            echo "    Route: LAN NodePort (${TARGET_URL})"
            return
        fi
    fi

    # Tier 3: Ephemeral kubectl port-forward tunnel (cleaned up on exit)
    echo "    Starting temporary kubectl port-forward on localhost:${local_port}..."
    kubectl port-forward -n "${NAMESPACE}" "svc/${SERVICE}" "${local_port}:${port}" >/dev/null 2>&1 &
    PORT_FORWARD_PID=$!
    sleep 2
    TARGET_URL="http://127.0.0.1:${local_port}"
    echo "    Route: Ephemeral port-forward (${TARGET_URL})"
}

assert_http_200() {
    local label="$1"
    local path="$2"
    local status_code
    status_code=$(curl_clean "${CURL_OPTS[@]}" -o /dev/null -w "%{http_code}" "${TARGET_URL}${path}" || echo "000")
    echo "    ${label} (${path}): HTTP ${status_code}"
    if [ "${status_code}" != "200" ]; then
        echo "ERROR: ${label} did not return HTTP 200 OK." >&2
        exit 1
    fi
}

run_inference_smoke_probe() {
    local predict_status
    PROBE_FILE=$(mktemp /tmp/taxon-probe-XXXXXX.png)
    PREDICT_BODY_FILE=$(mktemp /tmp/taxon-predict-XXXXXX.json)
    printf '%s' "${PROBE_PNG_BASE64}" | base64 -d > "${PROBE_FILE}"

    predict_status=$(curl_clean "${CURL_OPTS[@]}" \
        -o "${PREDICT_BODY_FILE}" \
        -w "%{http_code}" \
        -X POST "${TARGET_URL}/api/v1/predict?dry_run=true" \
        -F "file=@${PROBE_FILE};filename=probe.png;type=image/png" || echo "000")

    echo "    Inference Probe (/api/v1/predict?dry_run=true): HTTP ${predict_status}"
    echo "    Prediction Response: $(cat "${PREDICT_BODY_FILE}")"
    if [ "${predict_status}" != "200" ]; then
        echo "ERROR: Live prediction smoke probe failed (expected HTTP 200, got ${predict_status})." >&2
        exit 1
    fi
}

assert_metrics_exposed() {
    local metrics_status metrics_matches
    metrics_status=$(curl_clean "${CURL_OPTS[@]}" -o /dev/null -w "%{http_code}" "${TARGET_URL}/metrics" || echo "000")
    metrics_matches=$(curl_clean "${CURL_OPTS[@]}" "${TARGET_URL}/metrics" | grep -c "taxon_predictions_total" || true)
    echo "    Prometheus /metrics: HTTP ${metrics_status} (${metrics_matches} references to taxon_predictions_total)"

    if [ "${metrics_status}" != "200" ] || [ "${metrics_matches}" -lt 1 ]; then
        echo "ERROR: Prometheus /metrics endpoint did not expose taxon_predictions_total with HTTP 200." >&2
        exit 1
    fi
}

main() {
    ensure_curl

    echo "=================================================================="
    echo "   TaxonVision-MLOps Kubernetes Deployment Verification"
    echo "=================================================================="

    echo "==> Step 1/5: Checking rollout status in namespace '${NAMESPACE}'..."
    kubectl rollout status deployment/"${DEPLOYMENT}" -n "${NAMESPACE}" --timeout=60s

    echo "==> Step 2/5: Inspecting pod states..."
    kubectl get pods -n "${NAMESPACE}" -l app.kubernetes.io/name="${DEPLOYMENT}" -o wide

    echo "==> Step 3/5: Resolving Service Endpoint..."
    resolve_target_url

    echo "==> Step 4/5: Probing HTTP health endpoints..."
    assert_http_200 "Liveness Probe" "/health"
    assert_http_200 "Readiness Probe" "/health/ready"

    echo "==> Step 5/5: Testing live ONNX inference and Prometheus telemetry..."
    run_inference_smoke_probe
    assert_metrics_exposed

    # Prune historical failed/evicted pods from older rollouts
    kubectl delete pods -n "${NAMESPACE}" --field-selector=status.phase=Failed --ignore-not-found=true >/dev/null 2>&1 || true

    echo "=================================================================="
    echo "   DEPLOYMENT VERIFICATION PASSED: TaxonVision is operational!"
    echo "=================================================================="
}

main
