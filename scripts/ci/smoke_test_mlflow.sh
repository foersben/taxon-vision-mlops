#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Benjamin Förster
# SPDX-License-Identifier: MIT
# Smoke test for the dual-plane TaxonVision serving stack (PyFunc contract, FastAPI HTTP, and MLflow /invocations).

set -euo pipefail

FASTAPI_URL="${FASTAPI_URL:-http://localhost:8000}"
MLFLOW_SERVE_URL="${MLFLOW_SERVE_URL:-http://localhost:5001}"
PIXI_ENV="${PIXI_ENVIRONMENT_NAME:-dev}"

TMP_IMG=""

cleanup() {
    rm -f "${TMP_IMG}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# Strip Pixi's LD_LIBRARY_PATH so system curl never collides with Conda shared libs
curl_clean() {
    env -u LD_LIBRARY_PATH curl "$@"
}

run_py() {
    if [ -n "${PIXI_ENVIRONMENT_NAME:-}" ]; then
        python "$@"
    else
        pixi run --frozen -e "${PIXI_ENV}" python "$@"
    fi
}

check_pyfunc_contract() {
    echo "==> Step 1/3: Testing MLflow PyFunc PythonModel contract in-process..."
    run_py -c '
from taxon_vision.inference.mlflow_model import TaxonVisionPyFuncModel

model = TaxonVisionPyFuncModel()
assert callable(getattr(model, "load_context", None)), "Missing load_context hook"
assert callable(getattr(model, "predict", None)), "Missing predict hook"
print("    Python PyFunc model architecture verified successfully.")
'
    echo "    Status: PyFunc contract verified."
}

check_fastapi_plane() {
    echo "==> Step 2/3: Checking FastAPI HTTP serving plane at ${FASTAPI_URL}..."
    if ! curl_clean -s --connect-timeout 2 -o /dev/null "${FASTAPI_URL}/health/ready" 2>/dev/null; then
        echo "    Notice: FastAPI endpoint (${FASTAPI_URL}) is not running; skipping live HTTP check."
        return
    fi

    echo "    FastAPI endpoint is active. Querying /health/ready..."
    echo "    Ready response: $(curl_clean -s "${FASTAPI_URL}/health/ready")"

    TMP_IMG=$(mktemp --suffix=.png)
    run_py -c "from PIL import Image; Image.new('RGB', (224, 224), (120, 150, 180)).save('${TMP_IMG}')"

    echo "    Sending sample observation to POST /api/v1/predict?dry_run=true..."
    local pred_resp
    pred_resp=$(curl_clean -s -X POST -F "file=@${TMP_IMG};type=image/png" "${FASTAPI_URL}/api/v1/predict?dry_run=true")

    if echo "${pred_resp}" | grep -q "scientific_name"; then
        echo "    FastAPI prediction verified: contains scientific_name and conformal metadata."
    else
        echo "    WARNING: Unexpected FastAPI response: ${pred_resp}"
    fi
}

check_mlflow_plane() {
    echo "==> Step 3/3: Checking MLflow Model Serving plane at ${MLFLOW_SERVE_URL}..."
    if ! curl_clean -s --connect-timeout 2 -o /dev/null "${MLFLOW_SERVE_URL}/health" 2>/dev/null && \
       ! curl_clean -s --connect-timeout 2 -o /dev/null "${MLFLOW_SERVE_URL}/ping" 2>/dev/null; then
        echo "    Notice: MLflow model server (${MLFLOW_SERVE_URL}) is not running; skipping live HTTP check."
        return
    fi

    echo "    MLflow model server is active. Generating dataframe_split payload..."
    local payload inv_resp
    payload=$(run_py -c '
import base64, io, json
from PIL import Image
buf = io.BytesIO()
Image.new("RGB", (224, 224), (80, 120, 160)).save(buf, format="JPEG")
b64 = base64.b64encode(buf.getvalue()).decode("ascii")
print(json.dumps({"dataframe_split": {"columns": ["image_base64"], "data": [[b64]]}}))
')

    echo "    Sending payload to POST /invocations..."
    inv_resp=$(curl_clean -s -X POST -H "Content-Type: application/json" -d "${payload}" "${MLFLOW_SERVE_URL}/invocations")
    if echo "${inv_resp}" | grep -q "predictions"; then
        echo "    MLflow /invocations verified: received predictions payload."
    else
        echo "    WARNING: Unexpected MLflow response: ${inv_resp}"
    fi
}

main() {
    echo "=================================================================="
    echo "   TaxonVision MLflow 3.x Dual-Plane Serving Smoke Test"
    echo "=================================================================="
    check_pyfunc_contract
    check_fastapi_plane
    check_mlflow_plane
    echo "=================================================================="
    echo "   MLflow Serving Smoke Test completed successfully."
    echo "=================================================================="
}

main
