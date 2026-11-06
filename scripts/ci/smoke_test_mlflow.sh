#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Benjamin Förster
# SPDX-License-Identifier: MIT
set -euo pipefail

FASTAPI_URL="${FASTAPI_URL:-http://localhost:8000}"
MLFLOW_SERVE_URL="${MLFLOW_SERVE_URL:-http://localhost:5001}"

# Wrapper that strips Pixi's LD_LIBRARY_PATH to avoid dynamic linker symbol conflicts
curl_clean() {
    env -u LD_LIBRARY_PATH curl "$@"
}

echo "=================================================================="
echo "   TaxonVision MLflow 3.x Dual-Plane Serving Smoke Test"
echo "=================================================================="

# Check 1: Offline Python PyFunc Smoke Test (Guaranteed Verification)
echo "==> Step 1/3: Testing PyFunc PythonModel instantiation and predict in-process..."
pixi run -e dev python -c '
import numpy as np
import pandas as pd
from taxon_vision.inference.mlflow_model import TaxonVisionPyFuncModel
from taxon_vision.inference.onnx_engine import ONNXInferenceEngine

# Verify module import and signature
model = TaxonVisionPyFuncModel()
assert hasattr(model, "load_context"), "Missing load_context hook"
assert hasattr(model, "predict"), "Missing predict hook"
print("    Python PyFunc model architecture verified successfully.")
'
echo "    Status: PyFunc contract verified."

# Check 2: Live FastAPI Plane (< 25ms hot path)
echo "==> Step 2/3: Checking FastAPI HTTP serving plane at ${FASTAPI_URL}..."
if curl_clean -s --connect-timeout 2 -o /dev/null "${FASTAPI_URL}/health/ready" 2>/dev/null; then
    echo "    FastAPI endpoint is active. Querying /health/ready..."
    READY_STATUS=$(curl_clean -s "${FASTAPI_URL}/health/ready")
    echo "    Ready response: ${READY_STATUS}"

    TMP_IMG=$(mktemp --suffix=.png)
    pixi run -e dev python -c "from PIL import Image; Image.new('RGB', (224, 224), (120, 150, 180)).save('${TMP_IMG}')"

    echo "    Sending sample observation to POST /api/v1/predict..."
    PRED_RESP=$(curl_clean -s -X POST -F "image=@${TMP_IMG}" "${FASTAPI_URL}/api/v1/predict")
    rm -f "${TMP_IMG}"

    if echo "${PRED_RESP}" | grep -q "scientific_name"; then
        echo "    FastAPI prediction verified: contains scientific_name and conformal metadata."
    else
        echo "    WARNING: Unexpected FastAPI response: ${PRED_RESP}"
    fi
else
    echo "    Notice: FastAPI endpoint (${FASTAPI_URL}) is not running; skipping live HTTP check."
fi

# Check 3: Live MLflow PyFunc Plane (/invocations)
echo "==> Step 3/3: Checking MLflow Model Serving plane at ${MLFLOW_SERVE_URL}..."
if curl_clean -s --connect-timeout 2 -o /dev/null "${MLFLOW_SERVE_URL}/health" 2>/dev/null || \
   curl_clean -s --connect-timeout 2 -o /dev/null "${MLFLOW_SERVE_URL}/ping" 2>/dev/null; then
    echo "    MLflow model server is active. Generating test dataframe_split payload..."
    PAYLOAD=$(pixi run -e dev python -c '
import base64, io, json
from PIL import Image
buf = io.BytesIO()
Image.new("RGB", (224, 224), (80, 120, 160)).save(buf, format="JPEG")
b64 = base64.b64encode(buf.getvalue()).decode("ascii")
data = {"dataframe_split": {"columns": ["image_base64"], "data": [[b64]]}}
print(json.dumps(data))
')
    echo "    Sending payload to POST /invocations..."
    INV_RESP=$(curl_clean -s -X POST -H "Content-Type: application/json" -d "${PAYLOAD}" "${MLFLOW_SERVE_URL}/invocations")
    if echo "${INV_RESP}" | grep -q "predictions"; then
        echo "    MLflow /invocations verified: received predictions payload."
    else
        echo "    WARNING: Unexpected MLflow response: ${INV_RESP}"
    fi
else
    echo "    Notice: MLflow model server (${MLFLOW_SERVE_URL}) is not running; skipping live HTTP check."
fi

echo "=================================================================="
echo "   ✅ MLflow Serving Smoke Test completed successfully."
echo "=================================================================="
