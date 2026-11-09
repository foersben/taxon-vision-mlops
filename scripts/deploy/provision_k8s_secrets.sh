#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Benjamin Förster
# SPDX-License-Identifier: MIT
#
# Summary:
#   Provisions in-cluster Kubernetes Secret manifests for TaxonVision-MLOps.
#
# Why:
#   Prevents storing plaintext database passwords and API tokens in Git repositories
#   or CI/CD credential stores. Decouples Kubernetes secret management by sourcing
#   authoritative credentials strictly from a restricted host environment file
#   (/etc/taxon-vision/production.env) and injecting them into the cluster via
#   idempotent client-side dry-run generation.
#
# How:
#   Validates environment file permissions (enforcing 0600/0400 ownership), extracts
#   required production variables, constructs Kubernetes 'Secret' definitions via
#   'kubectl create secret generic --dry-run=client -o yaml', and applies them to the
#   designated namespace without logging secret values.
#
# Arguments:
#   --env-file <path>   Path to environment file (default: /etc/taxon-vision/production.env)
#   --namespace <ns>    Target Kubernetes namespace (default: taxon-vision)
#   --dry-run           Render generated Secret YAML to stdout without applying to K8s
#   --mock              Generate synthetic dummy credentials for verification/testing
#   --help              Display usage synopsis

set -euo pipefail

ENV_FILE="/etc/taxon-vision/production.env"
NAMESPACE="taxon-vision"
DRY_RUN=false
MOCK_MODE=false

print_usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Options:
  --env-file PATH   Path to environment file (default: /etc/taxon-vision/production.env)
  --namespace NS    Target Kubernetes namespace (default: taxon-vision)
  --dry-run         Print generated Secret manifests without executing kubectl apply
  --mock            Use mock fixtures (for CI and dry-run syntax verification)
  --help            Show this help message
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --env-file)
            ENV_FILE="$2"
            shift 2
            ;;
        --namespace)
            NAMESPACE="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --mock)
            MOCK_MODE=true
            shift
            ;;
        --help|-h)
            print_usage
            exit 0
            ;;
        *)
            echo "Error: Unknown argument '$1'" >&2
            print_usage >&2
            exit 1
            ;;
    esac
done

if [[ "${MOCK_MODE}" == "true" ]]; then
    OPTUNA_STORAGE_URI="postgresql+psycopg://optuna:mock_optuna_pass@postgres:5432/optuna"
    TELEMETRY_DATABASE_URL="postgresql+psycopg://telemetry:mock_telemetry_pass@postgres:5432/telemetry"
    INTERNAL_API_TOKEN="mock_internal_token_secret_1234567890"
    RETRAIN_WEBHOOK_SECRET="mock_retrain_webhook_secret_abcdef"
    POSTGRES_PASSWORD="mock_postgres_password"
    AIRFLOW_FERNET_KEY="mock_fernet_key_placeholder_value_0000="
    AIRFLOW_JWT_SECRET="mock_airflow_jwt_secret_key_32bytes"
    JENKINS_API_TOKEN="mock_jenkins_api_token"
else
    if [[ ! -f "${ENV_FILE}" ]]; then
        echo "Error: Environment file not found at '${ENV_FILE}'." >&2
        echo "Hint: In CI/local testing, execute with '--dry-run --mock'." >&2
        exit 1
    fi

    # Verify file permissions on real environment file
    FILE_PERMS=$(stat -c "%a" "${ENV_FILE}" 2>/dev/null || stat -f "%OLp" "${ENV_FILE}" 2>/dev/null || echo "unknown")
    if [[ "${FILE_PERMS}" != "600" && "${FILE_PERMS}" != "400" && "${FILE_PERMS}" != "unknown" ]]; then
        echo "Warning: Insecure permissions (${FILE_PERMS}) on ${ENV_FILE}. Expected 0600 or 0400." >&2
    fi

    # Read key-value pairs safely without arbitrary shell execution
    while IFS='=' read -r key value || [[ -n "$key" ]]; do
        # Ignore comments and empty lines
        [[ "$key" =~ ^[[:space:]]*# ]] && continue
        [[ -z "${key// }" ]] && continue
        # Trim leading/trailing whitespace and strip quotes
        key=$(echo "$key" | xargs)
        value=$(echo "$value" | sed -e 's/^[[:space:]]*["'\'']//' -e 's/["'\''][[:space:]]*$//')
        export "$key=$value"
    done < "${ENV_FILE}"

    OPTUNA_STORAGE_URI="${TAXON_OPTUNA_STORAGE_URI:-${OPTUNA_STORAGE_URI:-}}"
    TELEMETRY_DATABASE_URL="${TAXON_TELEMETRY__DATABASE_URL:-${TELEMETRY_DATABASE_URL:-}}"
    INTERNAL_API_TOKEN="${TAXON_INTERNAL_API_TOKEN:-${INTERNAL_API_TOKEN:-}}"
    RETRAIN_WEBHOOK_SECRET="${TAXON_RETRAIN_WEBHOOK_SECRET:-${RETRAIN_WEBHOOK_SECRET:-}}"
    POSTGRES_PASSWORD="${POSTGRES_PASSWORD:-}"
    AIRFLOW_FERNET_KEY="${AIRFLOW__CORE__FERNET_KEY:-${AIRFLOW_FERNET_KEY:-}}"
    AIRFLOW_JWT_SECRET="${AIRFLOW__API__AUTH_JWT_SECRET:-${AIRFLOW_JWT_SECRET:-}}"
    JENKINS_API_TOKEN="${JENKINS_API_TOKEN:-}"
fi

# Ensure required core variables are non-empty
if [[ -z "${OPTUNA_STORAGE_URI}" || -z "${TELEMETRY_DATABASE_URL}" || -z "${INTERNAL_API_TOKEN}" || -z "${RETRAIN_WEBHOOK_SECRET}" ]]; then
    echo "Error: Missing one or more required production secrets." >&2
    exit 1
fi

KUBECTL_BIN=$(command -v kubectl || echo "")

generate_secrets_yaml() {
    cat <<EOF
apiVersion: v1
kind: Secret
metadata:
  name: taxon-vision-secrets
  namespace: ${NAMESPACE}
  labels:
    app.kubernetes.io/part-of: taxon-vision-mlops
type: Opaque
stringData:
  OPTUNA_STORAGE_URI: "${OPTUNA_STORAGE_URI}"
  TELEMETRY_DATABASE_URL: "${TELEMETRY_DATABASE_URL}"
  INTERNAL_API_TOKEN: "${INTERNAL_API_TOKEN}"
  RETRAIN_WEBHOOK_SECRET: "${RETRAIN_WEBHOOK_SECRET}"
  POSTGRES_PASSWORD: "${POSTGRES_PASSWORD}"
---
apiVersion: v1
kind: Secret
metadata:
  name: airflow-secrets
  namespace: ${NAMESPACE}
  labels:
    app.kubernetes.io/part-of: taxon-vision-mlops
type: Opaque
stringData:
  AIRFLOW__CORE__FERNET_KEY: "${AIRFLOW_FERNET_KEY}"
  AIRFLOW__API_AUTH__JWT_SECRET: "${AIRFLOW_JWT_SECRET}"
  JENKINS_API_TOKEN: "${JENKINS_API_TOKEN}"
EOF
}

if [[ "${DRY_RUN}" == "true" ]]; then
    echo "# Dry-run secret manifest generation for namespace: ${NAMESPACE}"
    generate_secrets_yaml
    exit 0
fi

if [[ -z "${KUBECTL_BIN}" ]]; then
    echo "Error: kubectl binary not found in PATH." >&2
    exit 1
fi

echo "Applying Kubernetes secrets to namespace '${NAMESPACE}'..."
generate_secrets_yaml | "${KUBECTL_BIN}" apply -f -
echo "Successfully provisioned 'taxon-vision-secrets' and 'airflow-secrets'."
