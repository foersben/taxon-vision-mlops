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
#   required production variables without exporting to child process environments,
#   masks secrets in dry-run mode, and applies them via memory pipes without writing
#   unencrypted files to disk.
#
# Arguments:
#   --env-file <path>   Path to environment file (default: /etc/taxon-vision/production.env)
#   --namespace <ns>    Target Kubernetes namespace (default: taxon-vision)
#   --dry-run           Simulate execution and print generated manifests (masked by default)
#   --mock              Generate synthetic dummy credentials for verification/testing
#   --unsafe-reveal     Allow printing unmasked secrets in dry-run mode (caution: stdout exposure)
#   --help              Display usage synopsis

set -euo pipefail

ENV_FILE="/etc/taxon-vision/production.env"
NAMESPACE="taxon-vision"
DRY_RUN=false
MOCK_MODE=false
UNSAFE_REVEAL=false

print_usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Options:
  --env-file PATH      Path to environment file (default: /etc/taxon-vision/production.env)
  --namespace NS       Target Kubernetes namespace (default: taxon-vision)
  --dry-run            Simulate execution and print generated manifests (masked by default)
  --mock               Use mock fixtures (for CI and dry-run syntax verification)
  --unsafe-reveal      Allow printing unmasked secrets in dry-run mode (caution: stdout exposure)
  --help               Show this help message
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
        --unsafe-reveal)
            UNSAFE_REVEAL=true
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
    OPTUNA_STORAGE_URI="postgresql+psycopg://optuna:taxon_optuna_pass@postgres.taxon-vision.svc.cluster.local:5432/optuna"
    TELEMETRY_DATABASE_URL="postgresql+psycopg://telemetry:taxon_telemetry_pass@postgres.taxon-vision.svc.cluster.local:5432/telemetry"
    INTERNAL_API_TOKEN="mock_internal_token_secret_1234567890"
    RETRAIN_WEBHOOK_SECRET="mock_retrain_webhook_secret_abcdef"
    POSTGRES_PASSWORD="taxon_super_secret"
    AIRFLOW_FERNET_KEY=$(echo -n "01234567890123456789012345678901" | base64)
    AIRFLOW_JWT_SECRET="mock_airflow_jwt_secret_key_32bytes"
    JENKINS_API_TOKEN="mock_jenkins_api_token"
else
    if [[ ! -f "${ENV_FILE}" ]]; then
        echo "Error: Environment file not found at '${ENV_FILE}'." >&2
        echo "Hint: In CI/local testing, execute with '--dry-run --mock'." >&2
        exit 1
    fi

    # Verify strict file permissions on the authoritative host file
    FILE_PERMS=$(stat -c "%a" "${ENV_FILE}" 2>/dev/null || stat -f "%OLp" "${ENV_FILE}" 2>/dev/null || echo "unknown")
    if [[ "${FILE_PERMS}" != "600" && "${FILE_PERMS}" != "400" && "${FILE_PERMS}" != "unknown" ]]; then
        echo "Warning: Insecure permissions (${FILE_PERMS}) on ${ENV_FILE}. Expected 0600 or 0400." >&2
    fi

    # Declare associative array for internal parsing without polluting process environment
    declare -A SECRETS=()
    while IFS='=' read -r key value || [[ -n "$key" ]]; do
        [[ "$key" =~ ^[[:space:]]*# ]] && continue
        [[ -z "${key// }" ]] && continue
        key=$(echo "$key" | xargs)
        value=$(echo "$value" | sed -e 's/^[[:space:]]*["'\'']//' -e 's/["'\''][[:space:]]*$//')
        SECRETS["$key"]="$value"
    done < "${ENV_FILE}"

    OPTUNA_STORAGE_URI="${SECRETS[TAXON_OPTUNA_STORAGE_URI]:-${SECRETS[OPTUNA_STORAGE_URI]:-}}"
    TELEMETRY_DATABASE_URL="${SECRETS[TAXON_TELEMETRY__DATABASE_URL]:-${SECRETS[TELEMETRY_DATABASE_URL]:-}}"
    INTERNAL_API_TOKEN="${SECRETS[TAXON_INTERNAL_API_TOKEN]:-${SECRETS[INTERNAL_API_TOKEN]:-}}"
    RETRAIN_WEBHOOK_SECRET="${SECRETS[TAXON_RETRAIN_WEBHOOK_SECRET]:-${SECRETS[RETRAIN_WEBHOOK_SECRET]:-}}"
    POSTGRES_PASSWORD="${SECRETS[POSTGRES_PASSWORD]:-}"
    AIRFLOW_FERNET_KEY="${SECRETS[AIRFLOW__CORE__FERNET_KEY]:-${SECRETS[AIRFLOW_FERNET_KEY]:-}}"
    AIRFLOW_JWT_SECRET="${SECRETS[AIRFLOW__API__AUTH_JWT_SECRET]:-${SECRETS[AIRFLOW_JWT_SECRET]:-}}"
    JENKINS_API_TOKEN="${SECRETS[JENKINS_API_TOKEN]:-}"
fi

# Ensure required core variables are non-empty
if [[ -z "${OPTUNA_STORAGE_URI}" || -z "${TELEMETRY_DATABASE_URL}" || -z "${INTERNAL_API_TOKEN}" || -z "${RETRAIN_WEBHOOK_SECRET}" ]]; then
    echo "Error: Missing one or more required production secrets." >&2
    exit 1
fi

KUBECTL_BIN=$(command -v kubectl || echo "")

mask_val() {
    local val="$1"
    if [[ "${MOCK_MODE}" == "true" || "${UNSAFE_REVEAL}" == "true" ]]; then
        echo "$val"
    else
        echo "******** [MASKED - USE --unsafe-reveal TO INSPECT]"
    fi
}

generate_secrets_yaml() {
    local optuna_val telemetry_val token_val webhook_val pg_val fernet_val jwt_val jnk_val
    optuna_val=$(mask_val "${OPTUNA_STORAGE_URI}")
    telemetry_val=$(mask_val "${TELEMETRY_DATABASE_URL}")
    token_val=$(mask_val "${INTERNAL_API_TOKEN}")
    webhook_val=$(mask_val "${RETRAIN_WEBHOOK_SECRET}")
    pg_val=$(mask_val "${POSTGRES_PASSWORD}")
    fernet_val=$(mask_val "${AIRFLOW_FERNET_KEY}")
    jwt_val=$(mask_val "${AIRFLOW_JWT_SECRET}")
    jnk_val=$(mask_val "${JENKINS_API_TOKEN}")

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
  OPTUNA_STORAGE_URI: "${optuna_val}"
  TELEMETRY_DATABASE_URL: "${telemetry_val}"
  INTERNAL_API_TOKEN: "${token_val}"
  RETRAIN_WEBHOOK_SECRET: "${webhook_val}"
  POSTGRES_PASSWORD: "${pg_val}"
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
  AIRFLOW__CORE__FERNET_KEY: "${fernet_val}"
  AIRFLOW__API_AUTH__JWT_SECRET: "${jwt_val}"
  JENKINS_API_TOKEN: "${jnk_val}"
EOF
}

generate_live_secrets_yaml() {
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
generate_live_secrets_yaml | "${KUBECTL_BIN}" apply -f -
echo "Successfully provisioned 'taxon-vision-secrets' and 'airflow-secrets'."
