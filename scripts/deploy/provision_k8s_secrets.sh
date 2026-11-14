#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Benjamin Förster
# SPDX-License-Identifier: MIT
#
# Summary:
#   Provisions encrypted-at-rest in-cluster Kubernetes Secret manifests for TaxonVision-MLOps.
#
# Why:
#   Prevents storing plaintext database passwords, S3 access keys, or API tokens in Git repositories
#   or CI/CD configuration. Decouples Kubernetes secret provisioning cleanly as a temporary workflow
#   prior to full GitOps migration to FluxCD + SOPS/age.
#
# How:
#   Validates file permissions (0600/0400) on the authoritative host env file (typically
#   /etc/taxon-vision/production.env), derives per-driver SQLAlchemy/psycopg URIs and SeaweedFS
#   s3.json payloads in memory, enforces fail-fast checks against mock or placeholder values during
#   live application, masks outputs in dry-run mode, and applies infra + app Secrets cleanly across
#   taxon-vision and taxon-vision-staging via memory pipes without temporary plaintext files.

set -euo pipefail

ENV_FILE="/etc/taxon-vision/production.env"
TARGET_NAMESPACE="all"
DRY_RUN=false
MOCK_MODE=false
INIT_ENV=false
FORCE_OVERWRITE=false
UNSAFE_REVEAL=false

print_usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS] [ENV_FILE]

Options:
  --env-file PATH      Path to environment file (default: /etc/taxon-vision/production.env)
  --namespace NS       Target namespace: all | taxon-vision | taxon-vision-staging (default: all)
  --init-env           Generate high-entropy credentials into ENV_FILE (chmod 0600) and exit
  --force              Allow overwriting an existing ENV_FILE when combined with --init-env
  --dry-run            Simulate execution and print generated manifests (masked by default)
  --mock               Generate ephemeral synthetic tokens for CI dry-run verification (requires --dry-run)
  --unsafe-reveal      Allow printing unmasked secrets in dry-run mode (caution: stdout exposure)
  --help               Show this help message
EOF
}

POSITIONAL_ENV=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --env-file)
            ENV_FILE="$2"
            shift 2
            ;;
        --namespace)
            TARGET_NAMESPACE="$2"
            shift 2
            ;;
        --init-env)
            INIT_ENV=true
            shift
            ;;
        --force)
            FORCE_OVERWRITE=true
            shift
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
        -*)
            echo "Error: Unknown argument '$1'" >&2
            print_usage >&2
            exit 1
            ;;
        *)
            POSITIONAL_ENV="$1"
            shift
            ;;
    esac
done

if [[ -n "${POSITIONAL_ENV}" ]]; then
    ENV_FILE="${POSITIONAL_ENV}"
fi

gen_hex() {
    local nbytes="${1:-24}"
    if command -v openssl >/dev/null 2>&1; then
        openssl rand -hex "${nbytes}"
    else
        python3 -c "import secrets; print(secrets.token_hex(${nbytes}))"
    fi
}

gen_fernet_key() {
    python3 -c "import base64, os; print(base64.urlsafe_b64encode(os.urandom(32)).decode('ascii'))"
}

lookup_keepass_optional() {
    local attr_key="$1"
    local attr_val="$2"
    if command -v secret-tool >/dev/null 2>&1; then
        secret-tool lookup "${attr_key}" "${attr_val}" 2>/dev/null || true
    fi
}

if [[ "${INIT_ENV}" == "true" ]]; then
    if [[ -f "${ENV_FILE}" && "${FORCE_OVERWRITE}" != "true" ]]; then
        echo "Error: Refusing to overwrite existing '${ENV_FILE}' without --force." >&2
        exit 1
    fi
    PARENT_DIR="$(dirname "${ENV_FILE}")"
    if [[ ! -d "${PARENT_DIR}" ]]; then
        mkdir -p "${PARENT_DIR}"
        chmod 700 "${PARENT_DIR}"
    fi

    EXISTING_JENKINS=$(lookup_keepass_optional "Title" "Jenkins")
    if [[ -z "${EXISTING_JENKINS}" ]]; then
        EXISTING_JENKINS=$(lookup_keepass_optional "service" "jenkins")
    fi
    EXISTING_DAGSHUB=$(lookup_keepass_optional "service" "dagshub")
    if [[ -z "${EXISTING_DAGSHUB}" ]]; then
        EXISTING_DAGSHUB=$(lookup_keepass_optional "Title" "DagsHub")
    fi

    (
        umask 077
        cat > "${ENV_FILE}" <<EOF
# TaxonVision-MLOps Out-of-Band Cluster Credentials (temporary until FluxCD + SOPS)
# Permissions MUST remain 0600 or 0400. Never commit this file to Git.

# PostgreSQL Superuser & Application Role Passwords (URL-safe hex avoids DSN escaping hazards)
POSTGRES_PASSWORD=$(gen_hex 24)
TELEMETRY_DB_PASSWORD=$(gen_hex 24)
OPTUNA_DB_PASSWORD=$(gen_hex 24)
AIRFLOW_DB_PASSWORD=$(gen_hex 24)

# In-Cluster SeaweedFS S3 IAM Credentials (sync identical values to Jenkins 'seaweedfs-s3-credentials')
SEAWEEDFS_ACCESS_KEY_ID=taxon_$(gen_hex 8)
SEAWEEDFS_SECRET_ACCESS_KEY=$(gen_hex 24)

# Grafana Operator Console Credentials
GF_SECURITY_ADMIN_USER=admin
GF_SECURITY_ADMIN_PASSWORD=$(gen_hex 24)

# Apache Airflow 3 Core Secrets & REST API Operator Credentials
AIRFLOW__CORE__FERNET_KEY=$(gen_fernet_key)
AIRFLOW__API_AUTH__JWT_SECRET=$(gen_hex 32)
AIRFLOW_API_USER=admin
AIRFLOW_ADMIN_PASSWORD=$(gen_hex 24)
JENKINS_USER=admin
JENKINS_API_TOKEN=${EXISTING_JENKINS:-REPLACE_WITH_JENKINS_ADMIN_API_TOKEN}
DAGSHUB_TOKEN=${EXISTING_DAGSHUB:-REPLACE_WITH_DAGSHUB_PAT_TOKEN}

# FastAPI Internal Bearer & Grafana Retrain Webhook Secrets
INTERNAL_API_TOKEN=$(gen_hex 32)
RETRAIN_WEBHOOK_SECRET=$(gen_hex 32)
EOF
    )
    chmod 600 "${ENV_FILE}"
    echo "Generated restricted credential file '${ENV_FILE}' (mode 0600)."
    echo "Next steps:"
    echo "  1. Archive these generated credentials in KeePassXC."
    echo "  2. If JENKINS_API_TOKEN or DAGSHUB_TOKEN still show REPLACE_WITH_*, set your real tokens and verify JENKINS_USER."
    echo "  3. Add SEAWEEDFS_ACCESS_KEY_ID / SEAWEEDFS_SECRET_ACCESS_KEY as Jenkins credential 'seaweedfs-s3-credentials'."
    exit 0
fi

if [[ "${MOCK_MODE}" == "true" ]]; then
    if [[ "${DRY_RUN}" != "true" ]]; then
        echo "Error: --mock mode is strictly limited to CI dry-run verification (--dry-run --mock)." >&2
        exit 1
    fi
    POSTGRES_PASSWORD="PLACEHOLDER_MOCK_PG_$(gen_hex 6)"
    TELEMETRY_DB_PASSWORD="PLACEHOLDER_MOCK_TEL_$(gen_hex 6)"
    OPTUNA_DB_PASSWORD="PLACEHOLDER_MOCK_OPT_$(gen_hex 6)"
    AIRFLOW_DB_PASSWORD="PLACEHOLDER_MOCK_AF_$(gen_hex 6)"
    SEAWEEDFS_ACCESS_KEY_ID="PLACEHOLDER_MOCK_S3_KEY"
    SEAWEEDFS_SECRET_ACCESS_KEY="PLACEHOLDER_MOCK_S3_SEC_$(gen_hex 6)"
    GF_SECURITY_ADMIN_USER="admin"
    GF_SECURITY_ADMIN_PASSWORD="PLACEHOLDER_MOCK_GF_$(gen_hex 6)"
    AIRFLOW_FERNET_KEY="$(gen_fernet_key)"
    AIRFLOW_JWT_SECRET="PLACEHOLDER_MOCK_JWT_$(gen_hex 8)"
    AIRFLOW_API_USER="admin"
    AIRFLOW_ADMIN_PASSWORD="PLACEHOLDER_MOCK_AF_ADMIN_$(gen_hex 6)"
    JENKINS_USER="benni"
    JENKINS_API_TOKEN="PLACEHOLDER_MOCK_JENKINS_$(gen_hex 6)"
    DAGSHUB_TOKEN="PLACEHOLDER_MOCK_DAGSHUB_$(gen_hex 6)"
    INTERNAL_API_TOKEN="PLACEHOLDER_MOCK_INTERNAL_$(gen_hex 8)"
    RETRAIN_WEBHOOK_SECRET="PLACEHOLDER_MOCK_WEBHOOK_$(gen_hex 8)"
else
    if [[ ! -f "${ENV_FILE}" ]]; then
        echo "Error: Environment file not found at '${ENV_FILE}'." >&2
        echo "Hint: Generate one with '--init-env --env-file ${ENV_FILE}' or run CI verification via '--dry-run --mock'." >&2
        exit 1
    fi

    FILE_PERMS=$(stat -c "%a" "${ENV_FILE}" 2>/dev/null || stat -f "%OLp" "${ENV_FILE}" 2>/dev/null || echo "unknown")
    if [[ "${FILE_PERMS}" != "600" && "${FILE_PERMS}" != "400" && "${FILE_PERMS}" != "unknown" ]]; then
        echo "Error: Insecure permissions (${FILE_PERMS}) on '${ENV_FILE}'. Require 0600 or 0400." >&2
        exit 1
    fi

    declare -A SECRETS=()
    while IFS= read -r raw_line || [[ -n "${raw_line}" ]]; do
        [[ "${raw_line}" =~ ^[[:space:]]*# ]] && continue
        [[ -z "${raw_line// }" ]] && continue
        key="${raw_line%%=*}"
        value="${raw_line#*=}"
        key="$(echo "${key}" | xargs)"
        value="$(echo "${value}" | sed -e 's/^[[:space:]]*["'\'']//' -e 's/["'\''][[:space:]]*$//')"
        SECRETS["${key}"]="${value}"
    done < "${ENV_FILE}"

    POSTGRES_PASSWORD="${SECRETS[POSTGRES_PASSWORD]:-}"
    TELEMETRY_DB_PASSWORD="${SECRETS[TELEMETRY_DB_PASSWORD]:-}"
    OPTUNA_DB_PASSWORD="${SECRETS[OPTUNA_DB_PASSWORD]:-}"
    AIRFLOW_DB_PASSWORD="${SECRETS[AIRFLOW_DB_PASSWORD]:-}"
    SEAWEEDFS_ACCESS_KEY_ID="${SECRETS[SEAWEEDFS_ACCESS_KEY_ID]:-}"
    SEAWEEDFS_SECRET_ACCESS_KEY="${SECRETS[SEAWEEDFS_SECRET_ACCESS_KEY]:-}"
    GF_SECURITY_ADMIN_USER="${SECRETS[GF_SECURITY_ADMIN_USER]:-admin}"
    GF_SECURITY_ADMIN_PASSWORD="${SECRETS[GF_SECURITY_ADMIN_PASSWORD]:-}"
    AIRFLOW_FERNET_KEY="${SECRETS[AIRFLOW__CORE__FERNET_KEY]:-${SECRETS[AIRFLOW_FERNET_KEY]:-}}"
    AIRFLOW_JWT_SECRET="${SECRETS[AIRFLOW__API_AUTH__JWT_SECRET]:-${SECRETS[AIRFLOW__API__AUTH_JWT_SECRET]:-${SECRETS[AIRFLOW_JWT_SECRET]:-}}}"
    AIRFLOW_API_USER="${SECRETS[AIRFLOW_API_USER]:-admin}"
    AIRFLOW_ADMIN_PASSWORD="${SECRETS[AIRFLOW_ADMIN_PASSWORD]:-}"
    JENKINS_USER="${SECRETS[JENKINS_USER]:-benni}"
    JENKINS_API_TOKEN="${SECRETS[JENKINS_API_TOKEN]:-}"
    DAGSHUB_TOKEN="${SECRETS[DAGSHUB_TOKEN]:-${DAGSHUB_TOKEN:-}}"
    if [[ -z "${DAGSHUB_TOKEN}" ]]; then
        DAGSHUB_TOKEN=$(lookup_keepass_optional "service" "dagshub")
    fi
    if [[ -z "${DAGSHUB_TOKEN}" && "${DRY_RUN}" != "true" ]] && command -v kubectl >/dev/null 2>&1; then
        DAGSHUB_TOKEN=$(kubectl get secret dagshub-credentials -n taxon-vision -o jsonpath='{.data.DAGSHUB_TOKEN}' 2>/dev/null | base64 -d 2>/dev/null || true)
    fi
    INTERNAL_API_TOKEN="${SECRETS[TAXON_INTERNAL_API_TOKEN]:-${SECRETS[INTERNAL_API_TOKEN]:-}}"
    RETRAIN_WEBHOOK_SECRET="${SECRETS[TAXON_RETRAIN_WEBHOOK_SECRET]:-${SECRETS[RETRAIN_WEBHOOK_SECRET]:-}}"

    # Support explicit URI overrides if supplied, otherwise derive deterministically from role passwords
    TELEMETRY_DATABASE_URL="${SECRETS[TAXON_TELEMETRY__DATABASE_URL]:-${SECRETS[TELEMETRY_DATABASE_URL]:-}}"
    OPTUNA_STORAGE_URI="${SECRETS[TAXON_OPTUNA_STORAGE_URI]:-${SECRETS[OPTUNA_STORAGE_URI]:-}}"
    OPTUNA_DASHBOARD_STORAGE_URI="${SECRETS[OPTUNA_DASHBOARD_STORAGE_URI]:-}"
    AIRFLOW_DB_URI="${SECRETS[AIRFLOW__DATABASE__SQL_ALCHEMY_CONN]:-${SECRETS[AIRFLOW_DB_URI]:-}}"
fi

PG_HOST="postgres.taxon-vision.svc.cluster.local:5432"
TELEMETRY_DATABASE_URL="${TELEMETRY_DATABASE_URL:-postgresql+psycopg://telemetry:${TELEMETRY_DB_PASSWORD}@${PG_HOST}/telemetry}"
OPTUNA_STORAGE_URI="${OPTUNA_STORAGE_URI:-postgresql+psycopg://optuna:${OPTUNA_DB_PASSWORD}@${PG_HOST}/optuna}"
OPTUNA_DASHBOARD_STORAGE_URI="${OPTUNA_DASHBOARD_STORAGE_URI:-postgresql+psycopg2://optuna:${OPTUNA_DB_PASSWORD}@${PG_HOST}/optuna}"
AIRFLOW_DB_URI="${AIRFLOW_DB_URI:-postgresql+psycopg2://airflow:${AIRFLOW_DB_PASSWORD}@${PG_HOST}/airflow}"

REQUIRED_VARS=(
    POSTGRES_PASSWORD
    TELEMETRY_DB_PASSWORD
    OPTUNA_DB_PASSWORD
    AIRFLOW_DB_PASSWORD
    SEAWEEDFS_ACCESS_KEY_ID
    SEAWEEDFS_SECRET_ACCESS_KEY
    GF_SECURITY_ADMIN_PASSWORD
    AIRFLOW_FERNET_KEY
    AIRFLOW_JWT_SECRET
    AIRFLOW_ADMIN_PASSWORD
    JENKINS_API_TOKEN
    INTERNAL_API_TOKEN
    RETRAIN_WEBHOOK_SECRET
    JENKINS_USER
)

for var_name in "${REQUIRED_VARS[@]}"; do
    val="${!var_name}"
    if [[ -z "${val}" ]]; then
        echo "Error: Required secret '${var_name}' is empty or missing." >&2
        exit 1
    fi
    if [[ "${DRY_RUN}" != "true" && "${val}" =~ ^(REPLACE_WITH_|CHANGE_ME_|PLACEHOLDER_) ]]; then
        echo "Error: Secret '${var_name}' still holds an unreplaced placeholder value." >&2
        exit 1
    fi
done

if [[ "${DRY_RUN}" != "true" && "${DAGSHUB_TOKEN:-}" =~ ^(REPLACE_WITH_|CHANGE_ME_|PLACEHOLDER_) ]]; then
    echo "Error: Secret 'DAGSHUB_TOKEN' still holds an unreplaced placeholder value." >&2
    exit 1
fi
if [[ -z "${DAGSHUB_TOKEN:-}" ]]; then
    echo "Note: DAGSHUB_TOKEN is empty; DagsHub Label Studio sync and MLflow experiment tracking remain inactive until configured." >&2
fi

render_s3_json() {
    local access_key="$1"
    local secret_key="$2"
    python3 -c '
import json, sys
doc = {
    "identities": [
        {
            "name": "taxon-operator",
            "credentials": [{"accessKey": sys.argv[1], "secretKey": sys.argv[2]}],
            "actions": ["Admin", "Read", "Write", "List", "Tagging"],
        }
    ]
}
print(json.dumps(doc, indent=2))
' "${access_key}" "${secret_key}"
}

mask_val() {
    local val="$1"
    if [[ "${DRY_RUN}" == "true" && "${MOCK_MODE}" == "true" ]] || [[ "${UNSAFE_REVEAL}" == "true" ]]; then
        printf '%s' "${val}"
    elif [[ "${DRY_RUN}" == "true" ]]; then
        printf '%s' "******** [MASKED - USE --unsafe-reveal TO INSPECT]"
    else
        printf '%s' "${val}"
    fi
}

emit_infra_secrets_yaml() {
    local ns="taxon-vision"
    local pg_pw tel_pw opt_pw af_pw s3_ak s3_sk gf_usr gf_pw gf_webhook
    local opt_dash_uri af_fernet af_jwt af_uri af_admin jnk_usr jnk_tok dh_tok s3_json_indented

    pg_pw="$(mask_val "${POSTGRES_PASSWORD}")"
    tel_pw="$(mask_val "${TELEMETRY_DB_PASSWORD}")"
    opt_pw="$(mask_val "${OPTUNA_DB_PASSWORD}")"
    af_pw="$(mask_val "${AIRFLOW_DB_PASSWORD}")"
    s3_ak="$(mask_val "${SEAWEEDFS_ACCESS_KEY_ID}")"
    s3_sk="$(mask_val "${SEAWEEDFS_SECRET_ACCESS_KEY}")"
    gf_usr="$(mask_val "${GF_SECURITY_ADMIN_USER}")"
    gf_pw="$(mask_val "${GF_SECURITY_ADMIN_PASSWORD}")"
    gf_webhook="$(mask_val "${RETRAIN_WEBHOOK_SECRET}")"
    opt_dash_uri="$(mask_val "${OPTUNA_DASHBOARD_STORAGE_URI}")"
    af_fernet="$(mask_val "${AIRFLOW_FERNET_KEY}")"
    af_jwt="$(mask_val "${AIRFLOW_JWT_SECRET}")"
    af_uri="$(mask_val "${AIRFLOW_DB_URI}")"
    af_admin="$(mask_val "${AIRFLOW_ADMIN_PASSWORD}")"
    jnk_usr="$(mask_val "${JENKINS_USER}")"
    jnk_tok="$(mask_val "${JENKINS_API_TOKEN}")"
    dh_tok=""
    if [[ -n "${DAGSHUB_TOKEN:-}" ]]; then
        dh_tok="$(mask_val "${DAGSHUB_TOKEN}")"
    fi
    s3_json_indented="$(render_s3_json "${s3_ak}" "${s3_sk}" | sed 's/^/    /')"

    cat <<EOF
apiVersion: v1
kind: Secret
metadata:
  name: postgres-secrets
  namespace: ${ns}
  labels:
    app.kubernetes.io/part-of: taxon-vision-mlops
type: Opaque
stringData:
  POSTGRES_PASSWORD: "${pg_pw}"
  TELEMETRY_DB_PASSWORD: "${tel_pw}"
  OPTUNA_DB_PASSWORD: "${opt_pw}"
  AIRFLOW_DB_PASSWORD: "${af_pw}"
---
apiVersion: v1
kind: Secret
metadata:
  name: seaweedfs-s3-secret
  namespace: ${ns}
  labels:
    app.kubernetes.io/part-of: taxon-vision-mlops
type: Opaque
stringData:
  AWS_ACCESS_KEY_ID: "${s3_ak}"
  AWS_SECRET_ACCESS_KEY: "${s3_sk}"
  s3.json: |
${s3_json_indented}
---
apiVersion: v1
kind: Secret
metadata:
  name: monitoring-secrets
  namespace: ${ns}
  labels:
    app.kubernetes.io/part-of: taxon-vision-mlops
type: Opaque
stringData:
  GF_SECURITY_ADMIN_USER: "${gf_usr}"
  GF_SECURITY_ADMIN_PASSWORD: "${gf_pw}"
  GF_RETRAIN_WEBHOOK_SECRET: "${gf_webhook}"
---
apiVersion: v1
kind: Secret
metadata:
  name: optuna-secrets
  namespace: ${ns}
  labels:
    app.kubernetes.io/part-of: taxon-vision-mlops
type: Opaque
stringData:
  OPTUNA_DASHBOARD_STORAGE_URI: "${opt_dash_uri}"
---
apiVersion: v1
kind: Secret
metadata:
  name: airflow-secrets
  namespace: ${ns}
  labels:
    app.kubernetes.io/part-of: taxon-vision-mlops
type: Opaque
stringData:
  AIRFLOW__CORE__FERNET_KEY: "${af_fernet}"
  AIRFLOW__API_AUTH__JWT_SECRET: "${af_jwt}"
  AIRFLOW__DATABASE__SQL_ALCHEMY_CONN: "${af_uri}"
  AIRFLOW_ADMIN_PASSWORD: "${af_admin}"
  JENKINS_USER: "${jnk_usr}"
  JENKINS_API_TOKEN: "${jnk_tok}"
  DAGSHUB_TOKEN: "${dh_tok}"
EOF
}

emit_app_secret_yaml() {
    local ns="$1"
    local opt_uri tel_uri tok webhook af_usr af_pw dh_tok
    opt_uri="$(mask_val "${OPTUNA_STORAGE_URI}")"
    tel_uri="$(mask_val "${TELEMETRY_DATABASE_URL}")"
    tok="$(mask_val "${INTERNAL_API_TOKEN}")"
    webhook="$(mask_val "${RETRAIN_WEBHOOK_SECRET}")"
    af_usr="$(mask_val "${AIRFLOW_API_USER}")"
    af_pw="$(mask_val "${AIRFLOW_ADMIN_PASSWORD}")"
    dh_tok=""
    if [[ -n "${DAGSHUB_TOKEN:-}" ]]; then
        dh_tok="$(mask_val "${DAGSHUB_TOKEN}")"
    fi

    cat <<EOF
apiVersion: v1
kind: Secret
metadata:
  name: taxon-vision-secrets
  namespace: ${ns}
  labels:
    app.kubernetes.io/part-of: taxon-vision-mlops
type: Opaque
stringData:
  OPTUNA_STORAGE_URI: "${opt_uri}"
  TELEMETRY_DATABASE_URL: "${tel_uri}"
  INTERNAL_API_TOKEN: "${tok}"
  RETRAIN_WEBHOOK_SECRET: "${webhook}"
  AIRFLOW_API_USER: "${af_usr}"
  AIRFLOW_API_PASSWORD: "${af_pw}"
  DAGSHUB_TOKEN: "${dh_tok}"
EOF
}

emit_all_manifests() {
    local namespaces=()
    if [[ "${TARGET_NAMESPACE}" == "all" ]]; then
        namespaces=("taxon-vision" "taxon-vision-staging")
    else
        namespaces=("${TARGET_NAMESPACE}")
    fi

    local first=true
    for ns in "${namespaces[@]}"; do
        if [[ "${ns}" == "taxon-vision" ]]; then
            [[ "${first}" == "false" ]] && echo "---"
            emit_infra_secrets_yaml
            first=false
        fi
        [[ "${first}" == "false" ]] && echo "---"
        emit_app_secret_yaml "${ns}"
        first=false
    done
}

if [[ "${DRY_RUN}" == "true" ]]; then
    echo "# Dry-run secret manifest generation (target=${TARGET_NAMESPACE})"
    emit_all_manifests
    exit 0
fi

KUBECTL_BIN=$(command -v kubectl || echo "")
if [[ -z "${KUBECTL_BIN}" ]]; then
    echo "Error: kubectl binary not found in PATH." >&2
    exit 1
fi

echo "Applying Kubernetes secrets (target=${TARGET_NAMESPACE})..."
emit_all_manifests | "${KUBECTL_BIN}" apply -f -
echo "Successfully provisioned encrypted Kubernetes Secrets (target=${TARGET_NAMESPACE})."
