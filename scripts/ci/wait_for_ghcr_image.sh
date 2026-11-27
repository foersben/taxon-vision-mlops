#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Benjamin Förster
# SPDX-License-Identifier: MIT
# Poll GHCR OCI v2 registry until the target commit SHA tag is published (or exit 1 on skip/timeout).

set -euo pipefail

SHORT_SHA="${1:?Usage: wait_for_ghcr_image.sh <short_sha> [repo]}"
GHCR_REPO="${2:-foersben/taxon-vision-mlops}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-360}"

COMMIT_MSG="$(git log -1 --pretty=%B 2>/dev/null || true)"
if echo "${COMMIT_MSG}" | grep -qiE "skip ci|ci skip"; then
    echo ">>> Commit requests skip CI; skipping GHCR image poll." >&2
    exit 1
fi

echo ">>> Waiting for GHCR image ghcr.io/${GHCR_REPO}:sha-${SHORT_SHA} (up to ${TIMEOUT_SECONDS}s)..." >&2
WAITED=0
while [ "${WAITED}" -lt "${TIMEOUT_SECONDS}" ]; do
    GHCR_TOK="$(curl -fsSL "https://ghcr.io/token?service=ghcr.io&scope=repository:${GHCR_REPO}:pull" | grep -o '"token":"[^"]*"' | cut -d'"' -f4 || true)"
    if [ -n "${GHCR_TOK}" ]; then
        CODE="$(curl -s -o /dev/null -w "%{http_code}" \
            -H "Authorization: Bearer ${GHCR_TOK}" \
            -H "Accept: application/vnd.oci.image.index.v1+json,application/vnd.docker.distribution.manifest.list.v2+json,application/vnd.oci.image.manifest.v1+json" \
            "https://ghcr.io/v2/${GHCR_REPO}/manifests/sha-${SHORT_SHA}" || echo "000")"
        if [ "${CODE}" = "200" ]; then
            echo ">>> Confirmed GHCR image sha-${SHORT_SHA} is published." >&2
            exit 0
        fi
    fi
    sleep 10
    WAITED=$((WAITED + 10))
done

echo ">>> GHCR image sha-${SHORT_SHA} not published within ${TIMEOUT_SECONDS}s." >&2
exit 1
