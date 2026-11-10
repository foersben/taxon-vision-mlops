---
type: Agent Rule
title: "Rule 03: Git Security & Signing"
status: stable
stale_after: "2027-01-01T00:00:00Z"
version: 1.0
description: "Mandatory GPG/SSH commit signing, signature validation, and failure escalation protocol."
tags: [git, security, signing, gpg]
generated: {by: process:scaffold-init, at: "2026-10-02T10:00:00Z"}
verified: {by: process:okf-updater, at: "2026-10-30T00:00:00Z"}
trigger: always_on
severity: critical
---

# Mandates: Git Security & Signing

Cryptographic integrity and non-repudiation of all repository contributions are strictly enforced at the commit level.

## 1. Mandatory Cryptographic Commit Signing

* Every Git commit created by human maintainers or AI agents must be cryptographically signed using a valid GPG or SSH key (`git commit -S`).
* Commits must verify cleanly under `git verify-commit HEAD` and display the verified signature badge on upstream GitHub remotes.
* The command-line flag `--no-gpg-sign` is strictly forbidden across all workflows, scripts, and agent tool invocations.

## 2. Hard Escalation on Verification Failure

* If a Git commit fails due to a missing cryptographic key, GPG agent timeout, or pinentry error, the agent must **halt immediately and report the error to the user**.
* Bypassing signing checks by modifying Git configuration (`git config commit.gpgsign false`), unsetting signing keys, or using `--no-verify` is a critical security violation.
* All production releases and milestone checkpoints must be accompanied by cryptographically signed annotated tags (`git tag -s -m ...`).
