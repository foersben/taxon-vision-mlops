---
type: Workflow
title: Conformal Uncertainty Audit
status: stable
stale_after: "2027-01-01T00:00:00Z"
version: 1.0
description: "Verifies statistical coverage guarantees and human referral rates."
tags: [workflow, automation]
generated: {by: process:scaffold-init, at: "2026-10-02T10:00:00Z"}
verified: {by: process:scaffold-init, at: "2026-10-02T10:00:00Z"}
---

# Conformal Uncertainty Audit (/audit-conformal)

* **Description:** Verifies statistical coverage guarantees and human referral rates.
* **Invocation:** Triggered via `/audit-conformal` in the AI IDE.

## Step-by-Step Procedure

1. Run quality gates via `just check`.
2. Execute targeted test passes via `just test`.
3. Validate OKF frontmatter via `just validate-okf`.
4. Compile documentation build via `just docs`.
