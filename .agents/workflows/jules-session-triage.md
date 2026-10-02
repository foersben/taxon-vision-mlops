---
type: Workflow
title: Jules Session Triage
status: stable
stale_after: "2027-01-01T00:00:00Z"
version: 1.0
description: "Protocol for triaging, reviewing, and cleaning up background Jules sessions."
tags: [workflow, automation]
generated: {by: process:scaffold-init, at: "2026-10-02T10:00:00Z"}
verified: {by: process:scaffold-init, at: "2026-10-02T10:00:00Z"}
---

# Jules Session Triage (/jules-session-triage)

* **Description:** Protocol for triaging, reviewing, and cleaning up background Jules sessions.
* **Invocation:** Triggered via `/jules-session-triage` in the AI IDE.

## Step-by-Step Procedure

1. Run quality gates via `just check`.
2. Execute targeted test passes via `just test`.
3. Validate OKF frontmatter via `just validate-okf`.
4. Compile documentation build via `just docs`.
