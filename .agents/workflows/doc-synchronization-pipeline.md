---
type: Workflow
title: Doc Synchronization Pipeline
status: stable
stale_after: "2027-01-01T00:00:00Z"
version: 1.0
description: "Audits and synchronizes Zensical documentation against codebase changes."
tags: [workflow, automation]
generated: {by: process:scaffold-init, at: "2026-10-02T10:00:00Z"}
verified: {by: process:scaffold-init, at: "2026-10-02T10:00:00Z"}
---

# Doc Synchronization Pipeline (/doc-synchronization-pipeline)

* **Description:** Audits and synchronizes Zensical documentation against codebase changes.
* **Invocation:** Triggered via `/doc-synchronization-pipeline` in the AI IDE.

## Step-by-Step Procedure

1. Run quality gates via `just check`.
2. Execute targeted test passes via `just test`.
3. Validate OKF frontmatter via `just validate-okf`.
4. Compile documentation build via `just docs`.
