---
type: Agent Skill
name: doc-synchronization-pipeline
title: Doc Synchronization Pipeline
status: stable
stale_after: "2027-01-01T00:00:00Z"
version: 1.0
description: Audits and synchronizes Zensical documentation against codebase changes.
tags: [skill, automation, docs]
generated: {by: process:scaffold-init, at: "2026-10-02T10:00:00Z"}
verified: {by: process:scaffold-init, at: "2026-10-02T10:00:00Z"}
---

# Doc Synchronization Pipeline

* **Description:** Audits and synchronizes Zensical documentation against codebase changes.
* **Invocation:** Triggered via `/doc-synchronization-pipeline` in the AI IDE.

## Step-by-Step Procedure

* Run quality gates via `just check`.
* Execute targeted test passes via `just test`.
* After architectural code changes, update OpenWiki documentation and grounded claims via `just wiki-update`.
* Validate OKF v0.2 structural integrity across `.agents/`, `docs/`, and `openwiki/` via `just wiki-validate`.
* Compile documentation build via `just docs`.
