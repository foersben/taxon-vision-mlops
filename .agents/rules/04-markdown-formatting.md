---
type: Agent Rule
title: Mandates
status: stable
stale_after: "2027-01-01T00:00:00Z"
version: 1.0
description: "Global markdown rules: standard hyphens, asterisk lists, 4-space indentation, and clean spacing."
tags: [markdown, style, typography]
generated: {by: process:scaffold-init, at: "2026-10-02T10:00:00Z"}
verified: {by: process:scaffold-init, at: "2026-10-02T10:00:00Z"}
trigger: always_on
severity: critical
---

# Mandates: Markdown Formatting Rules

These rules apply globally across the workspace for editing any markdown file:

* Always use the standard hyphen (`-`) instead of the en-dash or em-dash in all Markdown documentation and UI text.
* Use `*` for all unordered lists.
* List syntax: Exactly 1 space after `*`, indented 4 spaces per level (0 spaces at level 0).
* Insert exactly 1 blank line before/after lists, code blocks, and headings.
* Trim all trailing whitespaces at line ends.
