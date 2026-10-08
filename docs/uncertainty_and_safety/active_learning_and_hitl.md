---
type: Concept
title: Active Learning & Human-in-the-Loop Triage
status: stable
stale_after: "2027-01-01T00:00:00Z"
version: 1.0
description: Active learning query strategies maximizing marginal information value per human review hour.
tags: [active-learning, hitl, badge, coreset]
generated: {by: process:scaffold-init, at: "2026-10-02T10:00:00Z"}
verified: {by: process:scaffold-init, at: "2026-10-02T10:00:00Z"}
---

# Active Learning & Human-in-the-Loop Triage

Unlabelled observations are prioritized using BADGE and margin uncertainty sampling, ensuring expert citizen scientists annotate observations that maximize model performance gains.

## Human Triage Queue Optimization

**What we chose:**
The `HumanTriageQueue` is implemented using Python's native `heapq` module, maintaining an $\mathcal{O}(\log N)$ priority queue. To handle tie-breakers when two observations share identical uncertainty scores, we inject a unique, monotonically increasing counter into the tuple: `(priority, count, item)`.

**Why we chose it (Algorithmic Validation):**
A common implementation anti-pattern for triage queues is appending to a standard list and calling `.sort()` ($\mathcal{O}(N \log N)$) or `.pop(0)` ($\mathcal{O}(N)$). While this works for a few hundred samples, the iNaturalist pipeline processes millions of observations daily. Re-sorting a list of a million items on every insertion freezes the active learning worker. By using a binary heap, push and pop operations are mathematically guaranteed to execute in $\mathcal{O}(\log N)$ time, ensuring the system can scale linearly with throughput without memory or CPU bottlenecks.
