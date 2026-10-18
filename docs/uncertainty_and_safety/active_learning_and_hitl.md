---
type: Concept
title: Active Learning & Human-in-the-Loop Triage
status: stable
stale_after: "2027-01-01T00:00:00Z"
version: 1.0
description: Active learning query strategies maximizing marginal information value per human review hour.
tags: [active-learning, hitl, badge, coreset]
generated: {by: process:scaffold-init, at: "2026-10-02T10:00:00Z"}
verified: {by: process:docs-librarian, at: "2026-10-18T22:00:00Z"}
---

# Active Learning & Human-in-the-Loop Triage

In fine-grained biological identification, expert human verification is a finite, high-cost operational resource. When Split Conformal Prediction yields an empty set ($\mathcal{C}(x) = \emptyset$), exceeds cardinality thresholds ($|\mathcal{C}(x)| > 3$), or flags out-of-distribution inputs, TaxonVision escalates the specimen to human taxonomists.

To maximize model performance gains per review hour, the platform decouples component-verified uncertainty query scoring from future batch diversity specifications.

---

## 1. Verified Component Implementation

### Margin Uncertainty Sampling (`query.py`)

In [`src/taxon_vision/active_learning/query.py`](file:///home/benni/Documents/antigravity_workspace/taxon-vision-mlops/src/taxon_vision/active_learning/query.py), uncertainty is evaluated via multi-class margin sampling:

$$M(x) = \hat{\pi}_{(1)}(x) - \hat{\pi}_{(2)}(x)$$

$$\text{Uncertainty Score} = 1.0 - M(x) \in [0.0, 1.0]$$

Where $\hat{\pi}_{(1)}$ and $\hat{\pi}_{(2)}$ represent the top-2 predicted class probabilities. Observations with near-identical probabilities ($\hat{\pi}_{(1)} \approx \hat{\pi}_{(2)}$) yield uncertainty near $1.0$, indicating maximum boundary ambiguity. This module is standalone unit-tested in `tests/unit/test_active_learning.py`.

### In-Memory Binary Max-Heap (`triage_queue.py`)

The [`TriageQueue`](file:///home/benni/Documents/antigravity_workspace/taxon-vision-mlops/src/taxon_vision/active_learning/triage_queue.py) is implemented using Python's native `heapq` module, maintaining an $O(\log N)$ priority queue. To handle tie-breakers when two observations share identical uncertainty scores without comparing complex object dicts, a unique monotonically increasing counter is injected into the tuple: `(-priority, count, item)`.

* **Push Complexity:** $O(\log N)$ worst-case insertion time.
* **Pop Complexity:** $O(\log N)$ retrieval of the single most informative observation.
* **Algorithmic Validation:** Re-sorting standard lists on every insertion introduces an $O(N \log N)$ bottleneck that freezes active learning workers at scale. A binary heap ensures sub-millisecond priority scheduling.

### Flagged Specimen Storage (`service/storage.py`)

In the active runtime serving path (`/api/v1/predict`), whenever `is_ood` or `requires_human_review` is triggered, [`service/storage.py`](file:///home/benni/Documents/antigravity_workspace/taxon-vision-mlops/src/taxon_vision/service/storage.py) atomically persists:

1. **Feature Vector:** Normalized latent representation saved as `.npy` in `TAXON_FLAGGED_STORAGE_DIR`.
2. **Metadata Record:** Classification probabilities, conformal set, and OOD energy score serialized to `.json`.
3. **Audit Ledger:** Atomic append to `manifest.jsonl` tracking triage provenance.

---

## 2. Phase 2.5 & Phase 3 Roadmap Specifications

The following capabilities represent planned architectural extensions detailed in the Technical Strategy Report (currently zero runtime code in `src/`):

* **BADGE Gradient Acquisition:** Computing loss gradients with respect to linear classification weights ($g_x = (p - e_{\hat{y}}) \otimes z$) and sampling diverse batches via k-means++ in gradient space.
* **Persistent Parquet/DuckDB Queue:** Replacing the in-memory binary heap with an on-disk Parquet queue queried via DuckDB for cross-worker multi-node persistence.
* **Label Studio Bidirectional Sync:** Automated webhook integration synchronizing expert labeler corrections from Label Studio into retraining partitions.
