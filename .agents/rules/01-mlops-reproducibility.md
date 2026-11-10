---
type: Agent Rule
title: "Rule 01: MLOps Reproducibility"
status: stable
stale_after: "2027-01-01T00:00:00Z"
version: 1.0
description: "Deterministic PRNG seeding, DVC dataset immutability, split ratios, and SHA256 verification."
tags: [mlops, reproducibility, dvc, determinism]
generated: {by: process:scaffold-init, at: "2026-10-02T10:00:00Z"}
verified: {by: process:okf-updater, at: "2026-10-30T00:00:00Z"}
trigger: always_on
severity: critical
---

# Mandates: MLOps Reproducibility

All training, data processing, and evaluation pipelines must enforce strict mathematical and operational reproducibility.

## 1. Deterministic PRNG Seeding Protocol

* All pseudo-random number generators across Python `random`, `numpy.random`, and `torch` must be initialized strictly from the global seed declared in `params.yaml:seed` (canonical baseline: `42`).
* In PyTorch execution blocks, deterministic operations must be enforced:

  ```python
  torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False
  ```

* Hardcoding ad-hoc seeds (e.g. `seed=123` or unseeded `random()`) in application or pipeline code is strictly forbidden.

## 2. Dataset Immutability & DVC Versioning

* All data transformations are orchestrated as pure functions through the 7-stage DVC pipeline defined in `dvc.yaml`.
* Stage inputs and outputs must be tracked via cryptographic content hashes recorded in `dvc.lock`. Manual editing of `dvc.lock` is strictly forbidden.
* Intermediate tensor artifacts (`train_embeddings.pt`, `val_embeddings.pt`, `cal_embeddings.pt`, `test_embeddings.pt`) must never be modified in-place; re-running feature extraction must reproduce identical MD5 checksums.

## 3. Epistemic Diamond Split Invariant

* Observation manifests must strictly partition into four non-overlapping splits adhering to the canonical ratios:
    * **Train Split:** 70% of compliant observations (used solely for cosine head parameter updates).
    * **Validation Split:** 10% of compliant observations (used for Optuna HPO early stopping).
    * **Calibration Split:** 10% of compliant observations (reserved exclusively for Split Conformal Prediction non-conformity quantile and OOD calibration).
    * **Test Split:** 10% of compliant observations (unbiased holdout for final SLO and accuracy reporting).
* Split leakage (e.g. utilizing calibration observations during training or tuning) invalidates statistical coverage and constitutes a fatal pipeline error.
