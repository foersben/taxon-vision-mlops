---
type: Agent Rule
title: "Rule 06: Conformal Coverage Invariants"
status: stable
stale_after: "2027-01-01T00:00:00Z"
version: 1.0
description: "Mathematical coverage guarantees, finite-sample correction quantiles, OOD energy gating, and human referral routing."
tags: [conformal, uncertainty, statistics, ood]
generated: {by: process:scaffold-init, at: "2026-10-02T10:00:00Z"}
verified: {by: process:okf-updater, at: "2026-10-30T00:00:00Z"}
trigger: always_on
severity: critical
---

# Mandates: Conformal Coverage Invariants

All predictive inference outputs must provide rigorous statistical coverage guarantees to prevent silent misclassifications.

## 1. Conformal Coverage Mathematical Guarantee

* The system employs Split Conformal Prediction to construct prediction sets $C(X) \subseteq \{1, \dots, K\}$ that mathematically guarantee marginal coverage at user-specified significance level $\alpha \in (0, 1)$:
  $$P(Y_{n+1} \in C(X_{n+1})) \ge 1 - \alpha$$
* The default significance level is configured in `params.yaml:conformal:alpha` (default $\alpha = 0.05 \implies 95\%$ coverage).
* Softmax confidence scores without conformal calibration must never be presented as probability guarantees to end users.

## 2. Finite-Sample Quantile Computation

* Given calibration non-conformity scores $s_i = 1 - \hat{\pi}(y_i \mid x_i)$ for $i \in \{1, \dots, n\}$, the non-conformity threshold $\hat{q}$ must be computed using the exact finite-sample correction index:
  $$p = \frac{\lceil (n + 1)(1 - \alpha) \rceil}{n}$$
* Approximating $\hat{q}$ via continuous asymptotic normal assumptions or ignoring the finite-sample correction factor is strictly forbidden.

## 3. Mandatory Human Referral Boundary

* **Empty Prediction Sets ($|C(X)| = 0$):** Occur when the model is entirely unfamiliar with the observation. The system must defer the classification to human experts via the triage queue (`status="needs_review"`).
* **Ambiguous Prediction Sets ($|C(X)| > k_{\text{max}}$):** When the cardinality of the prediction set exceeds the ambiguity ceiling defined in `params.yaml:conformal:max_prediction_set_size` (default $k_{\text{max}} = 5$), the observation must be flagged for expert review.
* Arbitrarily truncating large prediction sets to top-1 without routing to human review is a safety violation.

## 4. Energy-Based Out-of-Distribution (OOD) Gating

* Negative Helmholtz free energy scores determine whether an input observation is in-distribution:
  $$E(x; T) = -T \log \sum_{k=1}^K \exp\left(\frac{f_k(x)}{T}\right)$$
* The OOD threshold $\tau_{\text{OOD}}$ must be calibrated on the calibration split to achieve at least 95% True Positive Rate (TPR) on in-distribution observations.
* Inputs with energy $E(x; T) > \tau_{\text{OOD}}$ must trigger immediate OOD routing before conformal set generation.
