---
type: Specification
title: Conformal Prediction Theory & Coverage Guarantees
status: stable
stale_after: "2027-01-01T00:00:00Z"
version: 1.0
description: Split conformal prediction mathematics guaranteeing marginal coverage error bounds.
tags: [conformal-prediction, uncertainty, statistics]
generated: {by: process:scaffold-init, at: "2026-10-02T10:00:00Z"}
verified: {by: process:scaffold-init, at: "2026-10-02T10:00:00Z"}
---

# Conformal Prediction Theory & Coverage Guarantees

For a user-specified significance level $\alpha \in (0, 1)$ (e.g., $\alpha = 0.05$ for 95% marginal coverage), given calibration non-conformity scores $s_i = 1 - \hat{f}(X_i)_{Y_i}$, we compute the conformal quantile $\hat{q}$:

$$\hat{q} = \text{Quantile}\left( \frac{\lceil (n+1)(1-\alpha) \rceil}{n}; \; s_1, \dots, s_n \right)$$

The prediction set for a novel image $X_{\text{test}}$ is then dynamically constructed:

$$C(X_{\text{test}}) = \left\{ y \in \mathcal{Y} : 1 - \hat{f}(X_{\text{test}})_y \le \hat{q} \right\}$$

Providing the finite-sample distribution-free theoretical guarantee:

$$P\left( Y_{\text{test}} \in C(X_{\text{test}}) \right) \ge 1 - \alpha$$
