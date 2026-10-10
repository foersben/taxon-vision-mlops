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

For a user-specified significance level $\alpha \in (0, 1)$ (e.g., $\alpha = 0.05$ for 95% marginal coverage), we define the non-conformity score for a calibration sample as $s_i = 1 - \hat{f}(X_i)_{Y_i}$. Based on the calibration set of size $n$, we compute the conformal quantile $\hat{q}$:

$$\hat{q} = \text{Quantile}\left(1 - \alpha; \frac{\lceil (n+1)(1-\alpha) \rceil}{n}\right)$$

The prediction set $\mathcal{C}(X_{\text{test}})$ for a novel image $X_{\text{test}}$ is then dynamically constructed:

$$\mathcal{C}(X_{\text{test}}) = \left\{ y \in \mathcal{Y} : 1 - \hat{f}(X_{\text{test}})_y \le \hat{q} \right\}$$

This provides the finite-sample distribution-free theoretical guarantee that the true label $Y_{n+1}$ will be included in the prediction set $\mathcal{C}(X_{n+1})$ with probability at least $1 - \alpha$:

$$P(Y_{n+1} \in \mathcal{C}(X_{n+1})) \ge 1 - \alpha$$

## Empty Prediction Sets

In some instances, the model may be highly uncertain, and no class satisfies the threshold condition, resulting in an empty prediction set $\mathcal{C}(X) = \emptyset$. When an empty set is generated, it triggers a mandatory escalation to human taxonomist triage, preventing the system from making unsupported deterministic predictions on highly anomalous inputs.
