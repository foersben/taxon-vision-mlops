---
type: Guide
title: Interactive Conformal & OOD Simulator
status: stable
stale_after: "2027-01-01T00:00:00Z"
version: 1.0
description: Interactive simulation workbench exploring split conformal coverage guarantees, finite-sample quantile inflation, and Helmholtz free energy gating.
tags: [interactive, simulator, conformal-prediction, uncertainty, free-energy, ood]
generated: {by: process:docs-librarian, at: "2026-10-18T12:00:00Z"}
verified: {by: process:docs-librarian, at: "2026-10-18T12:00:00Z"}
---

# Interactive Conformal & OOD Simulator

Explore the mathematical mechanics governing TaxonVision-MLOps uncertainty bounds. Adjust the significance level $\alpha$, Helmholtz Free Energy cutoff $\tau_{\text{OOD}}$, and logit temperature $T$ to observe real-time impacts on coverage guarantees, empirical quantiles, and human triage dispatch rates.

---

## 1. Live Parameter Simulator

<div class="simulator-panel">
    <div class="simulator-controls">
        <div class="simulator-control-group">
            <label for="sim-alpha-slider">
                <span>Significance Level (&alpha;)</span>
                <span id="sim-alpha-val" class="badge-tag info">0.05</span>
            </label>
            <input type="range" id="sim-alpha-slider" min="0.01" max="0.20" step="0.01" value="0.05">
            <small style="opacity: 0.7;">Error tolerance. Sets nominal coverage to (1 - &alpha;).</small>
        </div>
        <div class="simulator-control-group">
            <label for="sim-energy-slider">
                <span>OOD Energy Cutoff (&tau;<sub>OOD</sub>)</span>
                <span id="sim-energy-val" class="badge-tag warning">-11.5</span>
            </label>
            <input type="range" id="sim-energy-slider" min="-15.0" max="-8.0" step="0.1" value="-11.5">
            <small style="opacity: 0.7;">Helmholtz Free Energy threshold for out-of-distribution rejection.</small>
        </div>
        <div class="simulator-control-group">
            <label for="sim-temp-slider">
                <span>Logit Temperature (T)</span>
                <span id="sim-temp-val" class="badge-tag success">1.0</span>
            </label>
            <input type="range" id="sim-temp-slider" min="0.5" max="3.0" step="0.1" value="1.0">
            <small style="opacity: 0.7;">Softmax scaling factor minimizing Expected Calibration Error.</small>
        </div>
    </div>
    <div class="simulator-results">
        <div class="simulator-result-item">
            <div class="label">Target Coverage</div>
            <div class="val" id="sim-out-coverage">95.0%</div>
        </div>
        <div class="simulator-result-item">
            <div class="label">Empirical Quantile (&qcirc;&alpha;)</div>
            <div class="val" id="sim-out-qhat">0.8421</div>
        </div>
        <div class="simulator-result-item">
            <div class="label">Mean Set Size |C(X)|</div>
            <div class="val" id="sim-out-setsize">1.72</div>
        </div>
        <div class="simulator-result-item">
            <div class="label">Human Triage Rate</div>
            <div class="val" id="sim-out-triage">5.4%</div>
        </div>
    </div>
</div>

---

## 2. Mathematical Formulations

### Finite-Sample Quantile Inflation

In Split Conformal Prediction, taking the standard asymptotic $(1 - \alpha)$ empirical quantile results in under-coverage due to finite-sample sampling variance. For $N_{\text{cal}}$ exchangeable calibration samples, we calculate the inflated quantile index $p$:

$$p = \frac{\lceil(N_{\text{cal}} + 1)(1 - \alpha)\rceil}{N_{\text{cal}}}$$

The non-conformity threshold $\hat{q}_\alpha$ is computed as the $p$-th empirical quantile over calibration true-class residual non-conformity scores:

$$s_i = 1 - \hat{\pi}_{y_i}(x_i)$$

$$\hat{q}_\alpha = \text{Quantile}\left(\{s_i\}_{i=1}^{N_{\text{cal}}}, \; p\right)$$

### The Conformal Decision Rule

For an incoming test photograph $x_{\text{test}}$, the multi-class prediction set $\mathcal{C}(x_{\text{test}})$ admits all taxa whose predicted probabilities meet the threshold:

$$\mathcal{C}(x_{\text{test}}) = \left\{ k \in \{1, \dots, K\} : \hat{\pi}_k(x_{\text{test}}) \ge 1 - \hat{q}_\alpha \right\}$$

Under exchangeability, this rule guarantees:

$$\mathbb{P}\left( Y_{\text{test}} \in \mathcal{C}(X_{\text{test}}) \right) \ge 1 - \alpha$$

---

## 3. Helmholtz Free Energy Decomposition

The system computes Helmholtz Free Energy $E(x; T)$ directly from raw logits $z \in \mathbb{R}^K$:

$$E(x; T) = -T \ln \sum_{k=1}^K \exp\left(\frac{z_k}{T}\right)$$

This decomposes into the expected internal energy $U(x)$ and scaled Shannon entropy $H(\hat{\pi})$:

$$E(x; T) = \underbrace{\sum_{k=1}^K \hat{\pi}_k(x)(-z_k)}_{U(x) \text{ (Activation Deficit)}} - T \cdot \underbrace{\left(-\sum_{k=1}^K \hat{\pi}_k(x)\ln \hat{\pi}_k(x)\right)}_{H(\hat{\pi}) \text{ (Decision Ambiguity)}}$$

* **Confident Inlier:** Low energy $E(x; T) \le \tau_{\text{OOD}}$ and low entropy $H(\hat{\pi}) \to 0$. Resolves via sub-15ms fast path.
* **Sister-Taxa Ambiguity:** Low energy (valid organism detected) but high entropy between two competing species. Generates Vision-RAG differential diagnosis card.
* **Semantic Outlier / Non-Biological Input:** Elevated free energy $E(x; T) > \tau_{\text{OOD}}$. Diverted to human taxonomist triage queue.
