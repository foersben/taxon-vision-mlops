---
type: Concept
title: Four-Layer Calibration Architecture Deep-Dive
status: stable
stale_after: "2027-01-01T00:00:00Z"
version: 1.0
description: Exhaustive analysis of the four decoupled calibration mechanisms in TaxonVision, spanning logit temperature, conformal quantiles, OOD free energy, and INT8 dynamic range.
tags: [calibration, conformal-prediction, uncertainty, ece, free-energy, quantization]
generated: {by: process:docs-librarian, at: "2026-10-18T12:00:00Z"}
verified: {by: process:docs-librarian, at: "2026-10-18T12:00:00Z"}
---

# Four-Layer Calibration Architecture Deep-Dive

A common architectural error in applied computer vision is treating "model calibration" as a single monolithic step. In high-stakes biodiversity monitoring, different operational stages require fundamentally different mathematical guarantees. TaxonVision-MLOps decouples calibration into **four distinct, non-overlapping architectural layers**:

```mermaid
flowchart TD
    subgraph L4["Layer 4: Integer Quantization Calibration"]
        Q["TaxonCalibrationDataReader<br/>(inference.calibration)"] -->|Scales S & Zero-Points Z| ONNX["INT8 ONNX Engine<br/>(Optimal Dynamic Range)"]
    end

    subgraph L1["Layer 1: Logit Probability Calibration"]
        T["Temperature Scaling T<br/>(uncertainty.conformal)"] -->|Minimizes ECE| PROB["Calibrated Softmax<br/>exp(z/T) / Sum"]
    end

    subgraph L3["Layer 3: Thermodynamic OOD Boundary"]
        E["Energy Cutoff tau_OOD<br/>(uncertainty.ood_detector)"] -->|95% In-Dist TPR| FILTER["Rejects Non-Biological Objects<br/>E(x; T) > tau_OOD"]
    end

    subgraph L2["Layer 2: Finite-Sample Conformal Sets"]
        C["Quantile Inflation q_hat<br/>(pipelines.calibrate)"] -->|1 - alpha Coverage| SETS["Multi-Class Prediction Sets<br/>P(Y in C(X)) >= 0.95"]
    end
```

---

## 1. Calibration Taxonomy & Comparison Matrix

| Architectural Layer | Calibration Target | Calibrated Parameter | Pipeline Module | Mathematical Invariant / Guarantee |
| :--- | :--- | :--- | :--- | :--- |
| **Layer 1** | Logit Confidence | Temperature $T \in \mathbb{R}^+$ | `conformal.py` | Minimizes Expected Calibration Error (ECE) via scaled logits $z/T$. |
| **Layer 2** | Non-Conformity Set | Empirical Quantile $\hat{q}_\alpha$ | `pipelines/calibrate.py` | Finite-sample marginal coverage: $\mathbb{P}(Y \in \mathcal{C}(X)) \ge 1 - \alpha$. |
| **Layer 3** | Semantic OOD Boundary | Energy Cutoff $\tau_{\text{OOD}}$ | `uncertainty/ood_detector.py` | Guarantees 95% in-distribution True Positive Rate: $E(x; T) \le \tau_{\text{OOD}}$. |
| **Layer 4** | Static INT8 Activations | Scales $S$, Zero-Points $Z$ | `inference/calibration.py` | Optimal dynamic range $[r_{\min}, r_{\max}]$ clipping via QDQ operators. |

---

## 2. Decoupled Safety Principle: Why Four Layers?

A single calibration mechanism cannot satisfy all operational and epistemic requirements:

* **Why temperature scaling cannot guarantee coverage:** Temperature scaling applies a strictly monotonic transformation to logits. While it aligns average confidence with empirical accuracy across a validation split, it does not output multi-class prediction sets or provide finite-sample guarantees for individual ambiguous observations.
* **Why conformal prediction cannot detect out-of-domain objects:** Softmax probabilities are shift-invariant: $\text{softmax}(z) = \text{softmax}(z + c)$. If an arbitrary non-biological photo (e.g., a car) produces uniform negative logits ($z_k \approx -15$), softmax still forces $\sum \pi_k = 1.0$, which can accidentally pass conformal thresholds.
* **Why free energy is required:** Helmholtz Free Energy preserves the absolute magnitude of logit activations:
  $$E(x; T) = -T \ln \sum_{k=1}^K \exp\left(\frac{z_k}{T}\right)$$
  When uninformative background clutter is presented, total activation is negligible, pushing $E(x; T) \gg \tau_{\text{OOD}}$ and triggering human triage before classification begins.
* **Why INT8 calibration is kept separate:** Quantization operates on hardware execution efficiency. Running INT8 calibration with representative iNaturalist specimens minimizes rounding error on CPU matrix multiplication without altering the higher-level decision boundaries.
