---
type: Concept
title: Four-Layer Calibration Architecture Deep-Dive
status: stable
stale_after: "2027-01-01T00:00:00Z"
version: 1.0
description: Exhaustive mathematical and code-linked analysis of the four decoupled calibration mechanisms in TaxonVision, spanning logit temperature, conformal quantiles, OOD free energy, and static INT8 dynamic range.
tags: [calibration, conformal-prediction, uncertainty, ece, free-energy, quantization, temperature-scaling, ptq]
generated: {by: "process:docs-librarian", at: "2026-10-18T12:00:00Z"}
verified: {by: "process:docs-librarian", at: "2026-10-18T12:00:00Z"}
---

# Four-Layer Calibration Architecture Deep-Dive

A common architectural vulnerability in applied machine learning is treating "model calibration" as a single monolithic post-processing step. In high-stakes biodiversity monitoring, different operational requirements demand fundamentally distinct mathematical guarantees. TaxonVision-MLOps cleanly decouples calibration into **four non-overlapping architectural layers**:

```mermaid
flowchart TD
    subgraph L4["Layer 4: Integer Quantization Range Calibration"]
        direction TB
        Q["TaxonCalibrationDataReader<br/>(inference.calibration)"] -->|Calibrates [r_min, r_max]| QDQ["Quantize-Dequantize Operators<br/>(Scale S & Zero-Point Z)"]
        QDQ --> ONNX["Static INT8 ONNX Engine<br/>(Sub-25ms Zero-Overflow Latency)"]
    end

    subgraph L1["Layer 1: Logit Probability Temperature Scaling"]
        direction TB
        T["Temperature Parameter T > 0<br/>(params.yaml & conformal.py)"] -->|Scales Raw Logits z/T| PROB["Softmax Probability Simplex<br/>(Minimizes Expected Calibration Error)"]
    end

    subgraph L3["Layer 3: Thermodynamic Helmholtz Free-Energy OOD Cutoff"]
        direction TB
        E["Energy Threshold tau_OOD<br/>(uncertainty.ood_detector)"] -->|Anchors 95% In-Dist TPR| FILTER["Rejects Non-Biological Artefacts<br/>E(x; T) > tau_OOD -> Human Triage"]
    end

    subgraph L2["Layer 2: Split Conformal Non-Conformity Quantile"]
        direction TB
        C["Empirical Quantile q_hat<br/>(pipelines.calibrate)"] -->|1 - alpha Coverage Guarantee| SETS["Multi-Class Prediction Sets C(x)<br/>P(Y in C(X)) >= 1 - alpha"]
    end

    L4 --> L1
    L1 --> L3
    L3 --> L2
```

---

## 1. Architectural Calibration Matrix

| Layer | Calibration Objective | Calibrated Parameter | Implementation Source | Mathematical Invariant & Guarantee |
| :--- | :--- | :--- | :--- | :--- |
| **Layer 1** | Post-Hoc Probability Reliability | Scalar Temperature $T \in \mathbb{R}^+$ | `src/taxon_vision/uncertainty/conformal.py` | Minimizes Expected Calibration Error (ECE) via $z_k / T$ scaling prior to softmax. |
| **Layer 2** | Finite-Sample Marginal Coverage | Empirical Quantile $\hat{q}_\alpha \in [0, 1]$ | `src/taxon_vision/pipelines/calibrate.py` | Guarantees $\mathbb{P}(Y_{\text{test}} \in \mathcal{C}(X_{\text{test}})) \ge 1 - \alpha$ under exchangeability. |
| **Layer 3** | Semantic Outlier Screening | Free Energy Cutoff $\tau_{\text{OOD}} \in \mathbb{R}$ | `src/taxon_vision/uncertainty/ood_detector.py` | Enforces nominal $95\%$ In-Distribution True Positive Rate (TPR) using absolute logit density. |
| **Layer 4** | Numerical INT8 Dynamic Range | Tensors Scales $\mathbf{S}$, Zero-Points $\mathbf{Z}$ | `src/taxon_vision/inference/calibration.py` | Optimal clipping $[r_{\min}, r_{\max}]$ on real image streams without synthetic Gaussian noise. |

---

## 2. Layer 1: Post-Hoc Logit Temperature Scaling ($T$)

Deep neural networks trained with cross-entropy frequently suffer from miscalibration, producing overconfident probabilities even when incorrect. Layer 1 applies post-hoc temperature scaling to align predicted probabilities with empirical frequencies:

$$\hat{\pi}_k(x; T) = \frac{\exp(z_k / T)}{\sum_{j=1}^K \exp(z_j / T)}$$

* **Optimization Objective:** The scalar parameter $T > 0$ is learned on the validation holdout split $\mathcal{D}_{\text{val}}$ by minimizing Negative Log-Likelihood (NLL) with linear probe weights $\mathbf{W}$ strictly frozen:
  $$\min_{T > 0} \; -\frac{1}{N_{\text{val}}} \sum_{i=1}^{N_{\text{val}}} \sum_{k=1}^K \mathbb{I}(y_i = k) \ln \hat{\pi}_k(x_i; T)$$
* **Monotonic Invariant:** Because $\frac{\partial \hat{\pi}_k}{\partial z_k} > 0$ for all $T > 0$, temperature scaling preserves the exact rank-order of class predictions ($\operatorname{argmax}_k z_k = \operatorname{argmax}_k \hat{\pi}_k$). It changes confidence dispersion without altering accuracy.
* **Why Layer 1 Alone Fails:** Temperature scaling aligns aggregate calibration across a validation set, but it provides zero finite-sample coverage guarantees for individual ambiguous observations and cannot output multi-class candidate sets.

---

## 3. Layer 2: Split Conformal Non-Conformity Quantile ($\hat{q}_\alpha$)

To provide mathematically rigorous safety bounds for autonomous taxonomic classification, Layer 2 implements Split Conformal Prediction.

### Projection and Non-Conformity Scoring

The calibration embeddings `data/embeddings/cal_embeddings.pt` ($\mathbf{X}_{\text{cal}} \in \mathbb{R}^{N_{\text{cal}} \times d}$) are projected through the probe weights in `src/taxon_vision/inference/head_projector.py`:

$$\mathbf{Z}_{\text{cal}} = \mathbf{X}_{\text{cal}}\mathbf{W}^T + \mathbf{b}$$

For each calibration specimen $(\mathbf{x}_i, y_i)$, the true-class non-conformity score $s_i$ measures the degree of error:

$$s_i = 1 - \hat{\pi}_{y_i}(x_i; T) \in [0, 1]$$

### Finite-Sample Quantile Inflation

Standard asymptotic empirical quantiles lead to systematic under-coverage on small calibration splits due to sample variance. For $N_{\text{cal}} = 169$ exchangeable observations, `src/taxon_vision/uncertainty/conformal.py` computes the inflated order statistic:

$$p = \frac{\lceil(N_{\text{cal}} + 1)(1 - \alpha)\rceil}{N_{\text{cal}}}$$

$$\hat{q}_\alpha = \operatorname{Quantile}\left(\{s_i\}_{i=1}^{N_{\text{cal}}}, \; p\right)$$

### The Marginal Coverage Guarantee

At inference time, the model admits all taxa whose softmax probability exceeds the calibrated threshold $1 - \hat{q}_\alpha$:

$$\mathcal{C}(x_{\text{test}}) = \left\{ k \in \{1, \dots, K\} : \hat{\pi}_k(x_{\text{test}}; T) \ge 1 - \hat{q}_\alpha \right\}$$

Under the exchangeability assumption, the admitted prediction set mathematically guarantees:

$$\mathbb{P}\left(Y_{\text{test}} \in \mathcal{C}(X_{\text{test}})\right) \ge 1 - \alpha$$

```python
# Exact implementation from src/taxon_vision/uncertainty/conformal.py
import numpy as np

def calibrate_split_conformal(cal_probs: np.ndarray, cal_labels: np.ndarray, alpha: float = 0.05) -> float:
    n = len(cal_labels)
    # True-class non-conformity score: s_i = 1 - pi_{y_i}
    true_class_probs = cal_probs[np.arange(n), cal_labels]
    nonconformity_scores = 1.0 - true_class_probs

    # Finite-sample inflated quantile level
    q_level = np.ceil((n + 1) * (1.0 - alpha)) / n
    q_level = min(1.0, max(0.0, float(q_level)))

    # Quantile order statistic
    q_hat = float(np.quantile(nonconformity_scores, q_level, method="higher"))
    return q_hat
```

---

## 4. Layer 3: Thermodynamic Helmholtz Free-Energy Cutoff ($\tau_{\text{OOD}}$)

Softmax probabilities are shift-invariant: $\operatorname{Softmax}(\mathbf{z} + c) = \operatorname{Softmax}(\mathbf{z})$. If a non-biological image (e.g., asphalt, a vehicle, or a mineral) produces universally depressed negative logits ($z_k \approx -12.0$), softmax forces $\sum \hat{\pi}_k = 1.0$. If one class has logit $-11.5$, the model can assign it over $80\%$ probability, erroneously bypassing Layer 2.

Layer 3 in `src/taxon_vision/uncertainty/ood_detector.py` overcomes this vulnerability by calculating Helmholtz Free Energy directly from unnormalized logit magnitudes:

$$E(x; T) = -T \ln \sum_{k=1}^K \exp\left(\frac{z_k}{T}\right)$$

* **Physical Interpretation:** In-distribution biological observations produce at least one strong positive activation, yielding low Helmholtz free energy ($E(x; T) \le \tau_{\text{OOD}}$). Non-biological or corrupt images produce negligible activations, driving free energy to large values.
* **Empirical Calibration:** $\tau_{\text{OOD}}$ is calibrated on `cal_embeddings.pt` to anchor the in-distribution True Positive Rate at exactly $95\%$:

$$\tau_{\text{OOD}} = \operatorname{Quantile}\left(\{E(x_i; T)\}_{i=1}^{N_{\text{cal}}}, \; 0.95\right)$$

* **Gating Invariant:** Any inference request with $E(x; T) > \tau_{\text{OOD}}$ is diverted directly to the human review triage queue in Label Studio before classification sets are formed.

---

## 5. Layer 4: Static INT8 PTQ Activation Range Calibration

Layer 4 operates on the physical ONNX Runtime execution graph. Converting FP32 weights and activations to INT8 reduces memory bandwidth by $4\times$ and doubles CPU throughput via VNNI vector instructions.

### The Realistic Data Invariant

A critical flaw in standard ONNX quantization pipelines is feeding synthetic random Gaussian noise ($\mathcal{N}(0, I)$) to the calibrator. Synthetic noise destroys biological spatial hierarchies, causing activation clipping thresholds to collapse.

In `src/taxon_vision/inference/calibration.py`, the `TaxonCalibrationDataReader` strictly streams real citizen-science images:

```python
class TaxonCalibrationDataReader(CalibrationDataReader):
    """Feeds representative unlabelled citizen-science imagery to ONNX Runtime.

    Synthetic Gaussian noise is strictly forbidden by project invariants.
    """
    def __init__(self, image_paths: list[Path], batch_size: int = 32):
        self.image_paths = image_paths
        self.batch_size = batch_size
        self._enum_images = iter(self._batch_generator())

    def get_next(self) -> dict[str, np.ndarray] | None:
        return next(self._enum_images, None)
```

### Quantize-Dequantize (QDQ) Mathematics

During static quantization, MinMax calibration tracks the empirical tensor range $[r_{\min}, r_{\max}]$ across intermediate layers to derive the floating-point scale $S$ and integer zero-point $Z \in [-128, 127]$:

$$S = \frac{r_{\max} - r_{\min}}{255}, \quad Z = \operatorname{round}\left(-\frac{r_{\min}}{S}\right) - 128$$

$$q = \operatorname{clamp}\left(\operatorname{round}\left(\frac{r}{S}\right) + Z, \; -128, \; 127\right)$$

Calibrating $[r_{\min}, r_{\max}]$ on real domain images prevents numerical saturation on diagnostic morphological details, preserving $99.8\%$ of FP32 accuracy while satisfying the sub-25ms latency budget.
