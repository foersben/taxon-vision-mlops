---
type: concept
title: Uncertainty Quantification & Safety
description: Mathematical foundation and operational implementation of split conformal prediction, Helmholtz free-energy out-of-distribution detection, forward-only CAM attribution, and active learning human triage.
tags: [uncertainty, conformal-prediction, ood-detection, free-energy, cam, active-learning, triage, safety]
verified:
  - by: openwiki/0.7.2
    at: 2026-10-15T19:46:36.219Z
sources:
  - id: openwiki-source-8e72812dd1b19bee5573c46c
    resource: repo://src/taxon_vision/pipelines/calibrate.py
  - id: openwiki-source-3371ed1f3cb580f413973243
    resource: repo://src/taxon_vision/active_learning/query.py
  - id: openwiki-source-73c7bf66219b83ba25891873
    resource: repo://src/taxon_vision/active_learning/triage_queue.py
  - id: openwiki-source-62eb77aedcfd0f37b33fbbb9
    resource: repo://src/taxon_vision/inference/cam.py
  - id: openwiki-source-644dda467779a35ba1ffdc4c
    resource: repo://src/taxon_vision/service/inference.py
  - id: openwiki-source-26c079c34916fb6d294b65ef
    resource: repo://src/taxon_vision/service/routes/predict.py
  - id: openwiki-source-0617eb2bb266324f1014c040
    resource: repo://src/taxon_vision/service/storage.py
  - id: openwiki-source-1c89821d53610d1110534353
    resource: repo://src/taxon_vision/uncertainty/conformal.py
  - id: openwiki-source-a75f91091f4743d43a541d2c
    resource: repo://src/taxon_vision/uncertainty/ood_detector.py
  - id: openwiki-source-a4ac326468c5488607b2ca34
    resource: repo://tests/invariants/test_conformal_coverage_invariant.py
  - id: openwiki-source-646d9607bd1eb2274982665b
    resource: repo://tests/unit/test_active_learning.py
  - id: openwiki-source-594d85639d80f755fd98f3bb
    resource: repo://tests/unit/test_conformal_bounds.py
  - id: openwiki-source-f446a1046c42a921f75106d0
    resource: repo://tests/unit/test_models_components.py
  - id: openwiki-source-f4894f7d85bd06b9a97eeedb
    resource: repo://tests/unit/test_ood_energy.py
generated: { by: "openwiki/0.7.2", at: "2026-10-15T19:46:36.219Z" }
---

# Uncertainty Quantification & Safety

In ecological computer vision, deploying deep neural networks for autonomous biodiversity monitoring introduces substantial operational risks. Natural biological distributions follow long-tailed power laws where rare, threatened, or cryptic species exhibit subtle morphological distinctions, while sensor traps frequently capture non-organism noise, occluded scenes, lens artifacts, or out-of-catalog taxa. Standard deep classifiers parameterize outputs using the softmax function, which normalizes uncalibrated logits onto a probability simplex. Consequently, standard models frequently output extreme confidence (>99%) on catastrophic misclassifications or meaningless noise.

To prevent silent classification failures and provide mathematically bounded reliability, TaxonVision replaces heuristic probability thresholds with a formal epistemic safety and uncertainty quantification pipeline. The system couples **Helmholtz free-energy out-of-distribution (OOD) detection**, distribution-free **Split Conformal Prediction sets**, sub-3ms **forward-only Class Activation Mapping (CAM)**, and an active learning **priority triage queue** for human taxonomist referral.

For related system context, see [System Architecture & Design](../architecture/system-overview.md), [Domain Entities & Taxonomy Perimeter](../architecture/domain-and-taxa.md), [Inference & Optimization](../workflows/inference-and-optimization.md), [Service & API](../operations/service-and-api.md), and [Test Strategy & Invariants](../testing/test-strategy-and-invariants.md).

---

## Epistemic Safety Gating Architecture

The inference lifecycle enforces a tiered uncertainty gating mechanism on every photographic occurrence before predictions are emitted to automated cataloging systems or client Hypermedia / REST interfaces.

<!-- openwiki: mermaid parse failed and this diagram was converted to a text fence so it does not break rendering. Fix the diagram source and restore the mermaid fence. Parser error: Heuristic: a semicolon inside a label breaks rendering; rephrase the label. -->
```text
flowchart TD
    IMG["Input Image Capture"] --> PRE["Image Preprocessing and Normalization"]
    PRE --> FWD["Forward Model Evaluation<br/>(ONNX Runtime or PyTorch TaxonClassifier)"]
    FWD --> LOGITS["Raw Class Logits z and Embeddings"]
    
    LOGITS --> ENG["Helmholtz Free Energy Scoring<br/>E(x; T) = -T * ln( sum exp(z_i / T) )"]
    LOGITS --> SOFT["Temperature-Scaled Softmax<br/>p = softmax(z / T)"]
    
    ENG --> OOD_CHECK{"Energy E(x) > energy_threshold<br/>or Entropy H(p) > entropy_threshold?"}
    SOFT --> OOD_CHECK
    
    OOD_CHECK -- "Yes (Elevated Energy or Noise)" --> OOD_FLAG["Set is_ood = True"]
    OOD_CHECK -- "No (In-Distribution Density)" --> CONF_SET["Split Conformal Set Construction<br/>C(x) = { i | (1 - p_i) <= q_hat }"]
    
    OOD_FLAG --> GATE{"Epistemic Referral Gating"}
    CONF_SET --> GATE
    
    GATE --> COND1{"Set Cardinality |C(x)| == 0<br/>(Empty Prediction Set)?"}
    GATE --> COND2{"Set Cardinality |C(x)| > k_max<br/>(Ambiguous Prediction Set)?"}
    GATE --> COND3{"Top-1 Confidence p_max < min_confidence<br/>(Low Model Confidence)?"}
    GATE --> COND4{"is_ood Flag Asserted<br/>(Out-of-Distribution Input)?"}
    
    COND1 -- Yes --> REFER["Mandate Human Expert Review<br/>requires_human_review = True"]
    COND2 -- Yes --> REFER
    COND3 -- Yes --> REFER
    COND4 -- Yes --> REFER
    
    COND1 -- No --> PASS_ALL
    COND2 -- No --> PASS_ALL
    COND3 -- No --> PASS_ALL
    COND4 -- No --> PASS_ALL["All Referral Checks Passed"]
    
    PASS_ALL --> AUTO["Automated Classification Accepted<br/>requires_human_review = False"]
    
    REFER --> STAGE["Persist Flagged Observation<br/>(Metadata JSON, Embeddings NPY, manifest.jsonl)"]
    STAGE --> TRIAGE["Enqueue into TriageQueue<br/>Max-Heap prioritized by Uncertainty (1 - Margin)"]
    
    AUTO --> RESP["Emit PredictionResponse<br/>(Top Taxon, Valid Conformal Set, Latency)"]
    TRIAGE --> RESP
    
    RESP -. "On Diagnostic Request" .-> CAM["Forward-Only CAM Heatmap<br/>L_CAM = ReLU( sum_k w_k^c * A^k )"]
```
Uncertainty and safety decision flow: Helmholtz free-energy OOD screening, split conformal set construction, multi-criteria referral gating, and priority triage.

When an observation enters `taxon_vision.service.inference.run_prediction`:
1. The preprocessed image tensor passes through the visual backbone and classification head in a zero-gradient block, returning unnormalized logit vector $z \in \mathbb{R}^K$ and optional spatial embedding representation $e \in \mathbb{R}^D$.
2. The logits undergo temperature scaling ($z / T$) and are converted into class probabilities $p \in \Delta^{K-1}$ via softmax.
3. The detector evaluates the Helmholtz free energy score $E(x; T)$ on the raw logits and predictive entropy $H(p)$ on the probability distribution.
4. The conformal engine evaluates candidate non-conformity scores against the calibrated threshold $\hat{q}$, assembling the finite-sample prediction set $C(x)$.
5. The gating logic checks whether any triage trigger fires: out-of-distribution status, empty set, set cardinality exceeding $k_{\text{max}}$, or top-1 probability failing $p_{\text{top1}} \ge \text{min\_confidence\_threshold}$.
6. If any referral condition holds, `requires_human_review` is asserted, and the observation metadata, confidence metrics, and dense visual embedding are asynchronously persisted for expert triage and active learning ingestion.

---

## Energy-Based Out-of-Distribution (OOD) Detection

Standard deep neural classifiers rely on Maximum Softmax Probability (MSP) to detect abnormal inputs. However, because softmax applies an exponential normalization divided by the partition function:

$$p_i = \frac{\exp(z_i / T)}{\sum_{j=1}^K \exp(z_j / T)}$$

the absolute magnitude of logit activations is discarded. When presented with uniform sensor noise, blurry backgrounds, or non-organism inputs, networks can produce uncalibrated, high-magnitude logits that translate into deceptive high-probability predictions.

### Helmholtz Free Energy Scoring

Following Liu et al. (2020), TaxonVision's `EnergyOODDetector` in `src/taxon_vision/uncertainty/ood_detector.py` defines the energy surface over unnormalized logits without softmax normalization. The Helmholtz free energy $E(x; T)$ of an input $x$ producing logit vector $z = f(x)$ at temperature $T$ is defined as the negative log-partition function:

$$E(x; T) = -T \cdot \log \sum_{i=1}^K \exp\left(\frac{z_i}{T}\right)$$

In physical systems and statistical mechanics, free energy maps monotonically to negative log marginal density: $\log p(x) \propto -E(x; T)$. This mathematical relationship creates a distinct separation:
- **In-Distribution (ID) Occurrences**: Organisms belonging to cataloged taxa generate high, concentrated logit activations along matching class dimensions. The sum of exponentiated logits is large, yielding a **low, highly negative free energy score** (e.g., $E \approx -3.92$).
- **Out-of-Distribution (OOD) Occurrences**: Corrupt captures, background foliage without organisms, or unknown taxa produce low or uniform logit values across all classes. The partition function remains small, resulting in an **elevated free energy score** closer to zero or positive (e.g., uniform zeros across 10 classes yield $E = -1.0 \cdot \ln(10) \approx -2.30$).

```
Logit Magnitude Surface:
In-Distribution:     [ 3.8, -0.5, -0.5, -0.5, ... ]  -->  Energy E = -3.92 (Accepted ID)
Out-of-Distribution: [ 0.0,  0.0,  0.0,  0.0, ... ]  -->  Energy E = -2.30 (Rejected OOD)
Decision Threshold:                                       energy_threshold = -2.40
```

### Numerically Stable Computation

To prevent floating-point overflow during exponentiation in 32-bit floating-point arithmetic, `EnergyOODDetector.compute_energy` implements the LogSumExp trick with maximum logit subtraction:

```python
scaled = logits / temp
max_val = np.max(scaled)
energy = -temp * (max_val + np.log(np.sum(np.exp(scaled - max_val))))
```

### Shannon Predictive Entropy & Dual-Gating

In addition to free energy, `EnergyOODDetector.compute_entropy` calculates the Shannon predictive entropy across the normalized probability distribution $p$:

$$H(p) = -\sum_{i=1}^K p_i \ln(p_i)$$

Probabilities are clipped to $\epsilon = 10^{-12}$ to avoid numerical singularities when $p_i \to 0$. In `run_prediction`, an occurrence is marked OOD if either energy or entropy surpasses configured safety thresholds:

$$\text{is\_ood} = (E(x; T) > \text{energy\_threshold}) \lor (H(p) > \text{entropy\_threshold})$$

### Empirical Threshold Calibration

Rather than setting arbitrary heuristic thresholds, `EnergyOODDetector.calibrate_threshold` and `calibrate_threshold_from_logits` compute the empirical energy distribution across holdout validation logits:

$$\theta_{\text{energy}} = \text{Percentile}_{95.0}\Big( \big\{ E(x_j; T) \big\}_{j=1}^{N_{\text{val}}} \Big)$$

Setting the cutoff at the 95th percentile of in-distribution validation occurrences guarantees a fixed 5% false-positive rejection rate on valid in-distribution captures while eliminating out-of-domain noise. During model training via `taxon_vision.models.training.runner`, this calibration runs automatically over validation logits.

---

## Split Conformal Prediction Theory & Coverage Guarantees

While energy scoring detects non-organism noise, ambiguous in-distribution observations (such as cryptic sister species with overlapping phenotypes) require statistical confidence bounds. Heuristic cutoffs (e.g., accepting predictions where softmax probability exceeds 80%) fail catastrophically because deep neural networks are notoriously uncalibrated on long-tailed biological distributions.

TaxonVision adopts **Split Conformal Prediction** (Angelopoulos & Bates, 2021) within `src/taxon_vision/uncertainty/conformal.py` and `scripts/calibrate_conformal.py` to produce prediction sets $C(x)$ with rigorous mathematical guarantees.

### Theoretical Foundations

Let $(X_1, Y_1), \dots, (X_n, Y_n)$ be a calibration dataset $\mathcal{D}_{\text{cal}}$ of independent, exchangeable occurrences drawn from an unknown joint distribution $P_{X, Y}$, and let $(X_{n+1}, Y_{n+1})$ be a new test observation. For any user-specified error significance level $\alpha \in (0, 1)$, split conformal prediction constructs a set-valued predictor $C(X) \subseteq \{1, \dots, K\}$ satisfying:

$$P\big(Y_{n+1} \in C(X_{n+1})\big) \ge 1 - \alpha$$

This coverage guarantee holds unconditionally in finite samples under exchangeability, without requiring parametric assumptions about the data geometry, underlying model architecture, or softmax calibration quality.

### Calibration Procedure (`src/taxon_vision/pipelines/calibrate.py`)

Calibration is executed on the dedicated calibration partition $\mathcal{D}_{\text{cal}}$ (`data/embeddings/cal_embeddings.pt`) that was excluded from backbone feature extraction and classifier head weight tuning:

1. **Non-Conformity Scoring**: For each calibration sample $(X_i, Y_i)$, compute the model probability assigned to the ground-truth taxon $Y_i$. The non-conformity score $s_i$ measures the degree to which the observation disagrees with the ground truth:
   $$s_i = 1 - p(Y_i \mid X_i)$$
   Scores range from $0.0$ (perfect confidence in the true label) to $1.0$ (zero probability assigned to the true label).
2. **Quantile Evaluation**: Compute the empirical $(1 - \alpha)$ quantile threshold $\hat{q}$ adjusted for finite-sample correction:
   $$p_{\text{level}} = \min\left(1.0, \max\left(0.0, \frac{\lceil (n + 1)(1 - \alpha) \rceil}{n}\right)\right)$$
   $$\hat{q} = \text{Quantile}\big(\{s_1, \dots, s_n\}, p_{\text{level}}, \text{method} = \text{"higher"}\big)$$
   Selecting the higher order statistic guarantees that the coverage probability satisfies $P(Y \in C(X)) \ge 1 - \alpha$.
3. **Artifact Serialization**: The resulting calibration metadata ($\alpha$, nominal coverage $1 - \alpha$, calibrated cutoff $\hat{q}$, sample size $n$) and Helmholtz OOD energy threshold $\tau_{\text{energy}}$ are saved to `config/conformal_config.yaml`.

### Online Prediction Set Construction (`ConformalPredictionEngine`)

During real-time inference, `ConformalPredictionEngine.predict_set` evaluates candidate classes against the pre-calibrated cutoff $\hat{q}$:

1. For each candidate taxon $i \in \{1, \dots, K\}$, calculate the non-conformity score $s_i = 1.0 - p_i$.
2. Admit taxon $i$ into the prediction set $C(x)$ if its non-conformity score does not exceed $\hat{q}$:
   $$C(x) = \big\{ i \in \{1, \dots, K\} \;\big|\; s_i \le \hat{q} \big\} \equiv \big\{ i \in \{1, \dots, K\} \;\big|\; p_i \ge 1 - \hat{q} \big\}$$
3. **Computational Efficiency**: Evaluating $s_i \le \hat{q}$ across $K$ classes requires an $O(K)$ vector operation using NumPy vectorization, achieving sub-millisecond execution with zero dynamic heap allocations.

### Referral Criteria

A prediction set satisfying marginal coverage may still be operationally unsafe for automated ingestion if the set is empty, contains too many competing taxa, or has inadequate top confidence. `ConformalPredictionEngine` and `run_prediction` enforce three referral criteria:

1. **Empty Set ($|C(x)| = 0$)**: No cataloged taxon achieved the minimum inclusion probability $p_i \ge 1 - \hat{q}$. This signals severe model uncertainty or a novel organism.
2. **Ambiguity Cardinality Overflow ($|C(x)| > k_{\text{max}}$)**: The set contains more candidate taxa than allowed by configuration (default $k_{\text{max}} = 3$). A large set indicates the model cannot separate competing sister species.
3. **Deficient Top-1 Confidence ($\max_i p_i < \text{min\_confidence\_threshold}$)**: Even if a singleton set is formed, if the top candidate probability falls below the threshold (default $0.60$), the prediction is marked for triage.

If any of these conditions are met, `refer_to_human` is returned as `True`.

---

## Forward-Only Class Activation Mapping (CAM) Attribution

In production ecological services operating under strict Service Level Agreements (SLAs)—such as TaxonVision's target sub-25ms p95 latency budget—diagnostic interpretability must not compromise throughput.

### The Latency Barrier of Grad-CAM

Conventional attribution techniques like Grad-CAM compute gradients of the target class score $y^c$ with respect to spatial feature map activations $A^k$:

$$\alpha_k^c = \frac{1}{Z} \sum_{i} \sum_{j} \frac{\partial y^c}{\partial A_{i, j}^k}$$

This formulation requires a full backward autograd pass, retaining computation graphs, allocating backward activation buffers, and increasing inference latency by 15ms to 30ms. In high-throughput settings, backward passes exceed the entire SLA budget.

### Mathematical Formulation of Forward-Only CAM

TaxonVision's classifier architecture couples a vision backbone (e.g., DINOv3, BioCLIP, MobileNetV4) with a linear projection head following global spatial pooling:

$$y^c = \sum_{k=1}^D w_k^c \left(\frac{1}{H \cdot W} \sum_{x=1}^H \sum_{y=1}^W A_{x, y}^k\right)$$

Because the classification head is purely linear, the spatial attribution map $L_{\text{CAM}}^c$ can be obtained in the forward direction by directly weighting spatial feature maps $A^k$ with the linear head weights $w^c$, bypassing backward differentiation entirely:

$$L_{\text{CAM}}^c = \text{ReLU}\left( \sum_{k=1}^D w_k^c \cdot A^k \right)$$

This forward projection executes in $<3\text{ms}$ while preserving anatomical focus on diagnostic morphological features (such as petal structures, wing venation, or plumage patterns).

### Multi-Backbone Projection Implementations

In `src/taxon_vision/inference/cam.py`, forward CAM supports both convolutional networks (CNNs) and Vision Transformers (ViTs) through runtime protocols:

```
Input Image (H_img, W_img)
       │
       ▼
Backbone forward_features()
       ├─────────────────────────────────────────┐
       ▼ (4D Tensor: B, C, H, W)                 ▼ (3D Tensor: B, N, D)
CNN Spatial Feature Maps                  ViT Patch Tokens
       │                                         │
       │ (Optional conv_head)                    │ Strip prefix [CLS] tokens
       ▼                                         ▼ Unflatten to (B, G, G, D) where G = sqrt(N)
torch.einsum("d, bdxy -> bxy", w_c, feats)    torch.einsum("d, bxy d -> bxy", w_c, patches)
       │                                         │
       └────────────────────┬────────────────────┘
                            ▼
                       ReLU(cam)
                            ▼
           Bilinear Interpolation to (H_img, W_img)
                            ▼
            Min-Max Normalization with Epsilon
                            ▼
              Normalized CAM Heatmap [0.0, 1.0]
```

- **Convolutional Backbones (4D: $B \times C \times H \times W$)**:
  In `_project_cnn_features`, if the backbone exposes a projection head (`CNNBackboneWithConvHead`), features pass through `conv_head(feats)` first. The resulting tensor is contracted with class weight vector $w_c \in \mathbb{R}^D$ via Einstein summation:
  ```python
  cam = torch.einsum("d, bdxy -> bxy", w_c, feats)
  return F.relu(cam)
  ```
- **Vision Transformers (3D: $B \times N \times D$)**:
  In `_project_vit_features`, class/prefix tokens are stripped using `num_prefix_tokens` (default 1). The remaining $N$ patch tokens are reshaped into a 2D spatial grid $G \times G$, verifying that $G = \sqrt{N}$ is an exact square. The spatial patch embeddings are then contracted with $w_c$:
  ```python
  spatial_patches = patches.view(-1, grid_size, grid_size, d)
  cam = torch.einsum("d, bxy d -> bxy", w_c, spatial_patches)
  return F.relu(cam)
  ```

### Normalization and Spatial Rescaling

The raw spatial map is upsampled to original image dimensions $(W, H)$ via bilinear interpolation without corner alignment:

```python
upsampled = F.interpolate(
    cam.unsqueeze(1),
    size=(height, width),
    mode="bilinear",
    align_corners=False,
).squeeze()
```

The resulting attribution is min-max scaled into $[0.0, 1.0]$ using a numerical stability epsilon ($\epsilon_{\text{cam}} = 10^{-8}$, declared in `config/default_config.yaml`):

$$\text{norm} = \frac{\text{cam} - c_{\min}}{c_{\max} - c_{\min} + \epsilon_{\text{cam}}}$$

If the model head lacks an `nn.Linear` layer or the backbone emits non-spatial outputs, the generator raises `SaliencyExtractionError`, which the web service maps to HTTP 422.

---

## Active Learning Query Strategies & Triage Queue

Human taxonomic expertise is an expensive, constrained resource. Random sampling of incoming field data over-indexes on abundant, easily identified organisms (e.g., domestic cats or dandelion weeds) while starving subtle, high-value edge cases. Active learning strategically routes only the most informative observations to expert reviewers.

### Multi-Class Margin Sampling

In `src/taxon_vision/active_learning/query.py`, the `margin_sampling` strategy computes uncertainty scores based on decision boundary contestation. For a predicted probability distribution sorted descending $p_{(1)} \ge p_{(2)} \ge \dots \ge p_{(K)}$:

$$\Delta = p_{(1)} - p_{(2)}$$
$$\text{Uncertainty Score} = 1.0 - \Delta$$

```
Margin Uncertainty Interpretation:
Decisive Identification:   p_top1 = 0.90, p_top2 = 0.05  -->  Margin = 0.85  -->  Uncertainty = 0.15 (Low priority)
Contested Decision:        p_top1 = 0.35, p_top2 = 0.34  -->  Margin = 0.01  -->  Uncertainty = 0.99 (High priority)
```

- An uncertainty score close to $1.0$ indicates that the model is torn between the top two sister taxa ($\Delta \approx 0$). Resolving this classification provides maximal gradient signal when retraining.
- An uncertainty score close to $0.0$ indicates decisive separation ($\Delta \approx 1$).
- Time complexity is $O(N \cdot K \log K)$ across batch size $N$ and class count $K$.

### The Priority `TriageQueue`

Ambiguous observations must be queued and served to taxonomists in strict order of uncertainty. Naive sorting of observation lists incurs $O(N \log N)$ overhead per insertion or $O(N)$ per dequeue.

`TriageQueue` in `src/taxon_vision/active_learning/triage_queue.py` implements a binary max-heap backed by Python's standard `heapq` module:

```python
@dataclass
class TriageItem:
    observation_id: str
    uncertainty_score: float
    conformal_set_size: int
    image_url: str

class TriageQueue:
    def __init__(self) -> None:
        self._heap: list[tuple[float, int, TriageItem]] = []
        self._counter: int = 0

    def push(self, item: TriageItem) -> None:
        heapq.heappush(self._heap, (-item.uncertainty_score, self._counter, item))
        self._counter += 1

    def pop(self) -> TriageItem | None:
        if not self._heap:
            return None
        _, _, item = heapq.heappop(self._heap)
        return item
```

Key engineering decisions in `TriageQueue`:
- **Min-to-Max Heap Conversion**: Python's `heapq` implements a min-heap. Storing the negated uncertainty score (`-item.uncertainty_score`) turns it into an effective max-heap, ensuring that the highest uncertainty item is always at the root.
- **Deterministic Tie-Breaking**: A monotonic sequence counter (`self._counter`) serves as the second tuple element. When two observations have identical uncertainty scores, the heap breaks ties based on arrival order without requiring comparison operators on `TriageItem`.
- **Complexity**: Enqueuing (`push`) and dequeuing (`pop`) run in strict $O(\log N)$ time, while queue inspection (`queue` property) returns an $O(N \log N)$ descending snapshot without mutating heap state.

### Observation Staging & Feature Persistence

When `run_prediction` flags an observation (`is_ood = True` or `requires_human_review = True`), `persist_flagged_observation` in `src/taxon_vision/service/storage.py` writes the record to disk:

1. **Tabular Decision Metadata**: Stored as an individual JSON file (`{observation_id}_meta.json`) containing timestamp, top prediction, conformal set names, OOD energy score, raw probabilities, and review flags.
2. **Dense Visual Embedding**: Serialized as a binary NumPy file (`{observation_id}_embedding.npy`). Caching the embedding ensures downstream active learning loops and vector stores (e.g., LanceDB or MinIO) can consume representations without re-executing heavy vision foundation forward passes.
3. **Audit Log Manifest**: Appends the JSON record to an append-only line-delimited log (`manifest.jsonl`).

---

## Configuration Reference & Invariants

The uncertainty and safety subsystem is governed by declarative schemas defined in `src/taxon_vision/config.py` and instantiated from `config/default_config.yaml`.

| Configuration Key | Schema Model | Default Value | Description |
|---|---|---|---|
| `conformal.alpha` | `ConformalConfig` | `0.05` | Nominal error rate bound ($1 - \alpha = 0.95$ marginal coverage guarantee). |
| `conformal.q_hat` | `ConformalConfig` | `0.85` | Calibrated non-conformity quantile cutoff. |
| `conformal.k_max` | `ConformalConfig` | `3` | Maximum prediction set size before triggering human expert referral. |
| `conformal.min_confidence_threshold` | `ConformalConfig` | `0.60` | Minimum top-1 confidence required to bypass human review. |
| `ood.energy_threshold` | `OODConfig` | `-2.40` | Helmholtz free energy cutoff above which inputs are rejected as OOD. |
| `ood.entropy_threshold` | `OODConfig` | `1.60` | Shannon predictive entropy threshold in nats for uncertainty referral. |
| `model.temperature` | `ModelConfig` | `1.0` | Softmax logit scaling temperature for energy and probability evaluation. |
| `model.cam_numerical_epsilon` | `ModelConfig` | `1.0e-8` | Numerical epsilon preventing division by zero during CAM min-max scaling. |
| `model.flagged_storage_dir` | `ModelConfig` | `data/flagged_observations` | Staging directory for flagged metadata and embeddings. |

### Architectural Invariants & Verification

The uncertainty subsystem is verified by dedicated unit and invariant test suites:

- **Empirical Conformal Coverage Invariant** (`tests/invariants/test_conformal_coverage_invariant.py`):
  Evaluates holdout coverage across Dirichlet probability vectors, confirming that empirical coverage matches or exceeds the theoretical $1 - \alpha$ bound under exchangeability.
- **Conformal Bound Checks** (`tests/unit/test_conformal_bounds.py`):
  Verifies that high-confidence inputs yield compact singleton sets (`pred_set == [0]`), diffuse distributions produce sets exceeding $k_{\text{max}}$ triggering referral (`assert refer`), and singleton sets with confidence below `min_confidence_threshold` (e.g., 48.2% < 60%) assert `refer_to_human`.
- **OOD Energy Detection** (`tests/unit/test_ood_energy.py`):
  Confirms that in-distribution logits produce energy below threshold ($E \approx -3.92 < -2.40$), flat zero logits trigger OOD detection ($E \approx -2.30 > -2.40$), and empirical threshold calibration accurately captures the 95th percentile.
- **Active Learning Margin & Priority Triage** (`tests/unit/test_active_learning.py`):
  Confirms margin sampling assigns higher uncertainty scores to contested boundaries ($p = [0.34, 0.33, 0.33]$ vs $[0.6, 0.3, 0.1]$) and verifies that `TriageQueue.pop` strictly extracts highest-uncertainty items first.
- **CAM Heatmap Bounds & Error Handling** (`tests/unit/test_models_components.py`):
  Validates that CAM heatmaps match image spatial dimensions, stay strictly within $[0.0, 1.0]$, and raise `SaliencyExtractionError` when provided invalid backbones or models lacking linear classification heads.
