---
type: workflow
title: Model Training & Experiment Tracking
description: Frozen backbone feature extraction, stratified embedding caching, linear head probing with Class-Balanced Loss, Bayesian hyperparameter tuning with Optuna, and MLflow experiment tracking with DagsHub.
tags: [model-training, embeddings, representation-caching, class-balanced-loss, linear-probing, mlflow, dagshub, optuna, hyperparameter-tuning, bioclip, dinov3, mobilenetv4]
verified:
  - by: openwiki/0.7.2
    at: 2026-11-08T17:07:25.915Z
sources:
  - id: openwiki-source-b6770f2af9eb66bdd4ad3d8e
    resource: repo://src/taxon_vision/models/factory.py
  - id: openwiki-source-3ff9bfdaaa95aaa3887fe758
    resource: repo://src/taxon_vision/models/head.py
  - id: openwiki-source-eebdf3cec35e0ebefdf79845
    resource: repo://src/taxon_vision/models/loss.py
  - id: openwiki-source-f4dcca38c42ca1d51de7164c
    resource: repo://src/taxon_vision/models/training/checkpoint.py
  - id: openwiki-source-d85184689573a45a9b19fe01
    resource: repo://src/taxon_vision/models/training/dataset.py
  - id: openwiki-source-e6a31393a2c56444d9f61713
    resource: repo://src/taxon_vision/models/training/embeddings.py
  - id: openwiki-source-eda873d85a5772df9776cff5
    resource: repo://src/taxon_vision/models/training/promotion.py
  - id: openwiki-source-2adf091ac1972d98fc4ca4fb
    resource: repo://src/taxon_vision/models/training/runner.py
  - id: openwiki-source-a88159face5e03f82447df3c
    resource: repo://src/taxon_vision/models/training/tracking.py
  - id: openwiki-source-d1f301f07a72e6096d67cc4b
    resource: repo://src/taxon_vision/models/training/tuning.py
  - id: openwiki-source-c05f29404711b017aa575cc1
    resource: repo://src/taxon_vision/models/training/types.py
  - id: openwiki-source-b4ed281978b2367bfb601607
    resource: repo://src/taxon_vision/models/training/visualizations.py
  - id: openwiki-source-db9076027e694f431684d0c0
    resource: repo://src/taxon_vision/pipelines/extract_features.py
generated: { by: "openwiki/0.7.2", at: "2026-11-08T17:07:25.915Z" }
---

# Model Training & Experiment Tracking

Ecological biodiversity classification presents steep machine learning challenges: citizen-science repositories like iNaturalist exhibit severe power-law (Zipfian) species imbalances, while modern vision foundation models (VFMs) such as Vision Transformers contain hundreds of millions of parameters. Full end-to-end backpropagation across deep backbones on every training epoch demands prohibitive GPU compute ($\ge 24\,\text{GB}$ VRAM), risks catastrophic forgetting of pre-trained visual representations, and introduces severe gradient instability on rare singleton classes.

To resolve these tensions, TaxonVision adopts a decoupled **two-stage representation learning and linear probing** paradigm:

1. **Offline Representation Caching**: Dense visual feature representations are extracted once by streaming images through a pre-trained, frozen Vision Foundation Model (BioCLIP-2, DINOv3, DINOv2, MobileNetV4, or EfficientNet) and persisted to disk as an immutable `EmbeddingSplit` artifact containing stratified training, validation, calibration, and test tensors.
2. **Rapid Linear Head Probing & Bayesian Tuning**: A lightweight classification head (`nn.Dropout` followed by `nn.Linear`) is trained directly on cached representation vectors in seconds. Optimization is driven by **Class-Balanced Loss** (Cui et al., CVPR 2019) using effective sample volumes to counter extreme class frequency disparity, while **Optuna** automates Bayesian hyperparameter optimization with early trial pruning.
3. **MLOps Lifecycle Tracking & Multi-Gate Promotion**: Training executions are wrapped in RAII context scopes that stream metrics, diagnostic plots (confusion matrices, PR curves, and Helmholtz free energy distributions), and DVC dataset manifest hashes to **MLflow** hosted on **DagsHub**, conditionally promoting candidates across a rigorous Four-Gate Promotion Architecture governing Model Registry aliases (`@Production`, `@Challenger`, `@Archived`).

For broader architectural and pipeline context, see [System Architecture & Design](../architecture/system-overview.md), [Data Ingestion & Dataset Versioning](data-ingestion-and-versioning.md), [Inference Optimization & ONNX Runtime](inference-and-optimization.md), [Uncertainty Quantification & Epistemic Safety](../concepts/uncertainty-and-safety.md), [Domain Entities & Taxonomy Perimeter](../architecture/domain-and-taxa.md), and [Test Strategy & Invariants](../testing/test-strategy-and-invariants.md).

---

## End-to-End Training & Tracking Workflow

The following architecture diagram traces the complete lifecycle from Parquet dataset manifests and remote image streaming through frozen backbone extraction, stratified embedding serialization, linear head optimization, and MLflow/DagsHub experiment governance:

<!-- openwiki: mermaid parse failed and this diagram was converted to a text fence so it does not break rendering. Fix the diagram source and restore the mermaid fence. Parser error: Heuristic: an unescaped angle bracket inside a label breaks rendering; rephrase the label. -->
```text
flowchart TD
    subgraph Data["1. Dataset Manifest and Streaming"]
        PARQUET["Parquet Dataset Manifest<br/>data/manifests/dataset_manifest.parquet"] --> STREAMER["S3ImageStreamer<br/>(Lazy S3 Open Data Image Stream)"]
        CATALOG["Taxa Catalog SSOT<br/>config/taxa_catalog.yaml"] --> MAP["build_taxon_index_map()<br/>(Maps iNat Taxon ID to dense index)"]
        STREAMER --> BATCH_ACC["Mini-Batch Accumulation<br/>(batch_size=32)"]
    end

    subgraph Extraction["2. Frozen Representation Extraction (pipelines/extract_features.py)"]
        VFM_FACTORY["create_feature_extractor()<br/>(BioCLIP-2, DINOv3, MobileNetV4)"] --> FROZEN_MODEL["Frozen VFM Backbone<br/>(requires_grad=False, eval mode)"]
        TRANSFORM["get_evaluation_transform()<br/>(Resize 256, CenterCrop 224, Normalize)"] --> BATCH_ACC
        BATCH_ACC --> EXTRACT_BATCH["extract_batch_representations()<br/>(Zero-grad forward pass)"]
        FROZEN_MODEL --> EXTRACT_BATCH
        EXTRACT_BATCH --> FULL_TENSORS["Unified Tensors<br/>Embeddings (N, D) and Labels (N,)"]
        FULL_TENSORS --> STRATIFY["create_stratified_embedding_split()<br/>(Stratified 70/10/10/10 4-way split)"]
        STRATIFY --> CACHE_FILE["4 Stratified Embedding Tensors<br/>data/embeddings/*.pt"]
    end

    subgraph Optimization["3. Linear Probing and Tuning (models/training/)"]
        CACHE_FILE --> LOAD_SPLIT["load_or_create_training_split()<br/>(train_embeddings.pt and val_embeddings.pt)"]
        LOAD_SPLIT --> TUNE_CHECK{"tune=True ?"}
        
        TUNE_CHECK -- "Yes" --> OPTUNA["tune_hyperparameters()<br/>(TPESampler and ASHA Pruner)"]
        OPTUNA --> BEST_HP["Sampled Optimal Hyperparameters<br/>(lr, weight_decay, dropout, beta)"]
        TUNE_CHECK -- "No" --> DEF_HP["Default Hyperparameters<br/>(lr=1e-3, beta=0.9999, dropout=0.2)"]
        
        BEST_HP --> BUILD_HEAD["build_training_components()<br/>TaxonClassifier Head: dim to num_classes"]
        DEF_HP --> BUILD_HEAD
        
        BUILD_HEAD --> CB_LOSS["ClassBalancedLoss(beta)<br/>E(N) = (1 - beta^N)/(1 - beta)"]
        BUILD_HEAD --> ADAMW["AdamW Optimizer"]
        
        CB_LOSS --> TRAIN_LOOP["train_head_on_cached_embeddings()<br/>(Epochs, LR Plateau, Early Stopping)"]
        ADAMW --> TRAIN_LOOP
        LOAD_SPLIT --> TRAIN_LOOP
    end

    subgraph Registry["4. Experiment Tracking and Registration (models/training/)"]
        TRAIN_LOOP --> CHECKPOINT["save_checkpoint()<br/>(models/checkpoints/head.pt)"]
        CHECKPOINT --> CLEAR_CACHE["get_classifier.cache_clear()<br/>(Runtime serving cache invalidation)"]
        
        TRAIN_LOOP --> SCOPE["mlflow_run_scope (RAII Context)<br/>(DagsHub MLflow Server)"]
        SCOPE --> DVC_HASH["get_dvc_dataset_hash()<br/>(Links dvc.lock manifest MD5)"]
        SCOPE --> LOG_METRICS["_log_mlflow_metrics()<br/>(train_loss, val_pr_auc, val_f1)"]
        SCOPE --> LOG_PLOTS["log_training_visualizations()<br/>(Confusion Matrix, PR Curves, Free Energy)"]
        SCOPE --> LOG_MODEL["MLflowTracker.log_model()<br/>(Logs candidate model artifact and signature)"]
    end
```
*Figure 1: End-to-end representation extraction, linear head optimization, and MLflow experiment governance.*

---

## Vision Foundation Model (VFM) Backbone Factory

The backbone factory in `src/taxon_vision/models/factory.py` manages foundation model instantiation, parameter freezing, and dimensional inspection via `timm` (PyTorch Image Models). The factory decouples downstream classification heads from specific backbone architectures, allowing researchers and operators to seamlessly interchange lightweight mobile CNNs and state-of-the-art vision transformers.

### Supported Architectures

The centralized application configuration (`src/taxon_vision/config.py` and `config/default_config.yaml`) defines the authoritative backbone registry:

| Canonical Identifier | Underlying `timm` / Hugging Face Identifier | Architectural Family | Parameter Count | Embedding Dim ($D$) | Primary Use Case |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `bioclip-2` | `hf-hub:imageomics/bioclip-2` | Vision Transformer (ViT) | ~86M | 768 | Domain-specific biological foundation model trained on TreeOfLife-10M. |
| `dinov3` | `vit_small_patch16_dinov3` | Vision Transformer (ViT) | ~22M | 384 | Self-supervised foundation model producing fine-grained visual representations. |
| `dinov3_vits14` | `vit_small_patch16_dinov3` | Vision Transformer (ViT) | ~22M | 384 | Alias mapping for patch-14/16 ViT configurations. |
| `dinov2_vits14` | `vit_small_patch14_dinov2.lvd142m` | Vision Transformer (ViT) | ~22M | 384 | Robust high-resolution self-supervised visual feature extractor. |
| `mobilenetv4_conv_small` | `mobilenetv4_conv_small.e2400_r224_in1k` | Convolutional Network (CNN) | ~3.8M | 1280 | Ultra-lightweight edge architecture; default fallback for offline tests and CI. |
| `efficientnet_b0` | `efficientnet_b0.ra_in1k` | Convolutional Network (CNN) | ~5.3M | 1280 | Standard convolutional baseline for constrained compute budgets. |

### Instantiation & Invariant Freezing

Backbones are created via `create_feature_extractor(model_name, pretrained, config, allow_fallback)` (`src/taxon_vision/models/factory.py`). The factory enforces three fundamental invariants:

1. **Head Removal**: The model is created with `num_classes=0`, which strips the pre-training classification layer and pools output activations into a flat $D$-dimensional feature embedding.
2. **Gradient Locking**: Parameter gradients are frozen across the entire network by setting `param.requires_grad = False` for all `model.parameters()`. This guarantees that no backward gradients propagate into the backbone during linear probing.
3. **Evaluation Mode**: The backbone is set to `model.eval()`, disabling stochastic dropout layers and locking batch normalization running statistics.

```python
# src/taxon_vision/models/factory.py
def create_feature_extractor(
    model_name: str | None = None,
    pretrained: bool = False,
    config: ModelConfig | None = None,
    allow_fallback: bool = False,
) -> nn.Module:
    model_cfg = config or get_settings().model
    target_name = model_name or model_cfg.default_extractor
    resolved_name = model_cfg.backbone_registry.get(target_name, target_name)

    try:
        model: nn.Module = timm.create_model(resolved_name, pretrained=pretrained, num_classes=0)
    except Exception as exc:
        if not allow_fallback:
            raise RuntimeError(f"Could not create backbone '{resolved_name}' (pretrained={pretrained}): {exc}") from exc
        # Explicit opt-in: lightweight un-pretrained model for offline CI and unit tests
        fallback_name = model_cfg.fallback_extractor
        model = timm.create_model(fallback_name, pretrained=False, num_classes=0)

    for param in model.parameters():
        param.requires_grad = False

    model.eval()
    return model
```

### Deterministic Dimension Resolution

Rather than hardcoding projection widths across CNN and ViT variants, `get_feature_dimension()` deterministically determines the feature dimension $D$ by executing a single forward pass with a zero dummy tensor `(1, channels, image_size, image_size)` inside `torch.inference_mode()`:

```python
# src/taxon_vision/models/factory.py
def get_feature_dimension(
    model: nn.Module,
    image_size: int | None = None,
    channels: int | None = None,
    config: ModelConfig | None = None,
) -> int:
    model_cfg = config or get_settings().model
    size = image_size if image_size is not None else model_cfg.image_size
    num_channels = channels if channels is not None else model_cfg.channels

    device = next(model.parameters()).device
    dummy_input = torch.zeros(1, num_channels, size, size, device=device)
    with torch.inference_mode():
        features = model(dummy_input)
    return int(features.shape[-1])
```

---

## Representation Caching & Stratified Splitting

Repeatedly passing high-resolution biological images through deep vision transformers during iterative classifier training wastes massive amounts of compute and energy. TaxonVision isolates feature extraction into an offline preprocessing pipeline (`src/taxon_vision/pipelines/extract_features.py`) that extracts dense embeddings once and serializes them into compact binary cache artifacts.

### Single Source of Truth (SSOT) Taxonomic Indexing

Citizen-science repositories assign arbitrary, non-contiguous integer IDs to taxa (e.g., Monarch Butterfly *Danaus plexippus* is `48662`, Western Honey Bee *Apis mellifera* is `47219`). Classification heads require contiguous dense labels $y \in [0, K-1]$.

The function `build_taxon_index_map()` dynamically parses `config/taxa_catalog.yaml`, mapping primary taxon IDs (`inat_taxon_id`, `id`) and synonyms (`inat_aliases`) to the dense ordinal class index $[0, K-1]$. This guarantees strict consistency across data ingestion, feature extraction, linear head training, and inference serving.

### Evaluation Preprocessing Pipeline

During feature caching, images must follow the upstream backbone evaluation protocol without stochastic data augmentation (such as random affine transformations or color jittering), ensuring that representations reflect canonical visual features. The function `get_evaluation_transform()` constructs the deterministic evaluation transformation:

1. **Aspect-Preserving Resize**: Resizes the shorter image edge to `resize_size` (default `round(image_size * eval_resize_ratio) = 256` for a 224px target) via bilinear interpolation.
2. **Center Crop**: Extracts a deterministic square crop of dimensions `(image_size, image_size)` (224 × 224).
3. **Tensor Conversion**: Converts uint8 PIL RGB images to float32 tensors scaled to $[0.0, 1.0]$.
4. **Channel Normalization**: Normalizes tensor channels using standard ImageNet distribution parameters ($\mu = [0.485, 0.456, 0.406]$, $\sigma = [0.229, 0.224, 0.225]$).

### Lazy S3 Streaming & Batch Collation

The function `stream_and_extract_representations()` (`src/taxon_vision/pipelines/extract_features.py`) pairs `S3ImageStreamer` with `extract_batch_representations()`:

- Observations are streamed on demand from public AWS Open Data S3 buckets via HTTP connection pools, preventing local disk bloat.
- Preprocessed tensors accumulate into batches of size `batch_size` (default 32).
- `extract_batch_representations()` stacks tensors into a contiguous 4D tensor `(B, 3, H, W)`, executes `extract_features(backbone, batch, device=device)` inside `torch.no_grad()`, and moves representations `(B, D)` directly to CPU host memory.

### Stratified Splitting & Invariant Enforcement

Because rare species may contain few observations, standard random splitting risks leaving validation sets with zero occurrences of endangered taxa, collapsing macro metrics.

The function `create_stratified_embedding_split()` executes multi-stage stratified partitioning via `_partition_stratified_subsets()` to preserve exact class proportions across partitions. It also computes empirical class sample counts:

$$N_c = \max\left(1, \sum_{i=1}^{N_{\text{train}}} \mathbb{I}(y_i = c)\right) \quad \text{for } c \in [0, K-1]$$

The resulting dataset is encapsulated within an immutable `EmbeddingSplit` container (`src/taxon_vision/models/training/types.py`):

```python
# src/taxon_vision/models/training/types.py
@dataclass(frozen=True)
class EmbeddingSplit:
    train_embeddings: torch.Tensor
    train_labels: torch.Tensor
    val_embeddings: torch.Tensor
    val_labels: torch.Tensor
    cal_embeddings: torch.Tensor | None = None
    cal_labels: torch.Tensor | None = None
    test_embeddings: torch.Tensor | None = None
    test_labels: torch.Tensor | None = None
    samples_per_class: Sequence[int] | None = None

    def __post_init__(self) -> None:
        if self.train_embeddings.ndim != 2 or self.val_embeddings.ndim != 2:
            raise ValueError(f"Embeddings must be 2D tensors, got train={self.train_embeddings.shape} and val={self.val_embeddings.shape}.")
        if self.train_embeddings.shape[0] != self.train_labels.shape[0]:
            raise ValueError(f"Train embeddings count ({self.train_embeddings.shape[0]}) does not match train labels count ({self.train_labels.shape[0]}).")
        if self.val_embeddings.shape[0] != self.val_labels.shape[0]:
            raise ValueError(f"Val embeddings count ({self.val_embeddings.shape[0]}) does not match val labels count ({self.val_labels.shape[0]}).")
        if self.train_embeddings.shape[1] != self.val_embeddings.shape[1]:
            raise ValueError(f"Feature dimension mismatch: train is {self.train_embeddings.shape[1]}, val is {self.val_embeddings.shape[1]}.")
```

### Artifact Serialization

The complete 4-way stratified partition is serialized via `torch.save()` into four discrete physical files under `data/embeddings/`:

* `train_embeddings.pt` (70%): Used for classification head training with Class-Balanced Loss.
* `val_embeddings.pt` (10%): Used for validation PR-AUC evaluation, early stopping, and Optuna pruning.
* `cal_embeddings.pt` (10%): Reserved exclusively for post-quantization Conformal Prediction quantile calibration and Helmholtz OOD energy threshold estimation.
* `test_embeddings.pt` (10%): Strictly held out for empirical coverage verification and Pareto benchmarking.

Each file stores a tensor dictionary containing `embeddings`, `labels`, `indices`, and `class_counts`. During training, `load_or_create_training_split()` loads `train_embeddings.pt` and `val_embeddings.pt`. If missing and synthetic mode is authorized (offline CI smoke tests), it falls back to deterministic synthetic Gaussian tensors generated by `_generate_synthetic_embedding_split()`.

---

## TaxonClassifier Head Architecture

The classification head (`src/taxon_vision/models/head.py`) wraps the frozen vision backbone and exposes dedicated execution pathways for both training on cached embeddings and full end-to-end inference serving.

```python
# src/taxon_vision/models/head.py
class TaxonClassifier(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        feature_dim: int,
        num_classes: int,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.feature_dim = feature_dim
        self.num_classes = num_classes

        for param in self.backbone.parameters():
            param.requires_grad = False

        self.head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(feature_dim, num_classes),
        )
```

### Execution Pathways

The module provides four distinct forward execution pathways:

1. `forward_features(x)`: Evaluates input image tensor $x \in \mathbb{R}^{B \times 3 \times H \times W}$ through the frozen backbone under `torch.no_grad()` to return pooled feature representations $z \in \mathbb{R}^{B \times D}$.
2. `forward_head(features)`: Projects cached feature vectors $z \in \mathbb{R}^{B \times D}$ through `self.head` to yield raw taxonomic class logits $l \in \mathbb{R}^{B \times K}$. This pathway is utilized during cached head training, running thousands of gradient updates in seconds without backbone computation.
3. `forward(x)`: End-to-end evaluation executing `forward_head(forward_features(x))`.
4. `forward_with_embeddings(x)`: Unified forward pass returning the named tuple `ClassifierOutputs(logits, embeddings)`. This eliminates redundant backbone passes for downstream consumers (conformal prediction sets, energy-based OOD scoring, forward-only Class Activation Mapping, and active learning queues).

### Checkpoint Restoration

The utility `load_head_checkpoint(head, checkpoint_file)` restores trained head parameters onto CPU host memory with `weights_only=True` and `strict=True`. It handles state dictionary containers (`state_dict` or `model_state_dict`), strips any `head.` namespace prefixes produced when saving full `TaxonClassifier` instances, and enforces that tensor shapes match the linear head dimensions exactly.

---

## Loss Formulations for Severe Taxonomic Imbalance

Biological observation frequencies follow an extreme power-law (Zipfian) distribution: a handful of common species (e.g., Western Honey Bee or Mallard Duck) comprise tens of thousands of sightings, while vulnerable or cryptic taxa may possess fewer than 10 observations.

Standard Cross-Entropy Loss evaluates uniform per-sample gradients:

$$\mathcal{L}_{\text{CE}}(z, y) = - \log \left( \frac{\exp(z_y)}{\sum_{k=1}^K \exp(z_k)} \right)$$

Under extreme imbalance, frequent classes dominate the loss gradients, causing the model to collapse minority class recall.

### Class-Balanced Loss vs. Focal Loss & Naive Inverse Weighting

Three distinct strategies exist in the literature to counteract class imbalance:

1. **Naive Inverse Frequency Weighting ($w_i \propto 1 / N_i$)**: Assigns weights inversely proportional to raw sample counts. While mathematically simple, singleton and rare classes receive astronomical weights ($1000\times$ common classes), creating extreme gradient variance, exploding step sizes, and optimization instability in early epochs.
2. **Focal Loss (Lin et al., ICCV 2017)**: Downweights easy well-classified examples dynamically using modulating factor $(1 - p_t)^\gamma$. While effective in dense object detection where background anchors outnumber objects, Focal Loss focuses on hardness rather than class volume, leaving rare classes vulnerable if high-confidence false negatives occur.
3. **Class-Balanced Loss (Cui et al., CVPR 2019)**: Formulates loss weighting based on the **effective number of samples** $E_n$, capturing data overlap in feature space. As more observations are sampled for a class, new samples increasingly occupy feature space already covered by previous instances. TaxonVision adopts Class-Balanced Cross-Entropy Loss in `src/taxon_vision/models/loss.py`.

### Mathematical Formulation of Effective Sample Volume

The effective number of samples $E_n$ for a class with $n$ observations is defined as:

$$E_n = \frac{1 - \beta^n}{1 - \beta}$$

where $\beta \in [0, 1)$ is a hyperparameter capturing representation volume overlap:

- As $\beta \to 0$, $E_n \to 1$: all classes have equal effective volume, collapsing the loss to standard unweighted Cross-Entropy.
- As $\beta \to 1$, $E_n \to n$: effective volume equals raw count, approaching inverse frequency weighting.
- In practice, $\beta \in [0.99, 0.9999]$ smooths the gradient scale smoothly across the long tail.

The corresponding class weight $W_i$ is inversely proportional to its effective volume:

$$W_i = \frac{1 - \beta}{1 - \beta^{N_i} + \epsilon}$$

where $\epsilon = 10^{-8}$ prevents division by zero. Weights are normalized such that their sum equals the total class count $K$ (mean weight is 1.0):

$$\widetilde{W}_i = W_i \cdot \frac{K}{\sum_{j=1}^K W_j}$$

The Class-Balanced Cross-Entropy Loss is evaluated as:

$$\mathcal{L}_{\text{CB}}(z, y) = - \widetilde{W}_y \log \left( \frac{\exp(z_y)}{\sum_{k=1}^K \exp(z_k)} \right)$$

```python
# src/taxon_vision/models/loss.py
class ClassBalancedLoss(nn.Module):
    def __init__(self, samples_per_class: Sequence[int], beta: float = 0.999, epsilon: float = 1e-8) -> None:
        super().__init__()
        if not samples_per_class:
            raise ValueError("samples_per_class cannot be empty.")
        if not (0.0 <= beta < 1.0):
            raise ValueError(f"beta must be in [0.0, 1.0), got {beta}.")

        self.beta = beta
        num_classes = len(samples_per_class)
        counts = np.maximum(np.array(samples_per_class, dtype=np.float64), 1.0)

        effective_num = 1.0 - np.power(self.beta, counts)
        weights = (1.0 - self.beta) / (effective_num + epsilon)

        norm_factor = np.sum(weights)
        if norm_factor > 0:
            weights = (weights / norm_factor) * num_classes

        self.register_buffer("weights", torch.tensor(weights, dtype=torch.float32))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        device_weights = self.weights.to(logits.device)
        return F.cross_entropy(logits, targets, weight=device_weights)
```

By registering normalized weights as a non-persistent buffer (`self.register_buffer("weights", ...)`), weights automatically migrate across accelerator devices with the module without being tracked as trainable parameters.

---

## Training Loops & Execution Lifecycles

TaxonVision supports two training execution paradigms: rapid cached embedding optimization (`src/taxon_vision/models/training/embeddings.py`) and standard dataloader batch iteration (`src/taxon_vision/models/training/loops.py`).

### Cached Embedding Training Loop (`train_head_on_cached_embeddings`)

When pre-computed embedding tensors are available, `train_head_on_cached_embeddings()` optimizes the linear head directly on GPU device memory without data loader bottlenecks:

1. **Random Permutation**: A pseudo-random permutation over training indices is generated via `torch.randperm(num_samples, device=device)`.
2. **Mini-Batch Slicing**: Slices of pre-computed embeddings and labels are extracted without dynamic heap allocations.
3. **Manifold Noise Regularization**: When `noise_std > 0.0`, zero-mean Gaussian perturbation is injected directly into feature vectors:
   $$z_{\text{perturbed}} = z + \mathcal{N}(0, \sigma^2 I)$$
   This smooths linear decision boundaries across sparse embedding clusters.
4. **Optimization Step**: The linear head computes class logits, evaluates `ClassBalancedLoss`, calls `loss.backward()`, and steps the AdamW optimizer.
5. **Validation & Pruning**: Evaluates validation loss, accuracy, Macro F1, and Macro PR-AUC, reporting to Optuna pruner callbacks.

### Standard Image Dataloader Loops (`train_head_epoch` & `evaluate_head`)

For end-to-end evaluation and legacy pipelines, `src/taxon_vision/models/training/loops.py` provides batch-level training and evaluation loops:

- `train_head_epoch(model, dataloader, optimizer, criterion)`: Enforces `model.backbone.eval()` and `model.train()`, iterating raw image batches, zeroing gradients, and optimizing `model.head` while backbone parameters remain frozen (`requires_grad=False`).
- `evaluate_head(model, dataloader, criterion)`: Executes a zero-gradient validation pass inside `torch.no_grad()`, returning the `HeadEvaluationResult(loss, accuracy)` named tuple.

### Long-Tail Evaluation Metrics

In biological taxonomies with power-law distributions, standard top-1 classification accuracy is deceptive: a naive model predicting only the top 10% most common species can achieve 90% top-1 accuracy while failing on 90% of all taxa.

The evaluation routine computes four complementary validation metrics:

| Metric | Computation | Importance for Long-Tail Evaluation |
| :--- | :--- | :--- |
| **Validation Loss** | $\mathcal{L}_{\text{CB}}(z_{\text{val}}, y_{\text{val}})$ | Differentiable proxy for convergence and overfitting. |
| **Top-1 Accuracy** | $\frac{1}{N_{\text{val}}} \sum_{i=1}^{N_{\text{val}}} \mathbb{I}(\hat{y}_i = y_i)$ | Overall operational accuracy across all test observations. |
| **Macro F1-Score** | $\frac{1}{K} \sum_{k=1}^K F1_k$ | Unweighted average of per-class harmonic precision-recall means, treating endangered taxa identically to dominant species. |
| **Macro PR-AUC** | $\frac{1}{|\mathcal{C}_{\text{present}}|} \sum_{c \in \mathcal{C}_{\text{present}}} \text{AP}_c$ | Area Under the Precision-Recall Curve (Average Precision) averaged across all represented classes. Primary ranking metric for model promotion and early stopping. |

### Learning Rate Scheduling & Early Stopping

The training loop coordinates adaptive learning rate reduction and early termination:

- **ReduceLROnPlateau**: If `reduce_lr_patience` is configured (default 2), learning rate decays by a factor of 0.5 whenever validation PR-AUC fails to achieve a new maximum.
- **Patience-Based Early Stopping**: If `early_stopping_patience` is configured (default 4), training terminates if validation PR-AUC fails to improve over the peak score for $P$ consecutive epochs.
- **Weight Restoration**: When `restore_best_weights=True`, the engine maintains a deep copy of the parameter state dictionary that achieved the peak validation PR-AUC, automatically restoring these weights upon loop exit.

---

## Automated Bayesian Hyperparameter Tuning

Rather than relying on manual trial-and-error, `src/taxon_vision/models/training/tuning.py` integrates **Optuna** to perform Bayesian hyperparameter optimization over cached representations.

### Search Space & Sampling

The `HyperparameterObjective` class searches candidate parameters across the following distributions:

```python
# src/taxon_vision/models/training/tuning.py
def _sample_hyperparameters(self, trial: optuna.Trial) -> dict[str, Any]:
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-1, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
        "dropout": trial.suggest_float("dropout", 0.0, 0.5),
        "cb_beta": trial.suggest_float("cb_beta", 0.99, 0.9999),
        "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
        "noise_std": trial.suggest_float("noise_std", 0.0, 0.05),
    }
```

### Tree-structured Parzen Estimator (TPE) & ASHA Pruning

Tuning uses a combination of Bayesian modeling and asynchronous pruning:

- **Tree-structured Parzen Estimator (TPE)**: Models the probability distributions of hyperparameters $p(x | y)$ for trials above and below a quantile threshold, focusing search effort on high-density performance regions.
- **Asynchronous Successive Halving Algorithm (ASHA)**: Enforced via `optuna.pruners.SuccessiveHalvingPruner(min_resource=2, reduction_factor=2)`. At each epoch, the pruner callback `trial.report(val_pr_auc, step=epoch)` checks trial trajectory against historical cohorts, terminating underperforming trials after 2 epochs to preserve GPU compute.
- **Persistent Study Storage**: Configured via `_resolve_optuna_storage()` supporting SQLite (`sqlite:///data/optuna_studies.db`). SQLite connections execute `PRAGMA journal_mode=WAL;` and `PRAGMA synchronous=NORMAL;` to support concurrent trial execution without database locking. Using `load_if_exists=True` enables warm-starting across executions.

### Artifact & Parquet Logging

Upon study completion:
- Optimal hyperparameters are saved to `models/checkpoints/best_params.yaml` keyed by backbone identifier.
- The complete trial history dataframe is exported to `reports/tuning_trials.parquet`.
- Diagnostic visualizations (`visualizations/optuna_optimization_history.png` and `visualizations/optuna_param_importances.png`) are logged to MLflow via `log_optuna_diagnostic_plots()`.

---

## MLflow & DagsHub Tracking Integration

TaxonVision integrates with MLflow tracking hosted on **DagsHub** (`https://dagshub.com/foersben/taxon-vision-mlops.mlflow`) to maintain cryptographic data lineage, model artifact registration, and automated governance.

### RAII Lifecycle Management

To prevent orphaned runs in `RUNNING` status following unhandled exceptions or CUDA out-of-memory errors, `src/taxon_vision/models/training/tracking.py` defines the `mlflow_run_scope` context manager:

```python
# src/taxon_vision/models/training/tracking.py
@contextmanager
def mlflow_run_scope(
    run_name: str,
    params: dict[str, Any] | None = None,
    *,
    log_file: Path | str | None = None,
    nested: bool = False,
    experiment_name: str | None = None,
) -> Generator[mlflow.ActiveRun | None, None, None]:
    active_run = None
    try:
        settings = get_settings()
        tracking_uri = _resolve_reachable_tracking_uri(settings)
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)

        target_exp = experiment_name or settings.mlflow.experiment_name
        if target_exp:
            mlflow.set_experiment(target_exp)

        active_run = mlflow.start_run(run_name=run_name, nested=True) if nested else mlflow.start_run(run_name=run_name)
        if params:
            mlflow.log_params(params)

        dataset_hash = get_dvc_dataset_hash()
        if dataset_hash:
            mlflow.log_param("dvc_dataset_hash", dataset_hash)

        _inject_lineage_tags()
    except Exception as err:
        logger.debug("MLflow tracking setup skipped or unavailable: %s", err)
        active_run = None

    try:
        yield active_run
    except Exception:
        if active_run:
            _end_mlflow_run_safely("FAILED")
        raise
    else:
        if active_run:
            _attach_mlflow_artifact_safely(log_file)
            _end_mlflow_run_safely("FINISHED")
```

If the remote MLflow tracking server is unreachable, `_resolve_reachable_tracking_uri()` logs a debug message and yields `None`, allowing offline local training to proceed without interruption.

### Cross-Stage Parent Run Resumption

Downstream DVC stages (such as ONNX export, conformal calibration, and holdout evaluation) must attach metrics and artifacts to the original training run rather than spawning disconnected experiments. The `resume_parent_mlflow_run()` context manager reads `reports/train_metrics.json`, parses `ParentRunContext(run_id, model_id, extractor, status)`, and re-enters the active parent run context.

### Cryptographic DVC Dataset Hash Linking

To guarantee auditability, every model trained in TaxonVision is linked to the immutable dataset snapshot that produced it.

The function `get_dvc_dataset_hash()` (`src/taxon_vision/models/training/dataset.py`) parses `dvc.lock`, extracts the MD5 output checksum of the `ingestion` stage associated with `data/manifests`, and logs it as `dvc_dataset_hash` in MLflow parameters and Model Registry version tags.

### Diagnostic Visualizations

At the conclusion of training, `log_training_visualizations()` (`src/taxon_vision/models/training/visualizations.py`) renders and uploads three diagnostic artifacts to MLflow:

1. **Normalized Confusion Matrix (`visualizations/confusion_matrix.png`)**: A class-normalized matrix annotated with integer counts, exposing specific pairwise taxonomic classification errors.
2. **Per-Class Precision-Recall Curves (`visualizations/precision_recall_curves.png`)**: Multiclass precision-recall curves displaying individual Average Precision (AP) scores per taxon, highlighting recall trade-offs across rare species.
3. **Helmholtz Free Energy Distribution (`visualizations/energy_distribution.png`)**: A histogram of validation sample free energies $E(x; T) = -T \log \sum_{k=1}^K \exp(z_k / T)$, annotated with the calibrated 95th-percentile Out-of-Distribution boundary threshold (`EnergyOODDetector.calibrate_threshold_from_logits()`). This threshold is also logged as the scalar metric `calibrated_ood_energy_threshold`.

---

## Four-Gate Model Promotion Architecture

Rather than promoting models immediately upon training completion using isolated validation metrics, TaxonVision deploys a two-stage governance architecture:

1. **Stage 4 (Training Pipeline)**: In `src/taxon_vision/models/training/runner.py`, candidate models are registered in the MLflow Model Registry under `taxon_vision_classifier` and tagged as `candidate_trained`. If legacy immediate promotion is invoked via `_promote_model_if_better()`, the model is tagged as `Production`, `Challenger`, or `Archived` based strictly on validation Macro PR-AUC.
2. **Stage 7 (Holdout Evaluation)**: In `src/taxon_vision/pipelines/evaluate.py`, candidate models undergo comprehensive holdout evaluation across four rigorous, sequential gates via `evaluate_promotion_gates()` (`src/taxon_vision/models/training/promotion.py`).

<!-- openwiki: mermaid parse failed and this diagram was converted to a text fence so it does not break rendering. Fix the diagram source and restore the mermaid fence. Parser error: Heuristic: a semicolon inside a label breaks rendering; rephrase the label. -->
```text
flowchart TD
    START["Candidate Model and Holdout Evaluation Payload"] --> G1{"Gate 1: Predictive Viability<br/>PR-AUC matches or beats incumbent?<br/>FP32-to-INT8 drop within tolerance?"}
    
    G1 -- "Fail" --> ARCH1["Verdict: Archived<br/>gate=below_incumbent or<br/>gate=quantization_drop_exceeded"]
    G1 -- "Pass" --> G2{"Gate 2: Conformal Safety<br/>Empirical coverage meets<br/>nominal 1 - alpha minus tolerance?"}
    
    G2 -- "Fail" --> ARCH2["Verdict: Archived<br/>gate=conformal_coverage_violation"]
    G2 -- "Pass" --> G3{"Gate 3: Physical Latency Budget<br/>Quantized p95 latency satisfies<br/>SLO execution ceiling?"}
    
    G3 -- "Fail" --> ARCH3["Verdict: Archived<br/>gate=latency_slo_violation"]
    G3 -- "Pass" --> G4{"Gate 4: Distribution Exchangeability<br/>Incumbent exists and live samples sufficient?<br/>Two-sample RBF MMD p-value above drift_alpha?"}
    
    G4 -- "No incumbent exists" --> BOOT["Verdict: Production<br/>gate=initial_production<br/>(Bootstrap initial serving baseline)"]
    G4 -- "Insufficient live samples" --> CHAL1["Verdict: Challenger<br/>gate=insufficient_traffic<br/>(Staged for shadow evaluation)"]
    G4 -- "Drift detected (p below drift_alpha)" --> CHAL2["Verdict: Challenger<br/>gate=calibration_not_exchangeable<br/>(Staged; alert drift investigation)"]
    G4 -- "All 4 gates passed" --> PROD["Verdict: Production<br/>gate=promoted_all_4_gates<br/>(Previous Production demoted to Challenger)"]
```
*Figure 2: Four-Gate Model Promotion Architecture evaluating predictive viability, conformal safety, latency SLOs, and exchangeability.*

### Promotion Gate Definitions & Invariants

The `PromotionCriteria` dataclass and `evaluate_promotion_gates()` enforce the following rules:

1. **Gate 1 (Predictive Viability & Quantization Fidelity)**:
   - Macro PR-AUC must match or exceed the incumbent score: $\text{PR-AUC}_{\text{candidate}} \ge \text{PR-AUC}_{\text{incumbent}}$.
   - Post-training dynamic INT8 quantization drop must not exceed tolerance: $\Delta_{\text{FP32} \to \text{INT8}} \le \text{quantization\_drop\_tol}$ (default 0.03 / 3%).
   - *Failure Action*: Verdict is `PromotionAlias.ARCHIVED` with gate `below_incumbent` or `quantization_drop_exceeded`.
2. **Gate 2 (Conformal Coverage Guarantee)**:
   - Empirical holdout coverage must meet nominal coverage ($1 - \alpha$) minus finite-sample variation margin:
     $$\text{Coverage}_{\text{empirical}} \ge (1 - \alpha) - \text{coverage\_tolerance} \quad (\text{default } 0.90 - 0.05 = 0.85)$$
   - *Failure Action*: Verdict is `PromotionAlias.ARCHIVED` with gate `conformal_coverage_violation`.
3. **Gate 3 (Physical Latency Budget)**:
   - Quantized 95th-percentile inference latency must satisfy execution budget: $p_{95} \le \text{latency\_slo\_ms}$ (default 120.0 ms on CPU).
   - *Failure Action*: Verdict is `PromotionAlias.ARCHIVED` with gate `latency_slo_violation`.
4. **Gate 4 (Exchangeability & Staging)**:
   - If no incumbent baseline exists in the registry, bootstrap initial `@Production` immediately (`gate="initial_production"`).
   - If live telemetry observations are fewer than `min_telemetry_samples` (default 50), stage candidate as `@Challenger` (`gate="insufficient_traffic"`).
   - Evaluates multivariate distribution exchangeability between calibration embeddings and live telemetry using a two-sample RBF MMD squared ($MMD^2$) permutation test. If permutation $p$-value falls below significance threshold ($p < \text{drift\_alpha}$, default 0.01), candidate is staged as `@Challenger` (`gate="calibration_not_exchangeable"`).
   - If all four gates pass, candidate is promoted to `@Production` and the replaced incumbent is automatically demoted to `@Challenger`.

The function `promote_candidate_model(target, criteria)` attaches cryptographic provenance tags (`git_sha`, `dvc_dataset_hash`, `extractor`, `val_pr_auc`, `empirical_coverage`, `p95_latency_ms`) directly to the registered version and applies the aliased routing tags.

---

## Operations & Infrastructure Management

### Multi-Tenant CUDA VRAM Management

When training on shared GPU hardware alongside active ONNX Runtime inference microservices, unbounded PyTorch memory allocation risks triggering Out-of-Memory (OOM) errors in production endpoints.

The function `configure_cuda_memory_budget()` (`src/taxon_vision/models/training/runner.py`) queries the environment variable `TAXON_CUDA_MEMORY_FRACTION` (defaulting to `0.7` / 70% of total VRAM) and applies `torch.cuda.set_per_process_memory_fraction()`. On a 16 GB GPU (e.g. RTX 5070 Ti), this permanently caps training allocation to ~11.2 GB, reserving at least 4.8 GB for co-located low-latency ONNX Runtime serving and OS display buffers under NVIDIA time-slicing.

### Checkpoint Persistence & Serving Cache Invalidation

Following linear head optimization:

1. Trained linear head weights are serialized to disk (`save_checkpoint()`, default `models/checkpoints/head.pt`).
2. `get_classifier.cache_clear()` is invoked on the runtime serving classifier provider (`src/taxon_vision/service/inference.py`). This invalidates in-memory classifier singletons, ensuring subsequent inference API requests immediately load the newly trained weights without requiring a service restart.

---

## Operational Execution & CLI Interfaces

TaxonVision provides unified command-line interfaces, executable modules, and REST API dispatch mechanisms for all training stages.

### Feature Extraction Pipeline

Run offline feature extraction from a Parquet dataset manifest:

```bash
# Extract representations using the default DINOv3 backbone into 4 stratified splits
python -m taxon_vision.pipelines.extract_features \
    --extractor dinov3 \
    --manifest data/manifests/dataset_manifest.parquet \
    --out-dir data/embeddings \
    --batch-size 32

# Run feature extraction via Pixi task automation
pixi run extract-features
```

### Classification Head Training Pipeline

Run linear head training on cached embeddings:

```bash
# Standard linear head training with default parameters
python -m taxon_vision.pipelines.train \
    --extractor dinov3 \
    --epochs 10 \
    --batch-size 64 \
    --lr 0.001 \
    --train-embeddings data/embeddings/train_embeddings.pt \
    --val-embeddings data/embeddings/val_embeddings.pt \
    --checkpoint models/checkpoints/head.pt

# Execute with Bayesian hyperparameter optimization across 20 Optuna trials
python -m taxon_vision.pipelines.train \
    --extractor dinov3 \
    --tune \
    --n-trials 20 \
    --epochs 10

# Strictly require pre-computed embedding cache (disabling synthetic test fallback)
python -m taxon_vision.pipelines.train \
    --require-cache \
    --cuda-memory-fraction 0.70

# Launch training via Pixi task automation
pixi run train
```

### Remote Head Training Dispatch

When managing headless edge devices or triggering retraining from external orchestrators (such as Airflow or GitHub Actions), the CLI can dispatch training remotely to an active FastAPI serving instance:

```bash
# Trigger remote head training via HTTP POST /training
python -m taxon_vision.pipelines.train \
    --api-url http://localhost:8000 \
    --extractor dinov3 \
    --epochs 10 \
    --batch-size 64 \
    --lr 0.001
```

The client submits a `TrainingRequest` JSON payload, waits synchronously for execution, and outputs summary metrics returned by the service.

---

## Verification & Invariant Testing

The training subsystem is validated by automated unit and invariant test suites:

| Test File | Test Routine | Invariant Verified |
| :--- | :--- | :--- |
| `tests/unit/test_class_balanced_loss.py` | `test_class_balanced_loss_computation` | Verifies that rare classes receive strictly higher loss weights than frequent classes ($W_{\text{rare}} > W_{\text{frequent}}$) and produces positive scalar loss. |
| `tests/unit/test_class_balanced_loss.py` | `test_class_balanced_loss_validation` | Enforces validation rules rejecting empty sample arrays or out-of-bounds $\beta \ge 1.0$. |
| `tests/unit/test_models_components.py` | `test_feature_extractor_factory_and_caching` | Verifies backbone instantiation, head stripping (`num_classes=0`), dimensional consistency, and dataset feature caching. |
| `tests/unit/test_models_components.py` | `test_feature_extractor_device_and_fallback` | Verifies device tensor dispatch and validates fallback substitution when `allow_fallback=True`. |
| `tests/unit/test_models_components.py` | `test_feature_extractor_and_head` | Verifies forward passing through `TaxonClassifier`, direct embedding evaluation via `forward_head()`, and dual-output projection. |
| `tests/unit/test_tracking.py` | `test_get_dvc_dataset_hash_success` | Verifies parsing of `dvc.lock` and extraction of the `data/manifests` MD5 checksum. |
| `tests/unit/test_tracking.py` | `test_mlflow_run_scope_success` & `test_mlflow_run_scope_failure_propagation` | Verifies RAII run lifecycle management, tagging runs as `FINISHED` on success and `FAILED` on unhandled exceptions. |
| `tests/unit/test_tracking.py` | `test_evaluate_promotion_gates_*` | Verifies the Four-Gate Model Promotion Architecture (predictive viability, quantization fidelity, conformal coverage, latency SLO, and exchangeability). |
| `tests/unit/test_tracking.py` | `test_promote_candidate_model_*` | Verifies registry alias assignment, previous production demotion, and cryptographic metadata tag application. |
| `tests/unit/test_tracking.py` | `test_promote_model_challenger` & `test_promote_model_archived` | Verifies conditional promotion gating against production baseline PR-AUC scores in Stage 4. |
| `tests/unit/test_tuning.py` | `test_hyperparameter_objective_trial` & `test_log_best_run` | Verifies Optuna objective execution, trial reporting, ASHA pruning hooks, and consolidated MLflow logging. |
| `tests/unit/test_cli_and_training.py` | `test_configure_cuda_memory_budget_cuda_mocked` | Verifies enforcement of per-process CUDA memory budget limits under GPU time-slicing. |
