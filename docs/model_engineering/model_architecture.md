---
type: System Pattern
title: Vision Model Architecture
status: stable
stale_after: "2027-01-01T00:00:00Z"
version: 1.0
description: Architecture of the frozen backbone factory, trainable linear classification head, and Class-Balanced Loss for long-tailed species distributions.
tags: [models, backbone, factory, head, loss, timm, pytorch]
generated: {by: process:docs-librarian, at: "2026-10-06T20:00:00Z"}
verified: {by: process:docs-librarian, at: "2026-10-06T20:00:00Z"}
sources:
  - resource: "src/taxon_vision/models/factory.py"
  - resource: "src/taxon_vision/models/head.py"
  - resource: "src/taxon_vision/models/loss.py"
---

# Vision Model Architecture

TaxonVision decouples feature representation learning from taxonomic classification by
combining a **frozen Vision Foundation Model (VFM) backbone** with a lightweight
**trainable linear head**. This design eliminates catastrophic forgetting while reducing
GPU training time for the classification stage to seconds rather than hours.

## Module Overview

| Module | Class / Function | Role |
| --- | --- | --- |
| `factory.py` | `create_feature_extractor()` | Instantiates frozen timm backbone |
| `factory.py` | `get_feature_dimension()` | Resolves embedding dimensionality |
| `factory.py` | `extract_features()` | Batched image to embedding projection |
| `head.py` | `TaxonClassifier` | End-to-end backbone + trainable head |
| `head.py` | `forward_features()` | Frozen backbone pass - images to embeddings |
| `head.py` | `forward_head()` | Head-only pass - embeddings to logits |
| `loss.py` | `ClassBalancedLoss` | Effective-number-of-samples weighted cross-entropy |

## Backbone Factory

The `factory.py` module provides a centralized registry-driven approach to instantiating
all supported backbones. The canonical model names (e.g. `bioclip-2`, `dinov3`,
`mobilenetv4_conv_small`) are resolved to their corresponding `timm` identifiers from the
application configuration file, eliminating hardcoded magic strings throughout the codebase.

All backbones are loaded with:

* `num_classes=0` - removes the original classification head, returning dense pooled features.
* `requires_grad=False` on every parameter - ensures backbone weights remain frozen during head training.
* `model.eval()` mode set immediately to disable Dropout and BatchNorm stochasticity.

```mermaid
sequenceDiagram
    autonumber
    participant Caller as Caller (runner / API)
    participant Factory as factory.py
    participant Config as taxon_vision.config
    participant Timm as timm / HuggingFace

    Caller->>Factory: create_feature_extractor(model_name, pretrained)
    Factory->>Config: get_settings().model.backbone_registry
    Config-->>Factory: dict[canonical_name -> timm_id]
    Factory->>Timm: timm.create_model(timm_id, num_classes=0, pretrained=pretrained)
    alt Download succeeds
        Timm-->>Factory: nn.Module (pretrained weights)
    else Download fails (offline / CI)
        Factory->>Timm: timm.create_model(timm_id, num_classes=0, pretrained=False)
        Timm-->>Factory: nn.Module (random weights)
    end
    Factory->>Factory: freeze all parameters (requires_grad=False)
    Factory->>Factory: model.eval()
    Factory-->>Caller: frozen nn.Module
    Caller->>Factory: get_feature_dimension(backbone)
    Factory->>Factory: probe with dummy tensor (1, 3, 224, 224)
    Factory-->>Caller: int (embedding dim D)
```

## TaxonClassifier

`TaxonClassifier` is the top-level end-to-end model that holds both the frozen backbone
and the trainable head as two separate sub-modules:

```text
TaxonClassifier
├── backbone: nn.Module     # frozen, requires_grad=False
└── head: nn.Sequential
    ├── nn.Dropout(p)       # regularization
    └── nn.Linear(D -> K)   # trainable weights only
```

It exposes two forward paths to support the **embedding-caching optimization**:

* `forward_features(images)` - runs the frozen backbone and returns `(batch, D)` feature embeddings. Wrapped in `torch.no_grad()`.
* `forward_head(features)` - projects pre-computed embeddings directly to `(batch, K)` logits, bypassing the backbone entirely. Used during all training and hyperparameter tuning loops.

```mermaid
sequenceDiagram
    autonumber
    participant API as FastAPI / Inference
    participant TC as TaxonClassifier
    participant BB as Backbone (frozen)
    participant Head as head: Dropout + Linear

    API->>TC: forward(images: Tensor[B, 3, H, W])
    TC->>BB: forward_features(images)
    BB-->>TC: features: Tensor[B, D]  (no_grad)
    TC->>Head: forward_head(features)
    Head-->>TC: logits: Tensor[B, K]
    TC-->>API: logits

    note over TC,Head: During training, forward_head() is called<br/>directly on cached embeddings, skipping<br/>the backbone pass entirely.
```

## Class-Balanced Loss

Biological observations follow a severe **Zipfian power-law distribution**: a handful of
common species contribute thousands of training images while rare endemic species may have
fewer than five observations. Naive inverse-frequency weighting amplifies gradient noise
from scarce classes.

`ClassBalancedLoss` instead uses the *effective number of samples* concept from Cui et al.
(CVPR 2019), where $\beta \in [0.0, 1.0)$ controls the degree of assumed feature overlap:

$$E_{N_i} = \frac{1 - \beta^{N_i}}{1 - \beta}, \quad W_i = \frac{1 - \beta}{1 - \beta^{N_i}}, \quad \sum_i W_i = K$$

* When $\beta \to 0$: weights collapse to uniform (standard cross-entropy).
* When $\beta \to 1$: weights approach inverse class frequency.

The `beta` value is part of the Optuna search space and is tuned per training run.

```mermaid
sequenceDiagram
    autonumber
    participant Runner as runner.py
    participant CBL as ClassBalancedLoss.__init__
    participant FWD as ClassBalancedLoss.forward

    Runner->>CBL: ClassBalancedLoss(samples_per_class, beta)
    CBL->>CBL: effective_num = (1 - beta^N) / (1 - beta)
    CBL->>CBL: weights = (1 - beta) / effective_num
    CBL->>CBL: normalize: weights *= K / sum(weights)
    CBL->>CBL: register_buffer("weights", Tensor[K])
    CBL-->>Runner: criterion: ClassBalancedLoss

    loop Each training batch
        Runner->>FWD: forward(logits: Tensor[B, K], targets: Tensor[B])
        FWD->>FWD: move weights to logits.device
        FWD->>FWD: F.cross_entropy(logits, targets, weight=weights)
        FWD-->>Runner: scalar loss Tensor
    end
```

## API Reference

::: taxon_vision.models.factory
::: taxon_vision.models.head
::: taxon_vision.models.loss
