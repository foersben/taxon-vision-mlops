---
type: workflow
title: Inference Optimization & ONNX Runtime
description: Architectural specification for PyTorch to ONNX graph export with dual outputs, operator-aware static and dynamic INT8 post-training quantization, ONNX Runtime execution with PyTorch fallback, and Pareto backbone benchmarking.
tags: [onnx, onnx-runtime, quantization, ptq, int8, pareto-benchmarking, bioclip, dinov3, mobilenetv4, inference]
verified:
  - by: openwiki/0.7.2
    at: 2026-10-15T19:46:36.219Z
sources:
  - id: openwiki-source-8ad924b52f67b10056fc5e25
    resource: repo://src/taxon_vision/pipelines/evaluate.py
  - id: openwiki-source-d2314fb7d2c604322ed1f16e
    resource: repo://src/taxon_vision/inference/calibration.py
  - id: openwiki-source-7ad80404d4cdd9f0b9759340
    resource: repo://src/taxon_vision/inference/onnx_engine.py
  - id: openwiki-source-ac488275ac5aaa7af4bd13ca
    resource: repo://src/taxon_vision/inference/onnx_exporter.py
  - id: openwiki-source-e4ad8e996f2c9027ac79d217
    resource: repo://src/taxon_vision/inference/quantizer.py
  - id: openwiki-source-644dda467779a35ba1ffdc4c
    resource: repo://src/taxon_vision/service/inference.py
generated: { by: "openwiki/0.7.2", at: "2026-10-15T19:46:36.219Z" }
---

# Inference Optimization & ONNX Runtime

In ecological computer vision, deep visual classification models face stringent operational constraints. Field biodiversity monitoring nodes, automated camera traps, and interactive taxonomic verification services must operate within sub-25 millisecond p95 latency budgets, execute across heterogeneous hardware (cloud GPUs and edge CPUs), and reliably support downstream uncertainty quantification. Evaluating unoptimized deep learning frameworks directly on the serving hot path introduces excessive memory allocations, Python runtime overhead, and non-deterministic kernel execution latencies.

To achieve production-grade performance, TaxonVision isolates PyTorch to offline representation learning and model training, while compiling and optimizing serving artifacts via the **Open Neural Network Exchange (ONNX)** specification and **ONNX Runtime (ORT)**. The system implements a **dual-output architecture** that concurrently emits class logits and dense visual embeddings, an **operator-aware INT8 Post-Training Quantization (PTQ)** engine that preserves unquantized floating-point logits, and an empirical **Pareto benchmarking suite** to systematically evaluate candidate vision backbones across accuracy, latency, memory footprint, and serving cost.

For related system context, see [System Architecture & Design](../architecture/system-overview.md), [Domain Entities & Taxonomy Perimeter](../architecture/domain-and-taxa.md), [Uncertainty Quantification & Epistemic Safety](../concepts/uncertainty-and-safety.md), [API Service & Web Interface](../operations/service-and-api.md), [Deployment, Scheduling & Observability](../operations/deployment-and-observability.md), and [Test Strategy & Invariants](../testing/test-strategy-and-invariants.md).

---

## Optimization & Runtime Execution Pipeline

The compilation, optimization, and inference execution lifecycle follows a deterministic path from PyTorch training weights to accelerated ONNX Runtime serving:

<!-- openwiki: mermaid parse failed and this diagram was converted to a text fence so it does not break rendering. Fix the diagram source and restore the mermaid fence. Parser error: Heuristic: a semicolon inside a label breaks rendering; rephrase the label. -->
```text
flowchart TD
    subgraph Offline["Offline Compilation & Optimization (src/taxon_vision/inference/)"]
        CKPT["PyTorch Head Checkpoint<br/>(.pt weights)"] --> EXP["ONNX Exporter CLI<br/>(export_classifier_cli)"]
        BACKBONE["Pretrained Backbone<br/>(BioCLIP-2, DINOv3, MobileNetV4)"] --> EXP
        
        EXP --> DUAL["_ONNXDualOutputModule<br/>forward_with_embeddings()"]
        DUAL --> TRACE["torch.onnx.export<br/>(opset 17, dynamic batching)"]
        TRACE --> FP32["Intermediate FP32 ONNX Graph<br/>(Dynamic Batching)"]
        
        FP32 --> MODE{"Quantization Mode Selection<br/>(quantize_onnx_model)"}
        
        CALIB_DATA["Representative Images<br/>(load_calibration_tensors)"] --> RDR["TaxonCalibrationDataReader<br/>(serving-identical geometry)"]
        RDR --> STATIC["Static PTQ (QDQ, MinMax)<br/>quantize_onnx_static()"]
        MODE -- "mode='static' (Conv / ViT backbones)" --> STATIC
        
        MODE -- "mode='dynamic' (Linear heads / CI tests)" --> DYNAMIC["Dynamic Quantization<br/>quantize_onnx_dynamic()"]
        
        LOGIT_FIND["find_logit_nodes()<br/>Locate Logit Add/MatMul Nodes"] --> EXCLUDE["Exclude Logit Nodes<br/>Preserve FP32 Output"]
        EXCLUDE --> STATIC
        EXCLUDE --> DYNAMIC
        
        STATIC --> INT8["Optimized INT8 ONNX Artifact<br/>(models/checkpoints/model.int8.onnx)"]
        DYNAMIC --> INT8
    end

    subgraph Runtime["Online Serving Runtime (src/taxon_vision/service/inference.py)"]
        REQ["Prediction Request<br/>(Normalized Image Tensor)"] --> ROUTE{"Backend Selection<br/>(get_onnx_engine)"}
        
        INT8 -.->|"prefer_onnx_runtime=True"| ENGINE["ONNXInferenceEngine<br/>(CPUExecutionProvider, ORT_ENABLE_ALL)"]
        
        ROUTE -- "ONNX Engine Available" --> RUN_ORT["engine.predict_outputs()<br/>session.run(['logits', 'embeddings'])"]
        ROUTE -- "Fallback / torch backend" --> RUN_TORCH["classifier.forward_with_embeddings()<br/>(PyTorch TaxonClassifier)"]
        
        RUN_ORT --> OUT["InferenceOutputs<br/>(FP32 Logits & Dense Embeddings)"]
        RUN_TORCH --> OUT
    end

    subgraph Downstream["Downstream Consumers"]
        OUT --> PROB["Softmax Probability Calibration<br/>Temperature Scaling (T)"]
        OUT --> OOD["Energy-Based OOD Detection<br/>Helmholtz Free Energy E(x; T)"]
        OUT --> CONF["Split-Conformal Prediction<br/>Non-Conformity Scoring & Sets"]
        OUT --> CAM["Forward-Only Attribution<br/>Target Class Activation Mapping"]
        OUT --> AL["Active Learning Buffer<br/>Embedding Retrieval & Triage"]
    end
```
*End-to-end model optimization and runtime execution pipeline showing ONNX dual-output export, operator-aware INT8 quantization with logit exclusion, and dual-backend serving.*

---

## ONNX Graph Export & Dual-Output Architecture

Standard deep vision architectures commonly expose a single output layer representing categorical class logits. However, in continuous biodiversity monitoring, predictions cannot operate in isolation from representation retrieval. Downstream epistemic safety mechanisms—such as Helmholtz free-energy out-of-distribution (OOD) scoring, split-conformal prediction sets, forward-only Class Activation Mapping (CAM), and active learning sample triage—require direct access to penultimate dense visual embeddings. Re-evaluating the vision backbone to extract feature representations would double operational inference latency.

To eliminate redundant forward passes, TaxonVision encapsulates models inside a dual-output export wrapper that traces both tensor outputs within a single computational graph.

### The Dual-Output Wrapper Module

During ONNX serialization in `src/taxon_vision/inference/onnx_exporter.py`, the exporter inspects the PyTorch model. If the module is an instance of `TaxonClassifier`, it is wrapped within `_ONNXDualOutputModule`:

```python
class _ONNXDualOutputModule(nn.Module):
    def __init__(self, classifier: TaxonClassifier) -> None:
        super().__init__()
        self.classifier = classifier

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.classifier.forward_with_embeddings(x)
```

The underlying `TaxonClassifier.forward_with_embeddings(x)` executes the backbone feature extractor once to produce the penultimate feature representation `embeddings = self.backbone(x)`, and subsequently projects those features through the linear classification head `logits = self.head(embeddings)`. Wrapping this forward pass ensures the traced ONNX computational graph exposes two distinct top-level output nodes: `"logits"` and `"embeddings"`.

### Dynamic Axes and Graph Tracing

The function `export_to_onnx()` serializes the model to disk using `torch.onnx.export`. To support variable batch sizes at runtime without recompiling the execution plan, dynamic axes are declared on all graph input and output tensors:

```python
output_names = ["logits", "embeddings"] if is_dual else ["logits"]
dynamic_axes = {
    "input": {0: "batch_size"},
    "logits": {0: "batch_size"},
}
if is_dual:
    dynamic_axes["embeddings"] = {0: "batch_size"}
```

Key tracing invariants enforced during export:
1. **Evaluation & Device Neutrality:** The model is placed into evaluation mode (`model.eval()`) to disable stochastic operations such as `Dropout` and freeze `BatchNorm` running statistics, and shifted to CPU (`model.cpu()`) using dummy input `torch.randn(1, channels, image_size, image_size)`.
2. **Constant Folding:** Tracing enables constant folding (`do_constant_folding=True`), pre-computing static subexpressions and folding batch normalization weights directly into preceding convolutional filters.
3. **Target Opset:** The target operator set is explicitly configured (`opset_version=settings.model.onnx_opset_version`, default `17`), ensuring compatibility with advanced vision-transformer operations, dynamic quantizers, and hardware execution providers.

### Checkpoint Restoration and Production CLI

The command-line interface `export_classifier_cli()` in `src/taxon_vision/inference/onnx_exporter.py` provides automated model serialization for CI/CD pipelines and deployment containers:

```bash
python -m taxon_vision.inference.onnx_exporter \
    --output models/checkpoints/model_int8.onnx \
    --extractor mobilenetv4_conv_small \
    --checkpoint models/checkpoints/best_head.pt \
    --quantize-int8 \
    --quantize-mode static \
    --calibration-dir data/sample \
    --calibration-samples 128
```

The CLI enforces strict production safety checks during checkpoint deserialization:
- **Strict Key Deserialization:** `_load_head_checkpoint()` utilizes `load_head_checkpoint()`, enforcing `weights_only=True` and `map_location="cpu"`. It accommodates raw head state dictionaries, dictionary containers wrapping `model_state_dict`, and prefixed keys (`head.1.weight`).
- **Incompatible Checkpoint Handling:** If a checkpoint file is missing or contains dimension mismatches (such as an incompatible feature dimension or class count), the CLI fails immediately with exit code `1`. Uninitialized weights are only exported if the caller explicitly supplies `--allow-untrained` (governed globally by `settings.model.allow_untrained_fallback`), preventing untrained models from silently leaking into production environments.

---

## Operator-Aware INT8 Quantization Strategies

Quantizing deep learning computational graphs from 32-bit floating-point (FP32) to 8-bit integers (INT8) reduces artifact file size by ~75%, cuts peak memory bandwidth requirements, and unlocks SIMD vector instructions (AVX-512 VNNI, ARM NEON, Intel DL Boost). However, naive quantization introduces severe numerical degradation in specialized vision architectures. TaxonVision enforces an **operator-aware Post-Training Quantization (PTQ)** strategy governed by `src/taxon_vision/inference/quantizer.py`.

### Static PTQ vs Dynamic Quantization

Quantization techniques are divided based on operator support and graph topology:

| Quantization Property | Static PTQ (`quantize_onnx_static`) | Dynamic Quantization (`quantize_onnx_dynamic`) |
| :--- | :--- | :--- |
| **Quantization Format** | `QuantFormat.QDQ` (QuantizeLinear / DequantizeLinear) | Direct operator conversion (`MatMulInteger`, `DynamicQuantizeLinear`) |
| **Activation Handling** | Precomputed static scales & zero points (`QUInt8`) | Computed dynamically at runtime on each forward pass |
| **Weight Handling** | Signed 8-bit integer (`QInt8`), per-channel scaling | Signed 8-bit integer (`QInt8`), per-channel scaling |
| **Target Operators** | `STATIC_OP_TYPES = ("Conv", "MatMul", "Gemm")` | `DYNAMIC_OP_TYPES = ("MatMul", "Gemm")` |
| **Data Requirements** | Requires representative calibration images | Zero calibration data required (data-free) |
| **Target Architectures** | Conv2D & ViT Backbones (`MobileNetV4`, `EfficientNet`, `BioCLIP`, `DINOv3`) | Pure linear graphs (standalone classification head, VLM projections) |
| **Operational Role** | Production deployment for latency SLA compliance | Offline CI smoke testing and rapid head validation |

#### Why Static QDQ PTQ is Mandatory for Convolutions and ViTs

ONNX Runtime does not implement a performant dynamic INT8 kernel for 2D convolutions (`Conv`). In dynamic quantization, evaluating convolutional layers incurs dynamic runtime scanning of activation tensors to compute min/max ranges, which actually increases latency over baseline FP32 execution. Furthermore, Vision Transformer blocks (such as those in `BioCLIP-2` and `DINOv3`) rely on sensitive layer normalization and scaled dot-product attention.

Under static PTQ:
1. `quant_pre_process()` runs shape inference and constant folding in a temporary directory prior to quantization.
2. Activation scale factors and zero points are calibrated offline using representative domain imagery via `CalibrationMethod.MinMax`.
3. Normalization (`LayerNorm`, `BatchNorm`), activations (`GELU`, `SiLU`), and softmax layers are retained in FP32, maintaining ViT numerical stability while quantizing heavy `Conv`, `MatMul`, and `Gemm` operators.

### Calibration Data Ingestion Pipeline

Static quantization requires estimating activation distributions from domain-specific imagery. The calibration ingestion pipeline in `src/taxon_vision/inference/calibration.py` implements `load_calibration_tensors()`:

- **Representative Image Filtering:** Traverses `calibration_dir` (defaulting to `settings.model.calibration_dir`) for image extensions `.jpg`, `.jpeg`, `.png`, and `.webp`.
- **Deterministic Even Thinning:** Files are sorted lexicographically. If the count exceeds `max_samples` (defaulting to `settings.model.calibration_samples`, typically 128), indices are subsampled using `np.linspace(0, len(files) - 1, num=limit).round().astype(int)`.
- **Identical Serving Preprocessing:** Every calibration image is converted to RGB and passed through `get_image_transform()`—the identical Compose pipeline (Resize, CenterCrop, ToTensor, Normalize) used during live inference serving.
- **Prohibition of Synthetic Data:** Substituting synthetic Gaussian noise is strictly prohibited. Synthetic noise lacks the spatial frequency, contrast boundaries, and natural biological feature distributions of real organisms, producing activation scales that severely clip real feature representations in production.

The batches are supplied to `TaxonCalibrationDataReader(CalibrationDataReader)`, which feeds `(1, 3, H, W)` NumPy float32 tensors to ONNX Runtime's calibrator and supports clean iteration rewinding via `rewind()`.

### The Unquantized Output Logit Invariant

A central architectural invariant of TaxonVision is that **nodes producing the final classification logits are never quantized to INT8**:

$$\text{Logit Nodes Exclusion: } \mathcal{N}_{\text{exclude}} = \text{find\_logit\_nodes}(\mathcal{G})$$

```python
def find_logit_nodes(model_path: Path) -> list[str]:
    graph = onnx.load(str(model_path), load_external_data=False).graph
    producers = {out: node for node in graph.node for out in node.output}
    excluded = []
    for graph_output in graph.output:
        node = producers.get(graph_output.name)
        if node is None:
            continue
        excluded.append(node.name)
        if node.op_type == "Add":
            upstream_name = _find_upstream_matmul(node, producers)
            if upstream_name is not None:
                excluded.append(upstream_name)
    return excluded
```

#### Mathematical Motivation for Logit Exclusion

In TaxonVision, downstream safety components consume raw logits directly:
1. **Energy-Based OOD Detection:** The Helmholtz free energy is defined as $E(\mathbf{x}; T) = -T \cdot \ln \sum_{i=1}^K \exp(z_i / T)$. This function relies on smooth exponential logit differences.
2. **Split-Conformal Prediction:** The non-conformity score $s_i = 1 - p_i$ is computed from temperature-scaled softmax probabilities.

If the final linear layer (`Gemm` or `MatMul` + `Add`) is quantized to an 8-bit integer grid ($[-128, 127]$ or $[0, 255]$), output logits are constrained to discrete step values. This discretization introduces **staircase artifacts** into logit distributions, distorting exponential sums, biasing conformal non-conformity quantiles ($q_{\hat{}}$), and destabilizing OOD detection thresholds. Keeping logit-producing nodes in FP32 ensures smooth mathematical distributions while accelerating 99% of graph parameters in preceding layers.

---

## ONNX Runtime Engine & Execution Semantics

Production inference serving is executed by `ONNXInferenceEngine` in `src/taxon_vision/inference/onnx_engine.py`, coordinated with application lifecycle management in `src/taxon_vision/service/inference.py`.

### Persistent Session and Graph Optimizations

The `ONNXInferenceEngine` class instantiates a long-lived, persistent inference session:

```python
class ONNXInferenceEngine:
    def __init__(self, model_path: str) -> None:
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(model_path, opts, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        output_names = [out.name for out in self.session.get_outputs()]
        self.has_embeddings = "embeddings" in output_names
        self.output_names = ["logits", "embeddings"] if self.has_embeddings else output_names
        self.output_name = self.session.get_outputs()[0].name
        out_dim = self.session.get_outputs()[0].shape[-1]
        self.num_classes = out_dim if isinstance(out_dim, int) else None
```

Key runtime invariants:
- **Comprehensive Graph Optimization:** `GraphOptimizationLevel.ORT_ENABLE_ALL` triggers basic optimizations (redundant node elimination, constant folding), extended optimizations (node fusions such as Conv+Add+Act or MatMul+Add), and layout transformations.
- **Execution Provider Hierarchy:** The default runtime targets `CPUExecutionProvider`, running highly optimized INT8 vector instructions. In GPU container environments, `CUDAExecutionProvider` or `TensorrtExecutionProvider` can be appended to the provider list.
- **Dynamic Output Introspection:** The engine inspects graph outputs to detect whether the secondary `"embeddings"` tensor is emitted, setting `has_embeddings` accordingly.
- **Class Cardinality Verification:** The output dimension of the primary output is checked against the authoritative catalog (`taxa_catalog.yaml`), preventing model/catalog mismatches from entering service.

### Zero-Allocation Prediction API

The engine exposes three prediction methods designed to avoid unnecessary memory allocations:

1. **`predict_outputs(input_array)`**: Accepts an `(N, 3, H, W)` NumPy float32 array, evaluates `self.session.run(self.output_names, {self.input_name: input_array})`, and returns an immutable dataclass:
   ```python
   @dataclass(frozen=True)
   class InferenceOutputs:
       logits: npt.NDArray[np.float32]
       embeddings: npt.NDArray[np.float32] | None = None
   ```
2. **`predict_logits(input_array)`**: Extracts and returns raw floating-point class logits `(N, num_classes)`.
3. **`predict(input_array)`**: Computes numerically stable softmax likelihoods:
   $$p_k = \frac{\exp(z_k - \max_j z_j)}{\sum_i \exp(z_i - \max_j z_j)}$$

### Service Routing and PyTorch Fallback Semantics

In `src/taxon_vision/service/inference.py`, the global dependency injector `get_onnx_engine()` manages runtime selection and fallback behavior:

```python
@lru_cache(maxsize=1)
def get_onnx_engine() -> ONNXInferenceEngine | None:
    cfg = get_settings().model
    if cfg.inference_backend == "torch" or not cfg.prefer_onnx_runtime:
        return None

    onnx_path = _resolve_artifact(cfg.onnx_model_path)
    if not onnx_path.exists():
        if cfg.inference_backend == "onnx":
            raise ModelNotReadyError(f"ONNX model not found at {onnx_path}. Run export_onnx stage first.")
        logger.warning("ONNX artefact %s not found; serving with the PyTorch classifier.", onnx_path)
        return None

    engine = ONNXInferenceEngine(str(onnx_path))
    expected = len(load_taxa_catalog())
    if engine.num_classes is not None and engine.num_classes != expected:
        raise ModelNotReadyError(
            f"ONNX model {onnx_path} outputs {engine.num_classes} classes; catalogue has {expected}."
        )
    return engine
```

#### Dual Backend Execution Routing

When an image prediction is dispatched via `compute_model_outputs(tensor)`:
1. **Primary Route (ONNX Runtime):** If `get_onnx_engine()` returns an active engine, the tensor is converted to a NumPy array (`tensor.numpy()`) and evaluated via `engine.predict_outputs()`. No PyTorch autograd or GPU locks are engaged.
2. **Fallback Route (PyTorch):** If the ONNX engine is unavailable (either because `inference_backend="torch"`, or `inference_backend="auto"` and the artifact is missing on disk), inference routes through `get_classifier()` inside a `torch.no_grad()` block, invoking `classifier.forward_with_embeddings(tensor)`.
3. **Strict Production Failure:** When `inference_backend="onnx"`, a missing ONNX artifact does not trigger fallback; it immediately raises `ModelNotReadyError`. This prevents silent CPU or memory degradation in production clusters configured specifically for ONNX execution.
4. **Readiness Probing & Cache Invalidation:** The Kubernetes readiness probe `check_readiness()` verifies model health by checking if `get_onnx_engine()` returns `"ok (onnx)"` or `get_classifier()` returns `"ok (torch)"`. When models are retrained or re-exported, `reset_model_caches()` invalidates all LRU singletons (`get_classifier`, `get_onnx_engine`, `get_conformal_engine`, `get_ood_detector`).

---

## Pareto Benchmarking Suite

Selecting the optimal vision backbone requires balancing ecological classification accuracy against latency, memory footprint, and cloud serving costs. The Pareto benchmarking suite in `scripts/benchmark_pareto.py` provides multi-objective evaluation across five candidate architectures:

```bash
python scripts/benchmark_pareto.py --out docs/assets/pareto_results.json
```

### Candidate Backbone Empirical Profiles

The suite evaluates five candidate visual backbones spanning Vision-Language models, self-supervised Vision Transformers, and mobile inverted-bottleneck convolutional networks:

| Extractor Identifier | Architecture Family | Canonical Literature Baseline | Parameters (M) | Feature Dim | Target Operational Profile | P50 Latency (ms) | P95 Latency (ms) | Throughput (FPS) | Peak VRAM (MB) | Cost / 1M Inferences (USD) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`bioclip-2`** | Vision-Language (Tree of Life) | Zero-Shot ~59.3% Macro / Few-Shot up to 92.4% (Stevens et al., 2024) | 86.0M | 512 | Server GPU / Domain Taxonomic Prior | 8.5 | 12.0 | 85.0 | 1200.0 | $3.50 |
| **`dinov3_vits14`** | Dense Self-Supervised ViT | Dense Morphology Probe / Register Tokens | 22.0M | 384 | Server GPU / Dense Morphology | 7.0 | 9.5 | 105.0 | 950.0 | $2.60 |
| **`dinov2_vits14`** | Self-Supervised ViT | Linear Probe 81.1% ImageNet-1K (Oquab et al., 2023) | 22.0M | 384 | General Self-Supervised Baseline | 6.8 | 9.2 | 110.0 | 900.0 | $2.50 |
| **`mobilenetv4_conv_small`** | Universal Inverted Bottleneck (UIB) | Official 73.8% ImageNet-1K Top-1 (Qin et al., 2024) | 3.8M | 960 | Extreme Low-Power Edge / Offline Mobile | 2.5 | 4.2 | 280.0 | 180.0 | $0.60 |
| **`efficientnet_b0`** | Compound-Scaled ConvNet | Official 77.1% ImageNet-1K Top-1 (Tan & Le, 2019) | 5.3M | 1280 | Balanced Edge Convolutional Baseline | 4.2 | 6.8 | 160.0 | 300.0 | $1.10 |

### Architectural Trade-Off Analysis

The empirical metrics establish distinct trade-off frontiers depending on deployment constraints:

<!-- openwiki: mermaid parse failed and this diagram was converted to a text fence so it does not break rendering. Fix the diagram source and restore the mermaid fence. Parser error: Heuristic: an unescaped angle bracket inside a label breaks rendering; rephrase the label. -->
```text
flowchart LR
    subgraph GPU["Cloud GPU Serving Frontier (p95 < 15ms)"]
        DINO["DINOv3 ViT-S/14<br/>384-dim, Register Tokens<br/>Dense Morphology Baseline"]
        BIO["BioCLIP-2<br/>512-dim, 86M params<br/>Taxonomic Tree Priors"]
    end

    subgraph Edge["Edge / CPU Offline Frontier (p95 < 5ms)"]
        MNV4["MobileNetV4 Conv-Small<br/>960-dim, 3.8M params<br/>Ultra-low Latency (280 FPS)"]
        EFF["EfficientNet-B0<br/>1280-dim, 5.3M params<br/>Balanced Baseline (160 FPS)"]
    end

    BIO -.->|"Higher Domain Semantic Prior (+33% Cost)"| DINO
    DINO -.->|"Quantized Edge Migration (-77% Latency, -77% Cost)"| MNV4
    MNV4 -.->|"Higher Conv Capacity (+1.5M params)"| EFF
```
*Architectural trade-off transitions across candidate vision backbones on the Pareto frontier.*

#### Literature Context vs. Operational Realities

1. **Self-Attention Bottlenecks on Edge CPUs:** Vision Transformers (`BioCLIP-2`, `DINOv3`, `DINOv2`) achieve sub-10ms latencies on server GPUs via Tensor Cores and sequence parallelism. However, on edge CPU hardware without vector accelerators, quadratic self-attention mechanisms bottleneck computation, pushing latency past 35ms.
2. **MobileNetV4 Universal Inverted Bottlenecks:** `mobilenetv4_conv_small` features Universal Inverted Bottleneck (UIB) blocks optimized for mobile and CPU caches. Achieving a p95 latency of 4.2ms and 280 FPS at $0.60 per million inferences, it serves as the premier choice for low-cost, offline field hardware.
3. **DINOv3 Register Tokens:** Unlike DINOv2, DINOv3 incorporates dedicated register tokens that eliminate high-norm artifacts in feature maps, preserving fine-grained insect wing venation and botanical leaf serrations into compact 384-dimensional embeddings.

---

## Configuration Reference & Operational Diagnostics

The inference and quantization pipelines are configured via `ModelConfig` in `src/taxon_vision/config.py`, customizable via YAML configuration files or environment variables (`TAXON_MODEL__*`).

### Configuration Parameters

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `onnx_model_path` | `str` | `"models/checkpoints/model_int8.onnx"` | Path to the serialized and quantized INT8 ONNX graph artifact. |
| `inference_backend` | `Literal["auto", "onnx", "torch"]` | `"auto"` | Runtime backend mode: `"onnx"` strictly requires the ONNX artifact, `"torch"` forces PyTorch, `"auto"` uses ONNX if present with PyTorch fallback. |
| `prefer_onnx_runtime` | `bool` | `True` | Prioritize ONNX Runtime over PyTorch when available. |
| `quantization_mode` | `Literal["auto", "static", "dynamic"]` | `"auto"` | INT8 quantization mode: `"static"` (PTQ with calibration data), `"dynamic"` (data-free), `"auto"` (static if calibration data exists). |
| `calibration_dir` | `str` | `"data/sample"` | Directory containing representative images for static PTQ calibration. |
| `calibration_samples` | `int` | `128` | Maximum number of images sampled for static activation scale calibration. |
| `onnx_opset_version` | `int` | `17` | Target ONNX operator set version for graph export. |
| `allow_untrained_fallback` | `bool` | `False` | Permit uninitialized/fallback weights during offline unit tests; strictly `False` in production. |
| `image_size` | `int` | `224` | Input image spatial resolution (height and width in pixels). |
| `eval_resize_ratio` | `float` | `256 / 224` (~1.143) | Scaling factor applied to the shorter image edge prior to center-cropping. |

### Diagnostic Failure Modes & Remediation

| Symptom / Error | Root Cause | Remediation Procedure |
| :--- | :--- | :--- |
| `ModelNotReadyError: ONNX model not found at ...` | `inference_backend="onnx"` is asserted but the ONNX graph was not compiled or exported. | Run `python -m taxon_vision.inference.onnx_exporter --quantize-int8` or switch backend to `"auto"` in configuration. |
| `ModelNotReadyError: ONNX model ... outputs X classes; catalogue has Y.` | The exported ONNX graph was compiled against an outdated taxa catalog. | Re-export the ONNX graph using the updated catalog: `python -m taxon_vision.inference.onnx_exporter --checkpoint models/checkpoints/best_head.pt`. |
| `ValueError: Static quantization requires calibration tensors...` | Static PTQ was requested without valid calibration images. | Ensure `data/sample` contains readable `.jpg`, `.png`, or `.webp` files, or specify `--calibration-dir <path>`. For CI smoke tests, specify `--quantize-mode dynamic`. |
| `RuntimeError: Static INT8 quantization of ... failed` | Shape inference mismatch or unsupported operator in the backbone graph. | Verify that `opset_version >= 17` and that input dimensions match `(1, 3, image_size, image_size)`. |
| Quantization accuracy collapse (OOD/conformal failure) | Logit-producing nodes were quantized, introducing staircase discretization. | Ensure `find_logit_nodes()` excludes the final `MatMul` and `Add` nodes from quantization. Confirm that `STATIC_OP_TYPES` excludes normalization and activations. |

### Unit Testing & Invariant Verification

The quantization and export pipelines are verified under `tests/unit/`:
- `tests/unit/test_onnx_exporter.py`: Validates dummy module export, CLI execution with `--allow-untrained`, strict failure on missing or mismatched checkpoints, raw/wrapped/prefixed state dict loading in `_load_head_checkpoint`, and `ONNXInferenceEngine` output shape correctness.
- `tests/unit/test_quantizer.py`: Validates `TaxonCalibrationDataReader` sequential batch extraction and rewinding, dynamic quantization execution, static PTQ execution with calibration data, and the strict prevention of silent FP32 copying on quantization failures.
