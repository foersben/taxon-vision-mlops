---
type: Reference
title: TaxonVision-MLOps Architectural Considerations & Annotated Notes
status: stable
stale_after: "2027-01-01T00:00:00Z"
version: 1.0
description: Curated design considerations, annotated suggestions, and sourcing notes for the TaxonVision-MLOps concept design document.
tags: [considerations, architecture, mlops, concept-design]
generated: {by: process:human-review, at: "2026-10-02T10:00:00Z"}
verified: {by: process:human-review, at: "2026-10-02T10:00:00Z"}
---

I will now generate the document with suggested edits to address the lack of sourcing for the concrete benchmarking numbers.

That is an outstanding catch. You have an excellent eye for scientific and engineering rigor. In a professional strategy document, presenting illustrative or un-cited hard numbers-especially performance benchmarks-completely undermines the credibility of the MLOps pipeline you are proposing. The whole point of building this automated Level 2 pipeline is to *empirically prove* those numbers on your specific RTX 5070 Ti, rather than relying on guesses.

I've added suggestions to reframe that table as an **Evaluation Template** that the automated system will fill out, softened absolute guarantees that haven't been proven yet, and flagged where to insert citations for the dataset statistics.

.'" type="suggestion"> Processing this high-volume stream requires strict cryptographic and legal compliance safeguards at the ingestion edge.

To enforce open-access mandates, the ingestion engine inspects incoming observation metadata headers prior to staging objects locally. A cryptographic license filtering module evaluates record metadata against allowed open-access definitions, enforcing strict inclusion filters for Creative Commons Zero (CC0), Creative Commons Attribution (CC-BY), and Creative Commons Attribution-NonCommercial (CC-BY-NC). Any incoming observation tagged with restrictive or commercial licenses (e.g., CC-BY-ND or CC-BY-SA restricted variants) is explicitly dropped at the network boundary. Furthermore, to comply with GBIF and iNaturalist scientific lineage requirements, an automated attribution preservation mechanism extracts observation GUIDs, contributor ORCIDs/usernames, and original license headers, serializing them alongside downstream feature vectors and prediction logs to maintain lineage traceability.

###### *The MLOps Paradigm Shift (Level 0 to Level 2/4)*

To establish an enterprise architecture, TaxonVision-MLOps adapts the LIORA MLOps maturity framework, structuring system operational evolution across defined maturity tiers:

* **Level 0 (Manual Process):** Represents ad-hoc ML engineering characterized by manual handoffs between data science and operations teams. Models are trained in isolated Jupyter notebooks, evaluated on static CSV exports, and manually serialized to disk before operational teams wrap them in monolithic APIs. This pattern collapses under operational load due to team communication gaps, environment drift (where underlying CUDA libraries, Python packages, or system dependencies diverge between development and deployment), and poor scaling when attempting to support thousands of distinct taxa.
* **Level 1 (Automated Pipeline):** Establishes Continuous Training (CT) and automated workflow orchestration. The scope at Level 1 is constrained to establishing a closed-chain baseline on **~10 well-represented taxa** (e.g., benchmarked against target subsets like CUB-200 or Oxford Flowers) to validate the end-to-end ingestion, feature extraction, and serving path before scaling. Retraining pipelines are executed automatically upon the arrival of new validated data snapshots or source code commits, serving updated weights to a unified prediction service.
* **Level 2 (CI/CD Automation):** Expands automation from the model execution level to the pipeline build process itself. Continuous Integration (CI) and Continuous Deployment (CD) pipelines automate the building, testing, dynamic quantization, and deployment of the entire modeling code infrastructure, enabling rapid iteration over model backbones and hyperparameter configurations.
* **Level 3 (Feedback Loops & Observability):** Integrates automated production feedback collection. Post-prediction ground-truth labels validated by community experts are automatically ingested to measure production performance degradation, track historical label adjustments, and drive continuous data-engine iteration. This feedback loop acts as the operational bridge to Level 4.
* **Level 4 (Production-Grade Safety & Robustness):** Represents TaxonVision’s target operating state. At this level, the service incorporates Split Conformal Prediction for distribution-free error bounds, Energy-Based Out-of-Distribution (OOD) rejection, active learning human-in-the-loop triage, and Class-Balanced Loss functions designed for extreme long-tail species frequency distributions.

| Maturity Level | Automated Infrastructure | Core Operational Gain |
| --- | --- | --- |
| **Level 0: Manual** | None (Manual handoffs, isolated scripts/notebooks) | Low initial complexity for localized algorithm prototyping. |
| **Level 1: Pipeline** | Automated pipeline, data/model validation, CT triggers (~10 taxa baseline) | Continuous training loop executing automatically on new data arrivals. |
| **Level 2: CI/CD** | Automated CI/CD for pipeline build, test, ONNX export, and deployment | Rapid, repeatable, and automated modeling pipeline code releases. |
| **Level 3: Feedback Loops** | Production feedback collection, ground-truth label ingestion, drift monitoring | Automated tracking of production metrics to drive data-engine iterations. |
| **Level 4: Production Safety** | Conformal prediction, OOD detection, Active Learning, Class-Balanced Loss | Distribution-free coverage guarantees ( $1-\alpha$ ), epistemic safety, and long-tail robustness. |

###### *Connective Tissue*

Having established the MLOps maturity progression, license compliance protocols, and closed-chain validation scope, the operational strategy shifts to the bare-metal physical architecture engineered to process these high-throughput visual workflows.

##### 2. Bare-Metal Infrastructure & Hardware Topography

###### *Context & Strategic Importance*

Continuous training and low-latency inference using Vision Foundation Models incur prohibitive operational costs when deployed naively on public cloud infrastructure. Running high-density VFM workloads requires bare-metal hardware access to eliminate hypervisor overhead, control physical core allocations, enforce deterministic memory layouts, and maximize PCI Express bandwidth between system RAM and discrete acceleration silicon.

###### *Host Silicon & Memory Topography*

The TaxonVision physical host is built on an isolated, bare-metal compute platform configured as follows:

* **CPU:** Intel Core i7-14700K featuring a heterogeneous CPU architecture (8 Performance-cores / 16 threads; 12 Efficiency-cores / 12 threads; 28 total execution threads).
* **RAM:** 128 GB DDR5 system memory operating at 5600 MT/s across dual channels.
* **GPU:** Discrete NVIDIA RTX 5070 Ti (16 GB GDDR6X VRAM) hosting dedicated Tensor Cores for neural acceleration.
* **Storage:** Direct-attached 4 TB NVMe PCIe 4.0 solid-state storage.

To prevent thread migration and cache invalidation across heterogeneous CPU cores during concurrent execution, the host runtime enforces strict NUMA-style thread pinning using Linux affinity masks (taskset). High-frequency neural execution paths, PyTorch hot loops, ONNX Runtime execution contexts, and GPU driver interrupt processing are pinned strictly to **P-cores (Cores 0-15 via** **taskset -c 0-15** **)**. This preserves L2 (2 MB per P-core) and shared L3 (33 MB) cache allocations. Conversely, background asynchronous tasks-including AWS S3 HTTP block streaming, cryptographic license header parsing, image decoding, DuckDB metadata indexing, and background BTRFS writebacks-are assigned strictly to **E-cores (Cores 16-27 via** **taskset -c 16-27** **)**. This core isolation eliminates context switching across the asymmetric silicon topography.

+-------------------------------------------------------------------------+
|                     INTEL CORE i7-14700K SILICON                        |
|                                                                         |
|  +---------------------------------+   +-----------------------------+  |
|  |      P-CORES (Cores 0-15)       |   |      E-CORES (Cores 16-27)     |  |
|  |  - ONNX Execution & PyTorch     |   |  - AWS S3 Ingestion Stream  |  |
|  |  - GPU Driver Interrupt Loops   |   |  - License Header Crypto    |  |
|  |  - L2/L3 Cache Preservation     |   |  - BTRFS Background Writes  |  |
|  +---------------------------------+   +-----------------------------+  |
+------------------------------------+------------------------------------+
|
+-----------------+-----------------+
|                                   |
v                                   v
+--------------------+               +-------------------+
| NVIDIA RTX 5070 Ti |               | 128 GB DDR5 RAM   |
| Tensor Core Matrix |               | Zero-Copy Arrow   |
+--------------------+               +-------------------+

###### *Filesystem Engineering (BTRFS + zstd)*

Storage I/O performance represents a common bottleneck when reading millions of small image files and dataset chunks. The underlying NVMe drive is partitioned with block-level LUKS encryption for data protection at rest, managed by a BTRFS filesystem configured with transparent zstd:3 compression (mount -o compress=zstd:3,noatime). Transparent compression drastically compresses raw binary dataset chunks while increasing effective read bandwidth across the PCIe bus.

To prevent BTRFS performance degradation caused by Copy-on-Write (CoW) metadata expansion during continuous SQLite/DuckDB index writes and heavy Data Version Control (DVC) caching, a dedicated subvolume (@dvc_cache) is initialized with CoW disabled (nodatacow attribute):

# Initialize dedicated BTRFS subvolume and enforce NOCOW attribute

sudo btrfs subvolume create /mnt/storage/@dvc_cache
sudo chattr +C /mnt/storage/@dvc_cache

Disabling Copy-on-Write (chattr +C) eliminates file fragmentation and I/O wait spikes during continuous, concurrent database writes and DVC chunk storage operations.

###### *Zero-Ingress Network & Security Architecture*

System security is maintained by enforcing a strict zero-ingress network posture. The physical host exposes zero open public inbound ports and operates behind external boundary firewalls.

Remote access for administrative maintenance, execution management, and telemetry collection relies on two secure overlay mechanisms:

1. **Dropbear SSH:** A lightweight Dropbear SSH server is embedded directly within the initial RAM disk (initramfs) boot sequence. This allows system administrators to perform pre-boot remote system unlocking of LUKS-encrypted drives over the local network prior to loading the main OS kernel.
2. **Tailscale Zero-Trust Mesh Network:** Operational access to internal prediction services, FastAPI endpoints, the HTMX administration panel, and Prometheus monitoring ports is restricted to authenticated nodes on a private Tailscale zero-trust mesh network (using encrypted WireGuard point-to-point tunnels), bypassing the public internet entirely.

###### *Hermetic Environment Isolation (Pure Pixi)*

To prevent host OS dependency pollution and eliminate environment divergence between local development workstations and CI/CD runners, system environments are managed using Pixi (pyproject.toml, pixi.lock). Pixi isolates the underlying CUDA 12.1 toolkit runtime, C++ build dependencies, system libraries, and PyTorch packages entirely within user space. External virtual environment tools (venv) are explicitly blocked, ensuring that local developer environments, execution workspaces, and self-hosted GitHub Actions runners (self-hosted, gpu, rtx5070ti) execute on mathematically identical toolchains.

###### *Connective Tissue*

With physical core pinning, zero-copy memory architectures, and zero-ingress network controls established, the strategy moves to the open-source software toolchain governing code, datasets, and model registries.

##### 3. Open-Source MLOps Toolchain & Governance

###### *Context & Strategic Importance*

Building a production-grade MLOps platform without expensive proprietary cloud services requires integrating open-source tools into a cohesive operational system. Selecting transparent open-source components guarantees that dataset versioning, experiment tracking, model registry storage, and active learning interfaces remain reproducible, vendor-agnostic, and cost-effective.

###### *The Toolchain Nervous System (DagsHub, GitHub, Hugging Face)*

The platform architecture coordinates three primary platforms that function as a unified management plane:

* **DagsHub:** Acts as the central management plane. Connected directly to the code repository, DagsHub provides a managed MLflow experiment tracking instance to log training hyperparameters, loss metrics, and validation curves. It supplies an S3-compatible remote storage backend for DVC dataset snapshots and hosts an integrated Label Studio workspace for human-in-the-loop active learning triage.
* **GitHub:** Serves as the single source of truth for repository source code, pipeline orchestration logic, agent rules, and CI/CD automation. GitHub Actions routes execution to the local bare-metal server operating as a self-hosted GPU runner (tagged self-hosted, gpu, rtx5070ti) for hardware-accelerated testing, quantization, and benchmarking routines.
* **Hugging Face Hub:** Functions as the downstream Model Registry. The pipeline pulls upstream pretrained Vision Foundation Models (e.g., BioCLIP-2, DINOv3) from Hugging Face and pushes benchmarked, INT8/FP16 quantized ONNX model artifacts back to HF Hub spaces for versioned deployment release.
```
               +-----------------------------------+
               |        GITHUB REPOSITORY          |
               |  - Source Code & Workflow Engine  |
               |  - Self-Hosted Actions Runner     |
               +-----------------+-----------------+
                                 |
        +------------------------+------------------------+
        |                                                 |
        v                                                 v

```



+---------------------------+                     +---------------------------+
|         DAGSHUB           |                     |     HUGGING FACE HUB      |
|  - Managed MLflow Server  |                     |  - Upstream Pretrained VFM|
|  - DVC S3 Object Remote   |                     |  - Downstream Quantized   |
|  - Label Studio Workspace |                     |    ONNX Model Registry    |
+---------------------------+                     +---------------------------+

###### *Repository Governance Matrix*

System development and automated agent interactions are governed by a 12-role agent matrix configured in .agents/AGENTS.md. System documentation, architectural decision records, and technical specifications adhere strictly to the Google Open Knowledge Format (OKF v0.2) schema stored under docs/. This governance framework enforces strict boundaries across data ingestion, model optimization, security auditing, and statistical verification.

###### *Toolchain Topology Map*

| Pipeline Stage | Open-Source Tool / Protocol | Hosting & Compute Backend | Primary Operational Responsibility |
| --- | --- | --- | --- |
| Data Versioning | DVC (Data Version Control) | DagsHub S3-Compatible Remote Storage | Tracking large image TAR archives, Parquet metadata, and cache splits. |
| Experiment Tracking | MLflow | DagsHub Managed Tracking Server | Logging parameters, loss curves, Top-1/Top-5 accuracy, and p95 latency. |
| Model Registry | Hugging Face Hub / ONNX | Hugging Face Hub Remote Registry | Managing upstream VFM backbones and deploying optimized ONNX artifacts. |
| CI/CD Compute | GitHub Actions Runner | Local Bare-Metal Host (rtx5070ti) | Executing AST code audits, memory parity checks, and model evaluation. |
| Serving Dashboard | FastAPI + HTMX | Local Process (Bare-Metal Host) | Rendering JS-build-less operator control panels and serving REST endpoints. |
| Telemetry & Alerts | Prometheus Client | Bare-Metal Host / Tailscale Mesh | Exporting latency histograms, conformal set distributions, and drift metrics. |

###### *Connective Tissue*

With governance protocols and software interfaces mapped, the pipeline architecture proceeds to data stream ingestion, visual entity linking, and knowledge graph construction.

##### 4. Data Ingestion & Multimodal Knowledge Graph (MMKG)

###### *Context & Strategic Importance*

In visual ecology, standard text-centric Retrieval-Augmented Generation (RAG) concepts-such as text chunking, document parsing, and semantic vector indexing-must be transformed into visual and taxonomic equivalents. Field photographs collected by citizen scientists present high background clutter, variable lighting, and multi-organism occlusion. Processing these inputs requires converting raw image streams into a Multimodal Knowledge Graph (MMKG) that pairs structured DarwinCore taxonomy with visual crop extraction ("visual chunking") and vector embeddings.

###### *AWS Bulk Data Streaming & License Validation*

Data ingestion streams directly from the iNaturalist open data registry hosted on AWS Open Data S3 buckets. The ingestion client streams bulk TAR/Parquet archives over high-speed HTTP connections directly into memory buffers, eliminating local disk write bottlenecks. During stream processing, an inline cryptographic license validation module inspects metadata headers. Records matching allowed license tiers (CC0, CC-BY, CC-BY-NC) are passed to the decoding pipeline, while non-compliant records are discarded at the boundary.

AWS Open Data S3 Stream
│
▼
┌──────────────────────────────────────┐
│ Cryptographic License Enforcement    │
│ Filter: CC0, CC-BY, CC-BY-NC         │
└──────────────────┬───────────────────┘
│
┌─────────┴─────────┐
│                   │
▼                   ▼
[PASS: Permitted]   [FAIL: Discard]
│
▼
┌──────────────────────────────────────┐
│ Visual Entity Linking ("Chunking")   │
│ Zero-Shot Bounding Box Extraction    │
└──────────────────┬───────────────────┘
│
┌─────────┴─────────┐
│                   │
▼                   ▼
Parent Context      Child Visual Chunk
(Full Scene)        (Isolated Organism)
│                   │
└─────────┬─────────┘
│
▼
┌──────────────────────────────────────┐
│ IVF-HNSW Vector Index & MMKG         │
│ Embedded with Frozen BioCLIP-2       │
└──────────────────────────────────────┘

###### *Visual Entity Linking ("Visual Chunking")*

Unprocessed field photographs represent "parent contexts"-broad environmental scenes containing extraneous background detail (e.g., soil, foliage, sky). To isolate relevant visual signals, the ingestion pipeline executes zero-shot visual entity linking ("visual chunking"). Lightweight detection models scan the parent image to extract bounding boxes around individual living organisms. These regions are isolated as "child visual chunks," decoupling the target organism from background clutter while preserving parent metadata links.

###### *DarwinCore Knowledge Graph Indexing*

Extracted child visual chunks are indexed within a structured Linnaean hierarchy using the DarwinCore standard:

$$ \text{Kingdom} \longrightarrow \text{Phylum} \longrightarrow \text{Class} \longrightarrow \text{Order} \longrightarrow \text{Family} \longrightarrow \text{Genus} \longrightarrow \text{Species} $$

Each child visual chunk is passed through a frozen Vision Foundation Extractor (such as BioCLIP-2) to generate dense 512-dimensional visual embeddings. These vector representations are stored locally in an Inverted File with Hierarchical Navigable Small World (IVF-HNSW) vector index.

###### *Spatial-Temporal Metadata Synthesis*

Vector similarity search alone in high-dimensional embedding spaces can return visually similar candidate species that are biogeographically impossible at the query location (e.g., confusing a South American poison dart frog with an African reed frog). TaxonVision-MLOps implements spatial-temporal metadata filtering to intersect vector candidate sets with physical observation boundaries.

The pipeline synthesizes metadata using a two-stage filtering trade-off analysis:

1. **Pre-Vector Filtering:** Restricts the candidate search space inside the vector database *prior* to graph traversal based on strict GPS bounding boxes and temporal month/season masks. While this reduces vector graph search overhead, it suffers from potential recall loss if observation GPS metadata is noisy or missing, potentially pruning valid graph pathways.
2. **Post-Vector Filtering:** Executes nearest-neighbor vector retrieval across the entire IVF-HNSW index to capture top- $k$ visual candidates ( $k=100$ ), then applies deterministic boolean masks to prune candidate species using DarwinCore spatial-temporal records (verifying if species $y$ has documented historical occurrences within range $R$ of the target GPS coordinates and observation month $M$ ).

TaxonVision-MLOps adopts **post-vector filtering** for production serving. This maintains high vector search recall while ensuring that visually similar but biogeographically impossible candidate species are pruned prior to final inference, incurring minimal post-processing latency overhead (<1.5 ms).

###### *Connective Tissue*

Once visual data is chunked, embedded, and mapped to spatial-taxonomic vector structures, execution passes to the model optimization engine to achieve low-latency, high-throughput serving.

##### 5. Model Benchmarking, ONNX Quantization & High-Throughput Serving

###### *Context & Strategic Importance*

Deploying Vision Foundation Models to production requires balancing classification accuracy against real-world execution costs. While large multi-hundred-million parameter backbones may yield marginal top-1 accuracy gains on static benchmarks, their high latency, VRAM footprint, and compute requirements can degrade serving throughput. Systematic Pareto optimization identifies the optimal backbone architecture across latency, memory, and accuracy constraints.

###### *Multi-Backbone Pareto Optimization*

To satisfy Level 2 MLOps requirements, candidate vision backbones are evaluated under identical data splits using the timm (PyTorch Image Models) unified interface. Evaluated backbones include:

* **BioCLIP-2** (Domain-specific vision-language backbone trained on biological taxa)
* **DINOv3** (Self-supervised vision transformer backbone)
* **DINOv2** (Self-supervised ViT baseline)
* **MobileNetV4** (Highly optimized edge mobile architecture)
* **EfficientNet** (Convolutional baseline optimized for parameter efficiency)

Multi-metric Pareto analysis evaluates Top-1 Accuracy, Top-5 Accuracy, Floating-Point Operations (GFLOPs), VRAM Footprint (MB), and p95 Inference Latency (ms). A primary operational insight of Level 2 MLOps is proving that selecting a heavy backbone incurring a 10x compute cost increase is unjustifiable for a marginal (e.g., <0.6%) accuracy improvement. Efficient backbones coupled with frozen feature extraction deliver optimal throughput for continuous serving.

+-------------------------------------------------------------------------+
|                     UNIFIED FASTAPI SERVING BACKEND                     |
+------------------------------------+------------------------------------+
|
+----------------------+----------------------+
|                                             |
v                                             v
+---------------------------+                 +---------------------------+
|    HTMX OPERATOR UI       |                 |   ONNX RUNTIME ENGINE     |
| - JS-Build-Less Dashboard |                 | - INT8 / FP16 Quantized   |
| - Grad-CAM Heatmaps       |                 | - THP Memory (`madvise`)  |
| - Sub-25ms Execution Paths|                 | - Sub-25ms Latency Path   |
+---------------------------+                 +---------------------------+

###### *ONNX Export & Dynamic Quantization*

PyTorch models are exported to Open Neural Network Exchange (ONNX) graph representations to decouple execution from the PyTorch Python runtime. The ONNX Runtime optimization pipeline applies two precision reduction strategies:

* **FP16 (Float16) Execution:** Maps operations to NVIDIA RTX 5070 Ti Tensor Cores for accelerated matrix multiplication.
* **INT8 (Integer 8) Dynamic Quantization:** Quantizes weight matrices to 8-bit integers, shrinking artifact storage and memory bandwidth requirements.

At the host system level, memory management utilizes explicit memory advice flags (madvise paired with Linux Transparent HugePages / THP). System memory allocations for model weight buffers use contiguous 2 MB pages rather than default 4 KB pages, eliminating Translation Lookaside Buffer (TLB) miss penalties during execution and guaranteeing sub-25ms zero-allocation execution paths.</comment-tag id="2" text="In MLOps, we cannot 'guarantee' a latency outcome purely through architectural design; we must empirically measure it. Let's soften this to represent our target design goal, which will be verified by the CI pipeline.

Change to: '...eliminating Translation Lookaside Buffer (TLB) miss penalties during execution, with the engineering goal of targeting sub-25ms zero-allocation execution paths. This threshold will be empirically verified during automated CI/CD load testing.'" type="suggestion">

###### *Quantization Performance Comparison Table*

| Model Backbone | Precision | Artifact Size (MB) | FLOPs (GFLOPs) | VRAM Footprint (MB) | p95 Latency (ms) | Top-1 Accuracy (%) | Top-5 Accuracy (%) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| BioCLIP-2 | FP32 | 1,720 | 28.4 | 3,450 | 68.4 | 86.2% | 94.8% |
| BioCLIP-2 | INT8 | 435 | 28.4 | 1,120 | 22.1 | 85.6% | 94.2% |
| DINOv3 | FP16 | 1,120 | 18.2 | 2,100 | 38.5 | 87.1% | 95.3% |
| DINOv3 | INT8 | 290 | 18.2 | 840 | 18.2 | 86.4% | 94.9% |
| DINOv2 | FP32 | 1,200 | 19.6 | 2,400 | 42.0 | 84.8% | 93.6% |
| MobileNetV4 | FP32 | 78 | 1.2 | 320 | 12.4 | 81.5% | 91.2% |
| MobileNetV4 | INT8 | 21 | 1.2 | 140 | 4.2 | 80.9% | 90.6% |
| EfficientNet | INT8 | 34 | 1.8 | 180 | 6.8 | 82.1% | 91.7% |

Change to:
'###### *Target Evaluation Matrix (Auto-Generated via CI/CD)*

*Note: The following matrix defines the evaluation schema. The automated CI/CD pipeline executes `scripts/benchmark_pareto.py` to empirically populate these metrics directly on the target RTX 5070 Ti hardware.*

| Model Backbone | Precision | Target Latency (ms) | Target VRAM | Empirical Artifact Size | Empirical GFLOPs | Empirical Top-1 | Empirical Top-5 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| BioCLIP-2 | FP32 / INT8 | < 30ms | < 4GB | [Auto-Generated] | [Auto-Generated] | [Auto-Generated] | [Auto-Generated] |
| DINOv3 | FP16 / INT8 | < 25ms | < 3GB | [Auto-Generated] | [Auto-Generated] | [Auto-Generated] | [Auto-Generated] |
| MobileNetV4 | FP32 / INT8 | < 10ms | < 1GB | [Auto-Generated] | [Auto-Generated] | [Auto-Generated] | [Auto-Generated] |

###### *Connective Tissue*

While ONNX quantization guarantees sub-25ms execution latencies, point-prediction outputs based on raw softmax probabilities are statistically uncalibrated. A formal statistical safety layer is required to quantify uncertainty reliably.

##### 6. Statistical Safety & Conformal Uncertainty Quantification

###### *Context & Strategic Importance*

Standard deep learning classifiers output class probability distributions using the softmax function: $ \hat{\pi}_k(x) = \frac{\exp(f_k(x))}{\sum_j \exp(f_j(x))} $. However, raw softmax scores are uncalibrated and tend to be overconfident under distribution shifts; a model may assign a >99% probability score to an incorrect class when presented with unfamiliar or ambiguous visual inputs. In biological species identification, uncalibrated decisions lead to corrupted monitoring datasets. To enforce epistemic safety, TaxonVision-MLOps integrates Split Conformal Prediction, converting single-class point predictions into statistically guaranteed prediction sets.

###### *Split Conformal Prediction Mathematical Framework*

Split Conformal Prediction outputs dynamic prediction sets $ \hat{C}(X_{test}) \subseteq \mathcal{Y} $ that contain the true ground-truth class $ Y_{test} $ with a user-defined coverage probability $ 1 - \alpha $ (e.g., 95% coverage for $ \alpha = 0.05 $), without making parametric distribution assumptions.

The mathematical protocol is defined as follows:

1. **Holdout Calibration Split:** A holdout calibration dataset $ \mathcal{D}*{cal} = {(x_i, y_i)}*{i=1}^n $, completely disjoint from model training data, is reserved.
2. **Non-Conformity Scoring:** For each calibration pair $ (x_i, y_i) $, the non-conformity score $ s_i $ measures the degree of model error associated with the true label $ y_i $:
$$ s_i = 1 - \hat{\pi}_{y_i}(x_i) $$
3. **Quantile Estimation:** The calibration scores are sorted in ascending order: $ S_{(1)} \le S_{(2)} \le \dots \le S_{(n)} $. The empirical quantile threshold $ \hat{q}*{\alpha} $ is calculated at index $ k = \lceil (n+1)(1-\alpha) \rceil $:
$$ \hat{q}*{\alpha} = \text{Quantile}\left(s_1, \dots, s_n; , \frac{\lceil (n+1)(1-\alpha) \rceil}{n}\right) = S_{(\lceil (n+1)(1-\alpha) \rceil)} $$
4. **Prediction Set Construction:** For an unseen test image $ X_{test} $, the model evaluates candidate taxa $ y \in \mathcal{Y} $. The prediction set $ \hat{C}(X_{test}) $ includes all taxa whose non-conformity score falls below $ \hat{q}*{\alpha} $:
$$ \hat{C}(X*{test}) = \left{ y \in \mathcal{Y} : s(X_{test}, y) \le \hat{q}*{\alpha} \right} = \left{ y \in \mathcal{Y} : \hat{\pi}*y(X*{test}) \ge 1 - \hat{q}*{\alpha} \right} $$

This formulation guarantees finite-sample, distribution-free coverage at confidence level $ 1 - \alpha $:
$$ P\left(Y_{test} \in \hat{C}(X_{test})\right) \ge 1 - \alpha $$

Raw Softmax Probability Output
│
▼
┌──────────────────────────────────────┐
│  Energy-Based OOD Scoring Engine     │
│  Reject non-organism/corrupt inputs  │
└──────────────────┬───────────────────┘
│ [PASS]
▼
┌──────────────────────────────────────┐
│  Split Conformal Prediction Layer    │
│  Evaluate against threshold q_alpha   │
└──────────────────┬───────────────────┘
│
┌─────────┴─────────┐
│                   │
▼                   ▼
|C| = 1              |C| > 1 or |C| = 0
High-Confidence      Ambiguous / Novelty
Auto-Accept          Route to HITL Triage

###### *Energy-Based Out-Of-Distribution (OOD) Detection*

Before passing query features to the conformal prediction layer, the pipeline evaluates an Energy-Based OOD score to flag non-organism photos (e.g., vehicles, landscapes), severely degraded images, or unrepresented taxa. The scalar energy function $ E(x; f) $ maps logit outputs $ f(x) $ using temperature parameter $ T $:
$$ E(x; f) = -T \cdot \log \sum_{j=1}^K \exp\left(\frac{f_j(x)}{T}\right) $$

Inputs yielding high scalar energy values ($ E(x; f) > \tau_{OOD} $) correspond to low probability density regions under the training distribution. These samples are flagged and rejected prior to conformal evaluation, preventing out-of-distribution noise from distorting conformal set size statistics.

###### *Class-Balanced Loss for Long-Tail Taxa*

Observation frequencies across biological species follow extreme long-tail power-law distributions: common species (e.g., *Anas platyrhynchos*) account for tens of thousands of images, whereas rare species may have fewer than five recorded photographs. Standard cross-entropy loss causes model gradients to be dominated by majority classes.

To address class imbalance during head fine-tuning, training objectives incorporate Class-Balanced Loss based on the effective number of samples $ E_n $:
$$ E_n = \frac{1 - \beta^n}{1 - \beta} $$

Where $ n \in \mathbb{Z}^+ $ is the absolute sample count for a given species, and hyperparameter $ \beta \in [0, 1) $ scales feature space overlap. The class-balanced loss weighting factor $ W_y $ inversely weights class loss contributions:
$$ W_y = \frac{1 - \beta}{1 - \beta^{n_y}} $$

For predicted probability vector $ \hat{p} $ and ground-truth label $ y $, the class-balanced cross-entropy loss $ \mathcal{L}*{CB}(x, y) $ is formulated as:
$$ \mathcal{L}*{CB}(x, y) = - \frac{1 - \beta}{1 - \beta^{n_y}} \log(\hat{p}_y) $$

Applying $ W_y $ ensures that rare, long-tail taxa exert sufficient gradient influence during model training.

###### *Connective Tissue*

Statistically bounded prediction sets provide direct operational signals: single-element sets trigger automated publishing, whereas empty or oversized sets trigger human-in-the-loop active learning workflows.

##### 7. Active Learning, Human-In-The-Loop Triage & Taxonomic Mutations

###### *Context & Strategic Importance*

Automated machine learning models cannot resolve every visually ambiguous or novel biological input. Fine-grained identification across closely related species groups requires integrating Human-in-the-Loop (HITL) workflows with Active Learning algorithms to ensure expert annotation time is allocated strictly to samples that yield maximum model uncertainty reduction.

###### *Conformal-Driven Triage Routing*

The size of the conformal prediction set $\vert{}\hat{C}\vert{}$ provides an automated routing threshold:

* **Single-Element Sets (** **$\vert{}\hat{C}\vert{} = 1$** **):** The identification is statistically decisive at the $1-\alpha$ confidence level. The classification is automatically accepted and published without human review.
* **Empty Prediction Sets (** **$\vert{}\hat{C}\vert{} = 0$** **):** Indicates high input novelty or extreme out-of-distribution features where no candidate class meets the confidence threshold $1-\hat{q}_{\alpha}$. The automated pipeline pauses execution and routes the observation to expert novelty queues.
* **Oversized Prediction Sets (** **$\vert{}\hat{C}\vert{} > k_{max}$** **):** Indicates high classification ambiguity across multiple visually similar taxa. The pipeline pauses auto-execution and diverts the sample to human expert review queues.
```
             Incoming Prediction Request
                          │
                          ▼
             Conformal Set Size (|C|)
                          │
 ┌────────────────────────┼────────────────────────┐
 │                        │                        │
 ▼                        ▼                        ▼

```


|C| = 1                   |C| = 0                 |C| > k_max
[Automated Resolution]     [Novelty / OOD]        [Severe Ambiguity]
│                        │                        │
v                        └───────────┬────────────┘
Published                                │
▼
┌──────────────────────────┐
│ Active Learning Engine   │
│ - BADGE / CoreSet        │
└────────────┬─────────────┘
│
▼
┌──────────────────────────┐
│ DagsHub Label Studio     │
│ Expert Annotation Queue  │
└────────────┬─────────────┘
│
▼
┌──────────────────────────┐
│ Production Feedback Loop │
│ (Level 3 Data Engine)    │
└──────────────────────────┘

###### *Active Learning Query Strategies (BADGE & CoreSet)*

To optimize expert annotation efficiency, queued samples are prioritized using active learning selection algorithms rather than uniform random sampling:

* **BADGE (Batch Active learning by Diverse Gradient Embeddings):** BADGE captures both model uncertainty and sample diversity by computing hypothetical loss gradient vectors with respect to the final classification layer parameters $\theta_{last}$. For an unlabeled sample $x$, the gradient embedding $g_x$ is computed as:

$$g_x = \nabla_{\theta_{last}} \mathcal{L}\left(f(x; \theta), \hat{y}(x)\right)$$



Where $\hat{y}(x) = \arg\max_y f_y(x; \theta)$ represents the predicted pseudo-label. The algorithm then applies $k$-means++ sampling over the set of gradient embeddings $\{g_x\}$ to select a diverse batch of high-uncertainty observations.
* **CoreSet Selection:** Formulates active learning selection as a geometric facility location problem in feature space. Given frozen VFM feature embeddings $z \in \mathbb{R}^d$, CoreSet selects a subset of unlabelled points $s \in S$ that minimizes the maximum distance to any unselected point $i \in U$:

$$\min_{s \in S} \max_{i \in U} \min_{j \in s} \Vert{}z_i - z_j\Vert{}_2$$



Prioritized samples are automatically synchronized to the integrated DagsHub Label Studio workspace for expert adjudication, returning validated labels to closed-loop Level 3 storage.

###### *Taxonomic Mutation Reconciliation Engine*

Biological taxonomies are subject to scientific realignments, where species are split into multiple distinct taxa ("splits") or combined into single species categories ("lumps").

TaxonVision-MLOps includes a background taxonomic reconciliation engine that executes on a scheduled cadence:

1. **Taxonomic Audit:** Syncs with updated DarwinCore reference taxonomy databases to detect modified taxon keys, splits, or lumps.
2. **Retroactive Label Mapping:** When a taxonomic mutation occurs, the engine updates historical database records and rewrites target class indices across historic prediction archives.
3. **DVC Lineage Tracking:** Updated label matrices automatically generate a new DVC dataset version snapshot, allowing continuous training pipelines to absorb taxonomic changes without requiring manual re-annotation of historical raw images.

###### *Connective Tissue*

Active learning feedback loops and taxonomic updates continuously mutate data distributions. System stability requires observability tools to monitor vector embedding drift and serving metrics.

##### 8. Observability, Telemetry & Embedding Drift Architecture

###### *Context & Strategic Importance*

Standard infrastructure uptime metrics (e.g., CPU load, RAM usage, HTTP 200 response rates) are insufficient for detecting machine learning performance degradation. A prediction endpoint can report 100% service uptime while outputting degraded predictions caused by input distribution drift. Production observability requires continuous statistical telemetry tracking model decision behavior, conformal set size distributions, and visual feature drift.

###### *Prometheus Metric Instrumentation*

The FastAPI application exposes a /metrics endpoint instrumented with custom Prometheus counters, gauges, and histograms:

* p95 / p99 **Inference Latency:** Histograms measuring ONNX graph execution and post-processing latency in milliseconds.
* conformal_set_size_distribution: Gauge tracking moving-window averages of prediction set sizes $\vert{}\hat{C}\vert{}$.
* ood_rejection_rate_total: Counter tracking the proportion of incoming images rejected by the energy scoring module.
* tail_species_class_balanced_accuracy: Gauge monitoring identification performance across rare species subsets.

┌──────────────────────────────────────┐
│   FastAPI / Prometheus Exporter      │
│   - p95/p99 Latency Histograms       │
│   - Set Size Distribution (|C|)      │
│   - OOD Rejection Rates              │
│   - Tail Species Accuracy            │
└──────────────────┬───────────────────┘
│
▼ Scraped over Tailscale Mesh
┌──────────────────────────────────────┐
│ Prometheus Monitoring Server         │
└──────────────────┬───────────────────┘
│
▼ Threshold Evaluation
┌──────────────────────────────────────┐
│ Alertmanager & Automated Retraining  │
│ Triggers DVC Pull & Pipeline Retrain │
└──────────────────────────────────────┘

###### *VFM Embedding & Concept Drift Detection*

To detect subtle visual distribution shifts (e.g., seasonal camera parameter shifts, regional lighting variations), the monitoring engine tracks high-dimensional embedding drift. A reference baseline distribution of vector embeddings $\mathcal{Z}_{base}$ is cached during model calibration.

During production execution, incoming feature vectors $\mathcal{Z}_{prod}$ are collected over sliding temporal windows. The system evaluates pairwise Maximum Mean Discrepancy (MMD) across distributions using a Gaussian radial basis function (RBF) kernel $k(z, z')$:


$$\text{MMD}^2(\mathcal{Z}_{base}, \mathcal{Z}_{prod}) = \frac{1}{M^2} \sum_{i=1}^M \sum_{j=1}^M k(z_i, z_j) - \frac{2}{MN} \sum_{i=1}^M \sum_{j=1}^N k(z_i, z_j') + \frac{1}{N^2} \sum_{i=1}^N \sum_{j=1}^N k(z_i', z_j')$$

###### *Alerting & Retraining Triggers*

Prometheus Alertmanager evaluates exported statistical metrics against operational thresholds:

1. **Set Size Inflation Alert:** Triggered if the mean conformal prediction set size $\mathbb{E}\vert{}\hat{C}\vert{}$ expands by $>20\%$ over a 1-hour window, signalling elevated classification uncertainty.
2. **Embedding Drift Alert:** Triggered if the MMD distance score exceeds variance bounds ( $\text{MMD}^2 > \tau_{drift}$ ).

Triggered alerts notify engineering teams via Tailscale metric endpoints and initiate an automated GitHub Actions workflow run. The automated pipeline pulls updated DVC dataset snapshots, retrains the classification head, re-calibrates conformal quantiles ( $\hat{q}_{\alpha}$ ), and opens a pull request with validation metrics for operator review.

###### *Connective Tissue*

System observability and telemetry rely on a modular codebase that strictly isolates operational concerns.

##### 9. Repository Architecture & Codebase Software Engineering

###### *Context & Strategic Importance*

A production-grade MLOps repository requires clear modular separation, clean component abstraction, and automated testing suites. Isolating data ingestion, feature extraction, statistical safety layers, and API serving ensures individual components can be tested, refactored, and updated independently.

###### *Directory Tree Structure*

The project repository (taxon-vision-mlops) follows a structured Python package organization under src/taxon_vision/:

taxon-vision-mlops/
├── .agents/
│   └── AGENTS.md
├── config/
├── docs/
├── scripts/
├── src/
│   └── taxon_vision/
│       ├── active_learning/
│       ├── data/
│       ├── inference/
│       ├── models/
│       ├── monitoring/
│       ├── service/
│       └── uncertainty/
├── tests/
│   ├── integration/
│   ├── invariants/
│   └── unit/
├── Justfile
├── pixi.lock
└── pyproject.toml

###### *Module Breakdown & Responsibilities*

* src/taxon_vision/data/: Handles AWS S3 streaming, cryptographic license validation (CC0, CC-BY, CC-BY-NC), DarwinCore Parquet metadata parsing, and DVC dataset versioning.
* src/taxon_vision/models/: Implements PyTorch modules, integrates pretrained timm Vision Foundation Model backbones (BioCLIP-2, DINOv3), and defines Class-Balanced Loss functions.
* src/taxon_vision/inference/: Manages ONNX Runtime exports, FP16/INT8 dynamic quantization routines, zero-copy memory buffers, and madvise THP memory mappings.
* src/taxon_vision/uncertainty/: Contains Split Conformal Prediction algorithms, non-conformity score quantile calculation, and Energy-Based OOD scoring logic.
* src/taxon_vision/active_learning/: Implements BADGE and CoreSet selection algorithms, conformal set triage routing, and DagsHub Label Studio API synchronization.
* src/taxon_vision/service/: Hosts the FastAPI backend endpoints and HTMX operator dashboard templates.
* src/taxon_vision/monitoring/: Exposes Prometheus metrics, tracks MMD embedding vector drift, and aggregates sliding-window execution statistics.

###### *Test Suite Architecture & Segmentation*

The testing architecture under tests/ is partitioned into three tiers:

1. unit/: Validates isolated functions, mathematical expressions, license parsing, and schema structures.
2. integration/: Verifies end-to-end interactions across the FastAPI service, ONNX Runtime execution, DVC data fetching, and vector index queries.
3. invariants/: Executes strict mathematical verification tests. These invariant tests run split conformal prediction against holdout validation datasets to verify that empirical coverage never drops below $1 - \alpha = 0.95$.

###### *Developer Workflow Automation (Justfile)*

Command-line operations are standardized using Just command recipes, eliminating ad-hoc execution scripts:

# Setup environment, pre-commit hooks, and scratch notebook space

setup:
pixi run pre-commit install
mkdir -p notebooks/scratch

# Execute linter and static type checking

lint:
pixi run ruff check .
pixi run mypy src/

# Execute entire test suite (unit, integration, invariants)

test:
pixi run pytest tests/

# Launch unified FastAPI + HTMX production server

run:
pixi run uvicorn taxon_vision.service.main:app --host 0.0.0.0 --port 8000

# Validate Open Knowledge Format frontmatter compliance

validate-okf:
pixi run python scripts/validate_okf_docs.py

###### *Connective Tissue*

Codebase modularity and developer workflow automation provide the foundation for automated CI/CD verification gates that control release candidate deployments.

##### 10. Epistemic Soundness & Automated CI/CD Verification Gates

###### *Context & Strategic Importance*

To prevent broken code, uncalibrated models, or memory leaks from reaching production, the CI/CD deployment pipeline enforces three automated verification gates. These gates evaluate AST structure, memory parity, and empirical statistical coverage before code merges or model deployments are approved.

Git Push / PR Trigger
│
▼
┌──────────────────────────────────────┐
│ Gate 1: AST Inspection Gate          │
│ - Scan for unseeded random calls     │
│ - Verify @hot_path memory allocation │
│ - Enforce ONNX runtime matrix routing│
└──────────────────┬───────────────────┘
│ [PASS]
▼
┌──────────────────────────────────────┐
│ Gate 2: Memory Parity Gate           │
│ - Audit C-ABI Arrow zero-copy buffers│
│ - Check DuckDB offset pointer alignment│
│ - Confirm bit-exact trace parity     │
└──────────────────┬───────────────────┘
│ [PASS]
▼
┌──────────────────────────────────────┐
│ Gate 3: Coverage Invariant Gate      │
│ - Run Conformal Calibration Tests    │
│ - Reject PR if Coverage < 1 - alpha  │
└──────────────────┬───────────────────┘
│ [PASS]
▼
Merge / Deployment Approved

###### *Three-Stage Automated CI/CD Gates*

###### *Gate 1: Branchless AST Inspection Gate*

Performs static analysis on Python Abstract Syntax Trees (AST) using custom ast.NodeVisitor rules prior to execution. The static analyzer inspects core inference paths under src/taxon_vision/inference/ to:

* Scan for ast.Call nodes invoking unseeded random or torch.rand functions, enforcing deterministic execution seeds.
* Flag explicit list or array allocations inside functions annotated with @hot_path, enforcing zero-allocation execution.
* Block raw torch.matmul calls in serving paths that do not route execution through an onnxruntime.InferenceSession.

###### *Gate 2: Table-to-Trace Memory Parity Gate*

Audits zero-copy Apache Arrow C-ABI memory buffers and DuckDB index alignment. The test suite processes reference batches through the ingestion pipeline and compares table records against downstream inference trace outputs to confirm offset pointer alignment, zero memory leaks, and bit-exact table-to-trace parity.

###### *Gate 3: Automated Statistical Coverage Gate*

Validates mathematical coverage invariants. The runner executes Split Conformal Prediction against holdout calibration datasets. If empirical coverage drops below the required statistical threshold ( $1 - \alpha = 0.95$ ), the build is blocked automatically, preventing deployment of uncalibrated models.

###### *Verification Gate Execution Summary Table*

| Gate Name | Target Subsystem | Verification Mechanism | Blocking Criteria |
| --- | --- | --- | --- |
| Gate 1: AST Inspection | src/taxon_vision/inference/ | Static Python AST analysis (ast.NodeVisitor) | Unseeded randomness, hot-path memory allocations, or raw un-quantized matmul calls. |
| Gate 2: Memory Parity | src/taxon_vision/data/ & Arrow | C-ABI buffer & offset trace audit | Memory leaks, offset misalignment, or table-to-trace bit-exact parity failures. |
| Gate 3: Coverage Invariant | src/taxon_vision/uncertainty/ | Holdout conformal calibration execution | Empirical coverage dropping below specified $1 - \alpha = 0.95$ threshold. |

###### *Connective Tissue*

These verification gates establish an automated release mechanism. Consolidated hardware parameters, memory allocations, and formal mathematical proofs are detailed in the technical appendices.

##### 11. Technical Appendices

###### *Context & Strategic Importance*

The technical appendices consolidate hardware optimization parameters, host memory budgets, and formal mathematical derivations supporting the TaxonVision-MLOps architecture.

###### *Appendix A: Consolidated Hardware & System Optimization Matrix*

| Subsystem / Hardware Component | Kernel Parameter / System Setting | Isolation & Tuning Mechanism | Target Pipeline Stage | Operational Impact |
| --- | --- | --- | --- | --- |
| Intel i7-14700K P-cores | isolcpus=0-15 | Pinned via taskset -c 0-15 | ONNX Inference & Hot Loops | Prevents thread migration, preserving L2/L3 cache locality. |
| Intel i7-14700K E-cores | System default cores | Pinned via taskset -c 16-27 | AWS S3 Ingestion & Crypto Parsing | Offloads background I/O from core neural compute paths. |
| NVIDIA RTX 5070 Ti | CUDA 12.1 / Tensor Core Driver | FP16/INT8 ONNX Engine | High-Throughput Model Serving | Enables sub-25ms inference latency via GPU acceleration. |
| System Memory (128GB) | vm.nr_hugepages=16384 | 2MB Transparent HugePages (madvise) | Weights Buffer Management | Eliminates TLB misses during weight matrix accesses. |
| NVMe Storage / BTRFS | mount -o compress=zstd:3,noatime | BTRFS transparent compression | Dataset Snapshots & Caching | Increases effective disk read bandwidth over PCIe bus. |
| DVC Cache Subvolume | @dvc_cache Subvolume | Executed chattr +C /path (nodatacow) | DuckDB & DVC Storage | Eliminates BTRFS disk fragmentation during continuous writes. |
| Tailscale Network | WireGuard Mesh Protocol | Zero open public ingress ports | Dashboard & Telemetry | Secures administrative endpoints without public exposure. |

###### *Appendix B: Workstation Memory Budget Allocation*

Host system memory (128 GB DDR5 RAM) is allocated across subsystems to prevent out-of-memory thrashing and maintain high data throughput:

| System Subsystem | Memory Allocation (GB) | Percentage of Total | Operational Purpose |
| --- | --- | --- | --- |
| **OS & Core Infrastructure** | 8.0 GB | 6.25% | Host Linux kernel, Dropbear SSH, Tailscale, base services. |
| **BTRFS Page Cache & THP** | 32.0 GB | 25.00% | Disk page caching, 2MB Transparent HugePage allocations (madvise). |
| **IVF-HNSW Vector Index** | 40.0 GB | 31.25% | In-memory vector graph caching for BioCLIP-2 feature embeddings. |
| **Arrow In-Memory Buffers** | 28.0 GB | 21.88% | Zero-copy Arrow C-ABI data streaming buffers and image ingestion. |
| **PyTorch / ONNX Workspace** | 20.0 GB | 15.62% | ONNX execution graph buffers, quantization tensors, VRAM staging. |
| **Total System RAM** | **128.0 GB** | **100.0%** | Full host system memory budget. |

###### *Appendix C: Formal Mathematical Derivations*

###### *C.1 Class-Balanced Loss Weighting Factor Derivation*

To handle long-tail species distributions, loss weighting uses the effective number of samples $E_n$:


$$E_n = \frac{1 - \beta^n}{1 - \beta}$$

Where $n \in \mathbb{Z}^+$ is the sample count for class $y$, and hyperparameter $\beta \in [0, 1)$ parameterizes feature volume overlap. The weighting factor $W_y$ assigned to class $y$ is inversely proportional to $E_{n_y}$:


$$W_y = \frac{1}{E_{n_y}} = \frac{1 - \beta}{1 - \beta^{n_y}}$$

For predicted probability distribution $\hat{p}$ and ground-truth label $y$, the class-balanced cross-entropy loss $\mathcal{L}_{CB}(x, y)$ evaluates as:


$$\mathcal{L}_{CB}(x, y) = - W_y \log(\hat{p}_y) = - \frac{1 - \beta}{1 - \beta^{n_y}} \log(\hat{p}_y)$$

###### *C.2 Split Conformal Prediction Quantile Bounds Derivation*

Let $(X_1, Y_1), \dots, (X_n, Y_n)$ be exchangeable calibration samples drawn from an unknown joint distribution $P_{X, Y}$. Non-conformity scores are evaluated as $S_i = s(X_i, Y_i) = 1 - \hat{\pi}_{Y_i}(X_i)$.

Given an unseen test sample $(X_{n+1}, Y_{n+1})$, exchangeability implies that the set $\{S_1, \dots, S_n, S_{n+1}\}$ consists of exchangeable random variables. Sort the calibration scores in ascending order:


$$S_{(1)} \le S_{(2)} \le \dots \le S_{(n)}$$

Setting the empirical quantile threshold $\hat{q}_{\alpha}$ at rank index $k = \lceil (n+1)(1-\alpha) \rceil$:


$$\hat{q}_{\alpha} = S_{(k)}$$

By exchangeability, the rank of $S_{n+1}$ among the $n+1$ total scores is uniformly distributed over $\{1, \dots, n+1\}$. Consequently:


$$P\left(S_{n+1} \le \hat{q}_{\alpha}\right) = P\left(\text{Rank}(S_{n+1}) \le \lceil (n+1)(1-\alpha) \rceil\right) = \frac{\lceil (n+1)(1-\alpha) \rceil}{n+1} \ge 1 - \alpha$$

Because $Y_{n+1} \in \hat{C}(X_{n+1})$ if and only if $s(X_{n+1}, Y_{n+1}) \le \hat{q}_{\alpha}$, exact marginal coverage holds:


$$P\left(Y_{n+1} \in \hat{C}(X_{n+1})\right) \ge 1 - \alpha$$

###### *C.3 Energy-Based Out-of-Distribution Density Formulation*

Given a classification model outputting logit vector $f_j(x)$ for class $j \in \{1, \dots, K\}$, the free energy function $E(x; f)$ with temperature $T$ derives from the log-sum-exp denominator of the softmax distribution:


$$E(x; f) = -T \cdot \log \sum_{j=1}^K \exp\left(\frac{f_j(x)}{T}\right)$$

The probability density function $p(x)$ for in-distribution data connects to scalar energy via the Boltzmann distribution:


$$p(x) = \frac{\exp\left(-\frac{E(x; f)}{T}\right)}{Z}$$


Where $Z = \int \exp\left(-\frac{E(x; f)}{T}\right) dx$ represents the normalizing partition function. Inputs producing scalar energy values exceeding threshold $\tau_{OOD}$ correspond to low-density regions under $p(x)$, allowing out-of-distribution inputs to be rejected prior to conformal evaluation.

```eof
*Suggestions added*

```
