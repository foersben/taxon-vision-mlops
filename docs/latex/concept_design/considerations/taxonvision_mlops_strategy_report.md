---
type: Reference
title: TaxonVision-MLOps Strategy Report
status: stable
stale_after: "2027-01-01T00:00:00Z"
version: 1.0
description: Structured MLOps strategy report for TaxonVision covering backbone selection, data pipeline design, conformal prediction, and deployment on edge hardware.
tags: [strategy, mlops, architecture, report]
generated: {by: process:human-review, at: "2026-10-02T10:00:00Z"}
verified: {by: process:human-review, at: "2026-10-02T10:00:00Z"}
---

### TaxonVision-MLOps: Automated End-to-End Architectural and Operational Strategy Report

##### 1. Executive Summary & Problem Formulation

###### *Context & Strategic Importance*
Fine-grained biological species identification at a global scale represents both a formidable computational challenge and an operational imperative for biodiversity monitoring. Citizen-science platforms generate vast, continuous streams of unstructured visual observations. However, translating raw field photographs into verified ecological intelligence requires transitioning away from ad-hoc, unmonitored notebook experiments toward an automated, production-grade Machine Learning Operations (MLOps) platform.

Notebook-bound models suffer from non-reproducible environments, parameter drift, silent degradation under distribution shifts, and fragile handoffs. The TaxonVision-MLOps platform resolves these operational bottlenecks by establishing an automated, versioned, monitored, and statistically bounded end-to-end pipeline. By framing visual identification as an enterprise service, TaxonVision-MLOps ensures that deployed Vision Foundation Models (VFMs) maintain high throughput, strict regulatory data compliance, and verifiable epistemic safety over long-term operational horizons.

┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                  TAXONVISION-MLOPS                                     │
│         Automated End-to-End MLOps Pipeline for Species Identification from Photos     │
│              Citizen Science (iNaturalist / GBIF) • Conformal Uncertainty             │
│             Production-Grade Engineering • Dual-Target Pixi (GPU / CPU)               │
└────────────────────────────────────────────────────────────────────────────────────────┘

###### *Citizen-Science Photograph Ingestion & Licensing*
Global biodiversity portals such as iNaturalist and the Global Biodiversity Information Facility (GBIF) aggregate continuous streams of species observations worldwide. Combined, these ecosystems host approximately 270 million raw observations, of which roughly 170 million have achieved "research grade" status through community validation [Source: iNaturalist Open Data Registry, AWS]. Processing this high-volume stream requires strict cryptographic and legal compliance safeguards at the ingestion edge.

To enforce open-access mandates, the ingestion engine inspects incoming observation metadata headers prior to staging objects locally. A cryptographic license filtering module evaluates record metadata against allowed open-access definitions, enforcing strict inclusion filters for Creative Commons Zero (CC0), Creative Commons Attribution (CC-BY), and Creative Commons Attribution-NonCommercial (CC-BY-NC). Any incoming observation tagged with restrictive or commercial licenses (e.g., CC-BY-ND or CC-BY-SA restricted variants) is explicitly dropped at the network boundary. Furthermore, to comply with GBIF and iNaturalist scientific lineage requirements, an automated attribution preservation mechanism extracts observation GUIDs, contributor ORCIDs/usernames, and original license headers, serializing them alongside downstream feature vectors and prediction logs to maintain lineage traceability.

###### *The MLOps Paradigm Shift (Level 0 to Level 2/4)*
To establish an enterprise architecture, TaxonVision-MLOps adapts the industry-standard MLOps maturity framework, structuring system operational evolution across defined maturity tiers:

* **Level 0 (Manual Process):** Represents ad-hoc ML engineering characterized by manual handoffs between data science and operations teams. Models are trained in isolated Jupyter notebooks, evaluated on static CSV exports, and manually serialized to disk before operational teams wrap them in monolithic APIs. This pattern collapses under operational load due to team communication gaps, environment drift (where underlying CUDA libraries, Python packages, or system dependencies diverge between development and deployment), and poor scaling when attempting to support thousands of distinct taxa.
* **Level 1 (Automated Pipeline):** Establishes Continuous Training (CT) and automated workflow orchestration. The scope at Level 1 is constrained to establishing a closed-chain baseline on **~10 well-represented taxa** (e.g., benchmarked against target subsets like CUB-200 or Oxford Flowers) to validate the end-to-end ingestion, feature extraction, and serving path before scaling. Retraining pipelines are executed automatically upon the arrival of new validated data snapshots or source code commits, serving updated weights to a unified prediction service.
* **Level 2 (CI/CD Automation):** Expands automation from the model execution level to the pipeline build process itself. Continuous Integration (CI) and Continuous Deployment (CD) pipelines automate the building, testing, dynamic quantization, and deployment of the entire modeling code infrastructure, enabling rapid iteration over model backbones and hyperparameter configurations.
* **Level 3 (Feedback Loops & Observability):** Integrates automated production feedback collection. Post-prediction ground-truth labels validated by community experts are automatically ingested to measure production performance degradation, track historical label adjustments, and drive continuous data-engine iteration. This feedback loop acts as the operational bridge to Level 4.
* **Level 4 (Production-Grade Safety & Robustness):** Represents TaxonVision’s target operating state. At this level, the service incorporates Split Conformal Prediction for distribution-free error bounds, Energy-Based Out-of-Distribution (OOD) rejection, active learning human-in-the-loop triage, and Class-Balanced Loss functions designed for extreme long-tail species frequency distributions.

| Maturity Level | Automated Infrastructure | Core Operational Gain |
| :--- | :--- | :--- |
| **Level 0: Manual** | None (Manual handoffs, isolated scripts/notebooks) | Low initial complexity for localized algorithm prototyping. |
| **Level 1: Pipeline** | Automated pipeline, data/model validation, CT triggers (~10 taxa baseline) | Continuous training loop executing automatically on new data arrivals. |
| **Level 2: CI/CD** | Automated CI/CD for pipeline build, test, ONNX export, and deployment | Rapid, repeatable, and automated modeling pipeline code releases. |
| **Level 3: Feedback Loops** | Production feedback collection, ground-truth label ingestion, drift monitoring | Automated tracking of production metrics to drive data-engine iterations. |
| **Level 4: Production Safety** | Conformal prediction, OOD detection, Active Learning, Class-Balanced Loss | Distribution-free coverage guarantees ( $1-\alpha$ ), epistemic safety, and long-tail robustness. |

---

##### 2. Bare-Metal Infrastructure & Hardware Topography

###### *Context & Strategic Importance*
Continuous training and low-latency inference using Vision Foundation Models incur prohibitive operational costs when deployed naively on public cloud infrastructure. Running high-density VFM workloads requires bare-metal hardware access to eliminate hypervisor overhead, control physical core allocations, enforce deterministic memory layouts, and maximize PCI Express bandwidth between system RAM and discrete acceleration silicon.

###### *Host Silicon & Memory Topography*
The TaxonVision physical host is built on an isolated, bare-metal compute platform configured as follows:
*   **CPU:** Intel Core i7-14700K featuring a heterogeneous CPU architecture (8 Performance-cores / 16 threads; 12 Efficiency-cores / 12 threads; 28 total execution threads).
*   **RAM:** 128 GB DDR5 system memory operating at 5600 MT/s across dual channels.
*   **GPU:** Discrete NVIDIA RTX 5070 Ti (16 GB GDDR6X VRAM) hosting dedicated Tensor Cores for neural acceleration.
*   **Storage:** Direct-attached 4 TB NVMe PCIe 4.0 solid-state storage.

To prevent thread migration and cache invalidation across heterogeneous CPU cores during concurrent execution, the host runtime enforces strict NUMA-style thread pinning using Linux affinity masks (`taskset`). High-frequency neural execution paths, PyTorch hot loops, ONNX Runtime execution contexts, and GPU driver interrupt processing are pinned strictly to **P-cores (Cores 0-15 via `taskset -c 0-15`)**. This preserves L2 (2 MB per P-core) and shared L3 (33 MB) cache allocations. Conversely, background asynchronous tasks-including AWS S3 HTTP block streaming, cryptographic license header parsing, image decoding, DuckDB metadata indexing, and background BTRFS writebacks-are assigned strictly to **E-cores (Cores 16-27 via `taskset -c 16-27`)**. This core isolation eliminates context switching across the asymmetric silicon topography.

###### *Filesystem Engineering (BTRFS + zstd)*
Storage I/O performance represents a common bottleneck when reading millions of small image files and dataset chunks. The underlying NVMe drive is partitioned with block-level LUKS encryption for data protection at rest, managed by a BTRFS filesystem configured with transparent `zstd:3` compression (`mount -o compress=zstd:3,noatime`). Transparent compression drastically compresses raw binary dataset chunks while increasing effective read bandwidth across the PCIe bus.

To prevent BTRFS performance degradation caused by Copy-on-Write (CoW) metadata expansion during continuous SQLite/DuckDB index writes and heavy Data Version Control (DVC) caching, a dedicated subvolume (`@dvc_cache`) is initialized with CoW disabled (`nodatacow` attribute):

```bash
# Initialize dedicated BTRFS subvolume and enforce NOCOW attribute
sudo btrfs subvolume create /mnt/storage/@dvc_cache
sudo chattr +C /mnt/storage/@dvc_cache
```

Disabling Copy-on-Write (`chattr +C`) eliminates file fragmentation and I/O wait spikes during continuous, concurrent database writes and DVC chunk storage operations.

###### *Zero-Ingress Network & Security Architecture*
System security is maintained by enforcing a strict zero-ingress network posture. The physical host exposes zero open public inbound ports and operates behind external boundary firewalls. Remote access for administrative maintenance, execution management, and telemetry collection relies on two secure overlay mechanisms:
1.  **Dropbear SSH:** A lightweight Dropbear SSH server is embedded directly within the initial RAM disk (initramfs) boot sequence. This allows system administrators to perform pre-boot remote system unlocking of LUKS-encrypted drives over the local network prior to loading the main OS kernel.
2.  **Tailscale Zero-Trust Mesh Network:** Operational access to internal prediction services, FastAPI endpoints, the HTMX administration panel, and Prometheus monitoring ports is restricted to authenticated nodes on a private Tailscale zero-trust mesh network (using encrypted WireGuard point-to-point tunnels), bypassing the public internet entirely.

###### *Hermetic Environment Isolation (Pure Pixi)*
To prevent host OS dependency pollution and eliminate environment drift between local development workstations and CI/CD runners, system environments are managed using Pixi (`pyproject.toml`, `pixi.lock`). Pixi isolates the underlying CUDA 12.1 toolkit runtime, C++ build dependencies, system libraries, and PyTorch packages entirely within user space. External virtual environment tools (`venv`) are explicitly blocked, ensuring that local developer environments, execution workspaces, and self-hosted GitHub Actions runners (`self-hosted, gpu, rtx5070ti`) execute on mathematically identical toolchains.

---

##### 3. Open-Source MLOps Toolchain & Governance

###### *Context & Strategic Importance*
Building a production-grade MLOps platform without expensive proprietary cloud services requires integrating open-source tools into a cohesive operational system. Selecting transparent open-source components guarantees that dataset versioning, experiment tracking, model registry storage, and active learning interfaces remain reproducible, vendor-agnostic, and cost-effective.

###### *The Toolchain Nervous System (DagsHub, GitHub, Hugging Face)*
The platform architecture coordinates three primary platforms that function as a unified management plane:
* **DagsHub:** Acts as the central management plane. Connected directly to the code repository, DagsHub provides a managed MLflow experiment tracking instance to log training hyperparameters, loss metrics, and validation curves. It supplies an S3-compatible remote storage backend for DVC dataset snapshots and hosts an integrated Label Studio workspace for human-in-the-loop active learning triage.
* **GitHub:** Serves as the single source of truth for repository source code, pipeline orchestration logic, agent rules, and CI/CD automation. GitHub Actions routes execution to the local bare-metal server operating as a self-hosted GPU runner (tagged `self-hosted, gpu, rtx5070ti`) for hardware-accelerated testing, quantization, and benchmarking routines.
* **Hugging Face Hub:** Functions as the downstream Model Registry. The pipeline pulls upstream pretrained Vision Foundation Models (e.g., BioCLIP-2, DINOv3) from Hugging Face and pushes benchmarked, INT8/FP16 quantized ONNX model artifacts back to HF Hub spaces for versioned deployment release.

###### *Repository Governance & Open Knowledge Format (OKF)*
System development and automated agent interactions are governed by a 12-role agent matrix configured in `.agents/AGENTS.md`. System documentation, architectural decision records, and technical specifications adhere strictly to the Google Open Knowledge Format (OKF v0.2) schema stored under `docs/`. This governance framework enforces strict boundaries across data ingestion, model optimization, security auditing, and statistical verification.

###### *Toolchain Topology Map*

| Pipeline Stage | Open-Source Tool / Protocol | Hosting & Compute Backend | Primary Operational Responsibility |
| :--- | :--- | :--- | :--- |
| Data Versioning | DVC (Data Version Control) | DagsHub S3-Compatible Remote Storage | Tracking large image TAR archives, Parquet metadata, and cache splits. |
| Experiment Tracking | MLflow | DagsHub Managed Tracking Server | Logging parameters, loss curves, Top-1/Top-5 accuracy, and p95 latency. |
| Model Registry | Hugging Face Hub / ONNX | Hugging Face Hub Remote Registry | Managing upstream VFM backbones and deploying optimized ONNX artifacts. |
| CI/CD Compute | GitHub Actions Runner | Local Bare-Metal Host (rtx5070ti) | Executing AST code audits, memory parity checks, and model evaluation. |
| Serving Dashboard | FastAPI + HTMX | Local Process (Bare-Metal Host) | Rendering JS-build-less operator control panels and serving REST endpoints. |
| Telemetry & Alerts | Prometheus Client | Bare-Metal Host / Tailscale Mesh | Exporting latency histograms, conformal set distributions, and drift metrics. |

---

##### 4. Vision RAG & Multimodal Knowledge Graph (MMKG)

###### *Context & Strategic Importance*
In visual ecology, standard text-centric Retrieval-Augmented Generation (RAG) concepts must be transformed into visual and taxonomic equivalents. Because running heavy Vision Foundation Models over all 170M "research grade" iNaturalist images is computationally infeasible for a single RTX 5070 Ti, the ingestion pipeline curates a *Reference Exemplar Catalog* for vector indexing.

###### *Reference Exemplar Cataloging & Visual Entity Linking*
Rather than indexing every incoming photograph, the pipeline aggregates DarwinCore metadata to identify the top 50 quintessential, high-quality, community-validated images per species. These reference images undergo zero-shot visual entity linking ("visual chunking"). Lightweight detection models scan the broad environmental scene (the "parent context") to extract bounding boxes around the target organisms (the "child visual chunks"). This decouples the target organism from background clutter while preserving parent metadata links.

###### *DarwinCore Knowledge Graph Indexing*
Extracted child visual chunks are indexed within a structured Linnaean hierarchy:
$$ \text{Kingdom} \longrightarrow \text{Phylum} \longrightarrow \text{Class} \longrightarrow \text{Order} \longrightarrow \text{Family} \longrightarrow \text{Genus} \longrightarrow \text{Species} $$

Each curated child chunk is passed through a frozen Vision Foundation Extractor (such as BioCLIP-2) to generate dense 512-dimensional visual embeddings.

###### *Single-Stage Metadata-Filtered HNSW Search*
A major risk in biodiversity vector search is *convergent evolution*-where unrelated species on different continents appear visually identical. If a user queries a European species, and the top 100 raw vector matches are all visually identical North American species, applying a geographical filter *after* the vector search (post-filtering) will result in an empty retrieval set.

To resolve this, TaxonVision-MLOps utilizes **Single-Stage Metadata-Filtered HNSW search**. The vector database applies DarwinCore spatial-temporal boolean masks (GPS bounding boxes, observation month) *during* the graph traversal. This ensures the system retrieves the top- $k$ ( $k=5$ ) visually similar candidate images that are *also* biogeographically possible, maintaining high recall without sacrificing latency.

###### *VLM Synthesis & Contextual Explainability*
Once the top- $k$ reference images are retrieved, they are not simply surfaced as classes. The Vision RAG pipeline passes the user's query image, the structured DarwinCore metadata, and the retrieved parent-child reference crops directly into a lightweight Vision-Language Model (VLM). This generates comparative, context-aware explainability for the human operator via the HTMX dashboard, outlining specific morphological differences between the query and the retrieved exemplars.

---

##### 5. Model Benchmarking, ONNX Quantization & High-Throughput Serving

###### *Context & Strategic Importance*
Deploying Vision Foundation Models to production requires balancing classification accuracy against real-world execution costs. Systematic Pareto optimization identifies the optimal backbone architecture across latency, memory, and accuracy constraints.

###### *Multi-Backbone Pareto Optimization*
To satisfy Level 2 MLOps requirements, candidate vision backbones are evaluated under identical data splits using the `timm` (PyTorch Image Models) interface. Evaluated backbones include **BioCLIP-2**, **DINOv3**, **DINOv2**, **MobileNetV4**, and **EfficientNet**. Multi-metric Pareto analysis evaluates Top-1 Accuracy, Top-5 Accuracy, Floating-Point Operations (GFLOPs), VRAM Footprint (MB), and p95 Inference Latency (ms).

###### *ONNX Export & Dynamic Quantization*
PyTorch models are exported to Open Neural Network Exchange (ONNX) graph representations to decouple execution from the PyTorch Python runtime. The ONNX Runtime optimization pipeline applies two precision reduction strategies:
* **FP16 (Float16) Execution:** Maps operations to NVIDIA RTX 5070 Ti Tensor Cores for accelerated matrix multiplication.
* **INT8 (Integer 8) Dynamic Quantization:** Quantizes weight matrices to 8-bit integers, shrinking artifact storage and memory bandwidth requirements.

At the host system level, memory management utilizes explicit memory advice flags (`madvise` paired with Linux Transparent HugePages / THP). System memory allocations for model weight buffers use contiguous 2 MB pages rather than default 4 KB pages, eliminating Translation Lookaside Buffer (TLB) miss penalties during execution, with the engineering goal of targeting sub-25ms zero-allocation execution paths. This threshold is empirically verified during automated CI/CD load testing.

###### *Target Evaluation Matrix (Auto-Generated via CI/CD)*
*Note: The following matrix defines the evaluation schema. The automated CI/CD pipeline executes `scripts/benchmark_pareto.py` to empirically populate these metrics directly on the target RTX 5070 Ti hardware.*

| Model Backbone | Precision | Target Latency (ms) | Target VRAM | Empirical Artifact Size | Empirical GFLOPs | Empirical Top-1 | Empirical Top-5 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| BioCLIP-2 | FP32 / INT8 | < 30ms | < 4GB | [Auto-Generated] | [Auto-Generated] | [Auto-Generated] | [Auto-Generated] |
| DINOv3 | FP16 / INT8 | < 25ms | < 3GB | [Auto-Generated] | [Auto-Generated] | [Auto-Generated] | [Auto-Generated] |
| MobileNetV4 | FP32 / INT8 | < 10ms | < 1GB | [Auto-Generated] | [Auto-Generated] | [Auto-Generated] | [Auto-Generated] |

###### *Operator Dashboard & Serving Architecture*
The serving framework couples a FastAPI backend with a lightweight, JavaScript-build-less HTMX frontend dashboard. Operators can monitor real-time p95/p99 latency, view streaming prediction set statistics, inspect low-latency Grad-CAM interpretability heatmaps, and trigger model retraining directly from the web panel.

---

##### 6. Statistical Safety & Conformal Uncertainty Quantification

###### *Context & Strategic Importance*
Standard deep learning classifiers output class probability distributions using the softmax function: $ \hat{\pi}_k(x) = \frac{\exp(f_k(x))}{\sum_j \exp(f_j(x))} $. However, raw softmax scores are uncalibrated and tend to be overconfident under distribution shifts. To enforce epistemic safety, TaxonVision-MLOps integrates Split Conformal Prediction, converting single-class point predictions into statistically guaranteed prediction sets.

###### *Split Conformal Prediction Mathematical Framework*
Split Conformal Prediction outputs dynamic prediction sets $ \hat{C}(X_{test}) \subseteq \mathcal{Y} $ that contain the true ground-truth class $ Y_{test} $ with a user-defined coverage probability $ 1 - \alpha $ (e.g., 95% coverage for $ \alpha = 0.05 $), without making parametric distribution assumptions.

1.  **Holdout Calibration Split:** A holdout calibration dataset $ \mathcal{D}_{cal} = \{(x_i, y_i)\}_{i=1}^n $, completely disjoint from model training data, is reserved.
2.  **Non-Conformity Scoring:** For each calibration pair $ (x_i, y_i) $, the non-conformity score $ s_i $ measures model error associated with the true label $ y_i $:
    $$ s_i = 1 - \hat{\pi}_{y_i}(x_i) $$
3.  **Quantile Estimation:** Calibration scores are sorted: $ S_{(1)} \le S_{(2)} \le \dots \le S_{(n)} $. The empirical quantile threshold $ \hat{q}_{\alpha} $ is calculated at index $ k = \lceil (n+1)(1-\alpha) \rceil $:
    $$ \hat{q}_{\alpha} = \text{Quantile}\left(s_1, \dots, s_n; \, \frac{\lceil (n+1)(1-\alpha) \rceil}{n}\right) = S_{(\lceil (n+1)(1-\alpha) \rceil)} $$
4.  **Prediction Set Construction:** For an unseen test image $ X_{test} $, the prediction set $ \hat{C}(X_{test}) $ includes all taxa whose non-conformity score falls below $ \hat{q}_{\alpha} $:
    $$ \hat{C}(X_{test}) = \left\{ y \in \mathcal{Y} : s(X_{test}, y) \le \hat{q}_{\alpha} \right\} = \left\{ y \in \mathcal{Y} : \hat{\pi}_y(X_{test}) \ge 1 - \hat{q}_{\alpha} \right\} $$

This formulation guarantees finite-sample, distribution-free coverage at confidence level $ 1 - \alpha $:
$$ P\left(Y_{test} \in \hat{C}(X_{test})\right) \ge 1 - \alpha $$

###### *Energy-Based Out-Of-Distribution (OOD) Detection*
Before passing query features to the conformal prediction layer, the pipeline evaluates an Energy-Based OOD score to flag non-organism photos (e.g., vehicles, landscapes). The scalar energy function $ E(x; f) $ maps logit outputs $ f(x) $ using temperature parameter $ T $:
$$ E(x; f) = -T \cdot \log \sum_{j=1}^K \exp\left(\frac{f_j(x)}{T}\right) $$
Inputs yielding high scalar energy values ($ E(x; f) > \tau_{OOD} $) correspond to low probability density regions under the training distribution and are rejected prior to conformal evaluation.

###### *Class-Balanced Loss for Long-Tail Taxa*
To address class imbalance during head fine-tuning for rare species, training objectives incorporate Class-Balanced Loss based on the effective number of samples $ E_n $:
$$ E_n = \frac{1 - \beta^n}{1 - \beta} $$
Where $ n \in \mathbb{Z}^+ $ is the absolute sample count for a given species, and hyperparameter $ \beta \in [0, 1) $ scales feature space overlap. The class-balanced cross-entropy loss $ \mathcal{L}_{CB}(x, y) $ is formulated as:
$$ \mathcal{L}_{CB}(x, y) = - \frac{1 - \beta}{1 - \beta^{n_y}} \log(\hat{p}_y) $$

---

##### 7. Active Learning, Human-In-The-Loop Triage & Taxonomic Mutations

###### *Conformal-Driven Triage Routing*
The size of the conformal prediction set $ |\hat{C}| $ provides an automated routing threshold:
* **Single-Element Sets (** **$ |\hat{C}| = 1 $** **):** The identification is statistically decisive. Automatically accepted.
* **Empty Prediction Sets (** **$ |\hat{C}| = 0 $** **):** Indicates high input novelty or extreme out-of-distribution features. Routed to expert novelty queues.
* **Oversized Prediction Sets (** **$ |\hat{C}| > k_{max} $** **):** Indicates high classification ambiguity. Diverted to human expert review queues.

###### *Active Learning Query Strategies (BADGE & CoreSet)*
To optimize expert annotation efficiency, queued samples are prioritized using active learning selection algorithms rather than uniform random sampling:
* **BADGE (Batch Active learning by Diverse Gradient Embeddings):** Captures both model uncertainty and sample diversity by computing hypothetical loss gradient vectors $ g_x = \nabla_{\theta_{last}} \mathcal{L}\left(f(x; \theta), \hat{y}(x)\right) $ and applying $ k $-means++ sampling over the set of gradient embeddings.
* **CoreSet Selection:** Selects a subset of unlabelled points $ s \in S $ that minimizes the maximum distance to any unselected point in feature space: $ \min_{s \in S} \max_{i \in U} \min_{j \in s} \|z_i - z_j\|_2 $.

Prioritized samples are synchronized to the integrated DagsHub Label Studio workspace for expert adjudication.

###### *Taxonomic Mutation Reconciliation Engine*
Biological taxonomies are subject to scientific realignments (splits and lumps). TaxonVision-MLOps includes a background taxonomic reconciliation engine that executes on a scheduled cadence to sync with updated DarwinCore reference taxonomy databases, remap historical labels retroactively, and trigger DVC lineage tracking snapshots for continuous training.

---

##### 8. Observability, Telemetry & Embedding Drift Architecture

###### *Context & Strategic Importance*
Production observability requires continuous statistical telemetry tracking model decision behavior, conformal set size distributions, and visual feature drift, beyond standard HTTP uptime metrics.

###### *Prometheus Metric Instrumentation*
The FastAPI application exposes a `/metrics` endpoint instrumented with custom Prometheus counters, gauges, and histograms, scraped securely over the Tailscale Mesh:
*   **Inference Latency:** Histograms measuring ONNX graph execution (p95/p99).
*   `conformal_set_size_distribution`: Gauge tracking moving-window averages of $ |\hat{C}| $.
*   `ood_rejection_rate_total`: Counter tracking the proportion of rejected images.
*   `tail_species_class_balanced_accuracy`: Gauge monitoring performance on rare species subsets.

###### *VFM Embedding & Concept Drift Detection*
To detect subtle visual distribution shifts, the system evaluates pairwise Maximum Mean Discrepancy (MMD) across baseline and production vector distributions using a Gaussian RBF kernel:
$$ \text{MMD}^2(\mathcal{Z}_{base}, \mathcal{Z}_{prod}) = \frac{1}{M^2} \sum_{i=1}^M \sum_{j=1}^M k(z_i, z_j) - \frac{2}{MN} \sum_{i=1}^M \sum_{j=1}^N k(z_i, z_j') + \frac{1}{N^2} \sum_{i=1}^N \sum_{j=1}^N k(z_i', z_j') $$

###### *Alerting & Retraining Triggers*
Prometheus Alertmanager evaluates exported metrics. A **Set Size Inflation Alert** triggers if $ \mathbb{E}|\hat{C}| $ expands by $ >20\% $ over a 1-hour window. An **Embedding Drift Alert** triggers if $ \text{MMD}^2 > \tau_{drift} $. These alerts initiate an automated GitHub Actions workflow to pull updated DVC snapshots, retrain the classification head, re-calibrate $ \hat{q}_{\alpha} $, and open a PR for operator review.

---

##### 9. Repository Architecture & Codebase Software Engineering

###### *Directory Tree Structure*
The project repository (`taxon-vision-mlops`) follows a structured Python package organization under `src/taxon_vision/`:

```text
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
```

###### *Module Breakdown & Responsibilities*
* `src/taxon_vision/data/`: AWS S3 streaming, cryptographic license validation, DarwinCore parsing, DVC versioning, and MMKG curation.
* `src/taxon_vision/models/`: PyTorch modules, `timm` VFM backbones, and Class-Balanced Loss.
* `src/taxon_vision/inference/`: ONNX Runtime exports, quantization, and `madvise` THP memory mappings.
* `src/taxon_vision/uncertainty/`: Split Conformal Prediction algorithms and Energy-Based OOD scoring logic.
* `src/taxon_vision/active_learning/`: BADGE/CoreSet selection, conformal set triage, and DagsHub Label Studio synchronization.
* `src/taxon_vision/service/`: FastAPI backend endpoints and HTMX operator dashboard templates.
* `src/taxon_vision/monitoring/`: Prometheus metrics, MMD embedding drift, and execution statistics.

###### *Test Suite Architecture & Segmentation*
The testing architecture under `tests/` is partitioned into three tiers:
1.  `unit/`: Validates isolated functions, mathematical expressions, license parsing, and schema structures.
2.  `integration/`: Verifies end-to-end interactions across FastAPI, ONNX Runtime, DVC, and vector index queries.
3.  `invariants/`: Executes strict mathematical verification tests (e.g., verifying empirical coverage never drops below $ 1 - \alpha = 0.95 $).

###### *Developer Workflow Automation (Justfile)*
Command-line operations are standardized using `Just` command recipes, eliminating ad-hoc execution scripts:

```makefile
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
```

---

##### 10. Epistemic Soundness & Automated CI/CD Verification Gates

To prevent broken code, uncalibrated models, or memory leaks from reaching production, the CI/CD deployment pipeline enforces three automated verification gates before code merges or deployments are approved.

| Gate Name | Target Subsystem | Verification Mechanism | Blocking Criteria |
| :--- | :--- | :--- | :--- |
| **Gate 1: AST Inspection** | `src/taxon_vision/inference/` | Static Python AST analysis (`ast.NodeVisitor`) | Unseeded randomness, hot-path memory allocations, or raw un-quantized matmul calls. |
| **Gate 2: Memory Parity** | `src/taxon_vision/data/` & Arrow | C-ABI buffer & offset trace audit | Memory leaks, offset misalignment, or table-to-trace bit-exact parity failures. |
| **Gate 3: Coverage Invariant** | `src/taxon_vision/uncertainty/` | Holdout conformal calibration execution | Empirical coverage dropping below specified $ 1 - \alpha = 0.95 $ threshold. |

---

##### 11. Technical Appendices

###### *Appendix A: Consolidated Hardware & System Optimization Matrix*

| Subsystem / Hardware Component | Kernel Parameter / System Setting | Isolation & Tuning Mechanism | Target Pipeline Stage | Operational Impact |
| :--- | :--- | :--- | :--- | :--- |
| Intel i7-14700K P-cores | `isolcpus=0-15` | Pinned via `taskset -c 0-15` | ONNX Inference & Hot Loops | Prevents thread migration, preserving L2/L3 cache locality. |
| Intel i7-14700K E-cores | System default cores | Pinned via `taskset -c 16-27` | AWS S3 Ingestion & Crypto Parsing | Offloads background I/O from core neural compute paths. |
| NVIDIA RTX 5070 Ti | CUDA 12.1 / Tensor Core Driver | FP16/INT8 ONNX Engine | High-Throughput Model Serving | Enables sub-25ms inference latency via GPU acceleration. |
| System Memory (128GB) | `vm.nr_hugepages=16384` | 2MB Transparent HugePages (`madvise`) | Weights Buffer Management | Eliminates TLB misses during weight matrix accesses. |
| NVMe Storage / BTRFS | `mount -o compress=zstd:3,noatime` | BTRFS transparent compression | Dataset Snapshots & Caching | Increases effective disk read bandwidth over PCIe bus. |
| DVC Cache Subvolume | `@dvc_cache` Subvolume | Executed `chattr +C /path` (`nodatacow`) | DuckDB & DVC Storage | Eliminates BTRFS disk fragmentation during continuous writes. |

###### *Appendix B: Workstation Memory Budget Allocation*
Host system memory (128 GB DDR5 RAM) is allocated across subsystems to prevent out-of-memory thrashing and maintain high data throughput:

| System Subsystem | Memory Allocation (GB) | Percentage of Total | Operational Purpose |
| :--- | :--- | :--- | :--- |
| **OS & Core Infrastructure** | 8.0 GB | 6.25% | Host Linux kernel, Dropbear SSH, Tailscale, base services. |
| **BTRFS Page Cache & THP** | 32.0 GB | 25.00% | Disk page caching, 2MB Transparent HugePage allocations (`madvise`). |
| **IVF-HNSW Vector Index** | 40.0 GB | 31.25% | In-memory vector graph caching for BioCLIP-2 feature embeddings. |
| **Arrow In-Memory Buffers** | 28.0 GB | 21.88% | Zero-copy Arrow C-ABI data streaming buffers and image ingestion. |
| **PyTorch / ONNX Workspace** | 20.0 GB | 15.62% | ONNX execution graph buffers, quantization tensors, VRAM staging. |
| **Total System RAM** | **128.0 GB** | **100.0%** | Full host system memory budget. |
