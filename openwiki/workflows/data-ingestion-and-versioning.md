---
type: workflow
title: Data Ingestion & Dataset Versioning
description: Asynchronous observation ingestion, S3 streaming, Creative Commons licensing validation, Parquet manifest generation, and DVC dataset tracking with DagsHub synchronization.
tags: [data-ingestion, dataset-versioning, inaturalist, s3-streaming, parquet, dvc, dagshub, licensing, attribution]
verified:
  - by: openwiki/0.7.2
    at: 2026-10-15T19:46:36.219Z
sources:
  - id: openwiki-source-c28ca7ab362d7136f953793e
    resource: repo://.dvc/config
  - id: openwiki-source-4935f5c0efb229189fbcdd99
    resource: repo://dvc.lock
  - id: openwiki-source-cfa03d5d7ac7270f0bb1bf28
    resource: repo://dvc.yaml
  - id: openwiki-source-14e6a4e1ddc06973b783ff20
    resource: repo://Justfile
  - id: openwiki-source-0793ecc43815420962b11dbf
    resource: repo://scripts/ingest_data.py
  - id: openwiki-source-5636f69cf274aaadd4c1c966
    resource: repo://src/taxon_vision/data/inat_client.py
  - id: openwiki-source-ff920988bce76958fa7d7039
    resource: repo://src/taxon_vision/data/license_filter.py
  - id: openwiki-source-0ca3c28435fdc7020685575f
    resource: repo://src/taxon_vision/data/s3_streamer.py
  - id: openwiki-source-cc156e96dc223f9f454ee279
    resource: repo://src/taxon_vision/domain/license.py
  - id: openwiki-source-a88159face5e03f82447df3c
    resource: repo://src/taxon_vision/models/training/tracking.py
  - id: openwiki-source-db9076027e694f431684d0c0
    resource: repo://src/taxon_vision/pipelines/extract_features.py
generated: { by: "openwiki/0.7.2", at: "2026-10-15T19:46:36.219Z" }
---

# Data Ingestion & Dataset Versioning

The data ingestion and dataset versioning subsystem provides the reproducible, auditable foundation for the TaxonVision machine learning pipeline. Biodiversity image datasets present unique operational challenges: global citizen-science repositories contain petabytes of media with skewed class distributions, diverse copyright licenses, and variable community identification quality. TaxonVision addresses these challenges through a zero-disk streaming architecture that pairs automated licensing verification with cryptographic dataset lineage tracking.

Rather than bulk-downloading gigabytes of image binaries to local workstations or build runners, TaxonVision separates lightweight observation metadata from image payloads. Metadata is ingested via an asynchronous, rate-limited iNaturalist API client, filtered against strict intellectual property criteria, and serialized into high-performance Apache Arrow Parquet manifests tracked by Data Version Control (DVC). Deep learning representation extractors and training routines consume these manifests directly, streaming medium-resolution imagery on demand from the AWS Open Data S3 registry via HTTP connection pooling.

For surrounding architectural context, see [Domain Entities & Taxonomy Perimeter](../architecture/domain-and-taxa.md), [System Overview](../architecture/system-overview.md), [Model Training & Tracking](model-training-and-tracking.md), and [Deployment & Observability](../operations/deployment-and-observability.md).

---

## Architectural Data Flow

The following diagram illustrates the progression of biological observation data from public repositories into immutable tabular manifests, S3 image streaming generators, and synchronized cloud storage:

<!-- openwiki: mermaid parse failed and this diagram was converted to a text fence so it does not break rendering. Fix the diagram source and restore the mermaid fence. Parser error: Heuristic: an unescaped angle bracket inside a label breaks rendering; rephrase the label. -->
```text
flowchart TD
    subgraph INat["iNaturalist API v1"]
        API["REST Endpoint /observations"]
    end

    subgraph Ingestion["Ingestion Pipeline (scripts/ingest_data.py)"]
        Client["INatAPIClient<br/>(60 req/min throttle, 1.0s delay)"]
        Filter["LicenseFilter<br/>(OpenLicense validation & attribution)"]
    end

    subgraph Storage["Manifest & Versioning Storage"]
        Parquet["data/manifests/dataset_manifest.parquet<br/>(Polars Arrow Table)"]
        DVCLock["dvc.lock<br/>(Stage checksums & manifest directory hash)"]
        DagsHub["DagsHub S3 Remote (s3://dvc)<br/>(Authenticated via Secret Service)"]
    end

    subgraph Downstream["Training & Extraction Consumers"]
        Streamer["S3ImageStreamer<br/>(Lazy generator & thread pool)"]
        AWSS3["AWS Open Data S3 Bucket<br/>(inaturalist-open-data)"]
        Extractor["Feature Extraction & Training<br/>(Stratified split & MLflow lineage)"]
    end

    API -->|"GET research-grade photos"| Client
    Client -->|"Raw observation dicts"| Filter
    Filter -->|"AttributionRecord validation"| Parquet
    Parquet -->|"dvc repro ingestion"| DVCLock
    DVCLock -->|"just dvc-push / secret-tool"| DagsHub
    Parquet -->|"Columns: URL, Taxon ID, UUID"| Streamer
    Streamer -->|"HTTP GET presigned image URL"| AWSS3
    AWSS3 -->|"Decoded RGB PIL.Image"| Extractor
    DVCLock -.->|"Manifest MD5 hash lineage"| Extractor
```

Data progression flowchart illustrating observation metadata ingestion, Parquet manifest persistence, DVC remote synchronization to DagsHub, and S3 image streaming into training workflows.

---

## Asynchronous iNaturalist API Client

Metadata retrieval is implemented by the `INatAPIClient` class in `src/taxon_vision/data/inat_client.py`. The client interfaces with the iNaturalist REST API v1 (`https://api.inaturalist.org/v1`) while strictly adhering to community terms of service.

### Rate Limiting and Spacing Invariant

The iNaturalist API v1 enforces an aggressive rate limit of 60 requests per minute (1 request per second) per IP address, with burst traffic penalized by temporary bans. The `INatAPIClient` enforces compliance at the transport layer by computing a constant minimum spacing delay:

$$\Delta t = \frac{60.0}{R_{\max}}$$

where $R_{\max}$ defaults to 60 requests per minute (`delay_seconds = 1.0`). Prior to executing any HTTP request, the client executes `await asyncio.sleep(self.delay_seconds)`. This cooperative scheduling prevents connection bursts when iterating through multiple catalog taxa.

### Quality Filtering & Resolution Standardization

Biological foundation models require high label purity. To guarantee taxon identification accuracy, `fetch_observations` queries the `/observations` endpoint with targeted filter parameters:

* `taxon_id`: Authoritative Darwin Core taxonomic identifier (e.g., `48662` for *Danaus plexippus*).
* `quality_grade`: Hardcoded to `"research"`. This parameter guarantees that community taxonomists have reached consensus on the organism's species identification.
* `has[]`: Hardcoded to `"photos"`, eliminating text-only occurrences.
* `per_page`: Number of observation records retrieved per request page (configured via `settings.ingestion.per_page`).

Default observation thumbnails returned by iNaturalist use a square crop (`square`, $75 \times 75$ pixels) unsuitable for neural feature extraction. The client intercepts image locators and rewrites the URL path segment:

```python
photo.get("url", "").replace("square", "medium")
```

This upgrades locators to medium resolution ($500\text{ px}$ along the longest dimension), providing sufficient spatial acuity for pre-trained vision backbones (such as DINOv3 or MobileNetV4) while remaining within AWS Open Data CDN delivery quotas.

Network calls are wrapped using an asynchronous `httpx.AsyncClient` with a 10.0-second timeout, raising standard `httpx.HTTPStatusError` on non-2xx responses or `httpx.RequestError` on connectivity failures.

---

## Licensing Auditing & Photographer Attribution

Citizen-science imagery is contributed under varied intellectual property designations, ranging from public domain dedications to restrictive "All Rights Reserved" copyrights. Ingesting proprietary imagery into machine learning pipelines creates legal risk and violates open science licensing terms. The `LicenseFilter` class (`src/taxon_vision/data/license_filter.py`) enforces strict copyright compliance and attribution retention before records reach disk.

### Permitted Licenses

TaxonVision accepts only Creative Commons licenses that permit model training and derived research works, as defined in `src/taxon_vision/domain/license.py`:

* `OpenLicense.CC0`: Public domain dedication.
* `OpenLicense.CC_BY`: Creative Commons Attribution.
* `OpenLicense.CC_BY_NC`: Creative Commons Attribution-NonCommercial.

The static method `LicenseFilter.is_license_permitted(license_str)` normalizes raw license strings by trimming whitespace, converting to uppercase, and replacing underscores with hyphens before attempting instantiation against `OpenLicense`. Inadmissible variants (such as `CC-BY-ND` NoDerivatives, commercial-only clauses, or unstated copyrights) return `False` and are discarded.

### Static Domain Anti-Scraping Protection

To prevent scraping of private or unverified assets directly from iNaturalist web endpoints, `LicenseFilter.is_static_domain_prohibited` inspects the asset locator:

```python
if "static.inaturalist.org" in image_url and not is_open_dataset_verified:
    return True
```

If an image locator points to `static.inaturalist.org` without prior verification against an official open data registry manifest (`is_open_dataset_verified = True`), the record is rejected. Legitimate mirrors such as `inaturalist-open-data.s3.amazonaws.com` pass verification automatically.

### Attribution Record Preservation

Under Creative Commons terms, downstream users must retain creator credits, license variants, and source pointers. For each compliant record, `LicenseFilter.process_record` creates an immutable Pydantic `AttributionRecord`:

```python
AttributionRecord(
    photographer_name=photographer,
    license_code=OpenLicense(license_str),
    license_url=str(raw.get("license_url", "https://creativecommons.org/licenses/")),
    observation_uuid=str(raw.get("observation_id", "")),
    source_portal="iNaturalist",
)
```

Blank photographer fields fall back gracefully to `"Unknown Contributor"`.

---

## Parquet Manifest Generation

The orchestration script `scripts/ingest_data.py` synthesizes API retrieval and licensing auditing into an immutable dataset manifest.

### Ingestion Execution Lifecycle

1. **Taxon Discovery**: Reads `settings.ingestion.taxa_to_fetch` from `config/params.yaml`. If unspecified, it dynamically extracts all species with an `inat_taxon_id` from the authoritative catalog via `load_taxa_catalog()`.
2. **Asynchronous Traversal**: Instantiates `INatAPIClient` and iterates through the target taxa, fetching `settings.ingestion.per_page` (default 200) observations per taxon.
3. **Audit Filtering**: Passes raw records through `LicenseFilter.process_record(obs)`. Non-compliant observations are discarded.
4. **Columnar Serialization**: Compiles valid records into a Polars DataFrame (`pl.DataFrame(all_records)`) and writes them directly to `data/manifests/dataset_manifest.parquet`.

```bash
# Execute full ingestion workflow via Pixi or Just
just ingest-data
# or directly:
pixi run --frozen -e dev python scripts/ingest_data.py
```

### Manifest Schema

The resulting Parquet file exhibits the following tabular structure:

| Field Name | Polars / Arrow Type | Description |
|---|---|---|
| `observation_uuid` | `String` / `Utf8` | Unique Darwin Core observation identifier. |
| `taxon_id` | `Int64` | External taxonomic identifier assigned by iNaturalist. |
| `image_url` | `String` / `Utf8` | Standardized medium-resolution image locator URL. |
| `photographer_name` | `String` / `Utf8` | Retained creator credit for licensing attribution. |
| `license_code` | `String` / `Utf8` | Normalized Creative Commons code (`CC0`, `CC-BY`, `CC-BY-NC`). |
| `license_url` | `String` / `Utf8` | Direct URL to Creative Commons legal deed. |

Using Apache Arrow Parquet provides significant performance advantages over CSV or JSON lines: columnar projection allows training loops to extract `(observation_uuid, taxon_id, image_url)` without parsing text fields, compression bounds manifest storage to kilobytes, and schema types remain strictly enforced.

---

## AWS Open Data S3 Streaming Reader

Storing millions of biological training images locally exhausts workstation SSDs and inflates cloud compute provisioning time. TaxonVision solves this via `S3ImageStreamer` (`src/taxon_vision/data/s3_streamer.py`), a lazy streaming reader that fetches images directly from the AWS Open Data iNaturalist S3 registry in memory during training and feature extraction.

### Architecture & Memory Bounds

`S3ImageStreamer` initializes by reading the Parquet manifest into memory using Polars:

```python
streamer = S3ImageStreamer(manifest_path="data/manifests/dataset_manifest.parquet")
```

The generator method `stream_images(limit, shuffle, balanced, seed, max_workers)` yields tuples of:

$$(u_i, t_i, I_i) \in (\text{UUID}, \text{TaxonID}, \text{PIL.Image})$$

Because `stream_images` is a lazy Python generator, only active image buffers remain allocated in RAM at any given moment. Disk usage is zero, eliminating temporary directory thrashing.

### Sequential vs. Concurrent Streaming Modes

To maximize network throughput while accommodating constrained environments, the reader supports dual streaming modes:

* **Sequential Streaming (`_stream_sequential`)**: Activated when `max_workers <= 1` or dataset slice $\le 2$. Reuses a single persistent `httpx.Client(timeout=15.0)` connection across requests to reduce TCP handshake overhead.
* **Concurrent Streaming (`_stream_concurrent`)**: Activated when `max_workers > 1`. Uses Python's `concurrent.futures.ThreadPoolExecutor` with worker concurrency up to `max_workers` (default 8). Records are processed in bounded chunks ($\text{chunk\_size} = \max(1, \min(32, N))$) via `executor.map`, yielding decoded images as worker threads complete downloads.

### Resilient Decoding Pipeline

Each record is downloaded and decoded inside `_fetch_record_image`:

```python
response = client.get(image_url)
response.raise_for_status()
raw_img = Image.open(io.BytesIO(response.content))
img = raw_img.convert("RGB") if raw_img.mode != "RGB" else raw_img
```

If an individual remote URL fails due to transient HTTP 404/503 errors or corrupt image bytes, the streamer logs a warning and returns `None`, allowing the generator to drop the corrupted image without crashing the downstream training loop.

### Streamer CLI Diagnostic Utility

Developers can verify network connectivity, evaluate throughput, or cache a small local inspection batch using the CLI entrypoint:

```bash
# Stream 100 sample images into local directory for inspection
just fetch-sample

# Direct CLI invocation with custom manifest and output target
pixi run --frozen -e dev python -m taxon_vision.data.s3_streamer \
    --manifest data/manifests/dataset_manifest.parquet \
    --limit 20 \
    --out data/sample/
```

---

## Balanced Taxon Sampling & Stratified Partitioning

Real-world citizen science observations follow an extreme long-tail distribution: common urban species receive tens of thousands of observations, whereas rare or endangered taxa have sparse records. Unchecked training on raw observations skews neural classifiers toward high-frequency majority classes.

### Balanced Manifest Sampling

`S3ImageStreamer` includes a manifest balancing algorithm implemented in `_prepare_balanced_manifest(df, limit, shuffle, seed)`:

1. **Taxon Partitioning**: Queries unique taxa present in the manifest:
   $$K = |\{t \mid t \in \text{df}[\text{"taxon\_id"}]\}|$$
2. **Per-Taxon Allocation**: Computes an equal ceiling quota:
   $$N_{\text{per\_taxon}} = \max\left(1, \left\lfloor\frac{\text{limit} + K - 1}{K}\right\rfloor\right)$$
3. **Partition Sampling**: Filters the dataframe for each taxon ID $k$. If `shuffle=True`, it samples the partition using `df.sample(fraction=1.0, shuffle=True, seed=seed)` before taking the top $N_{\text{per\_taxon}}$ records.
4. **Reassembly & Truncation**: Concatenates all taxon slices with `pl.concat`, optionally reshuffles the entire collection, and slices to the exact requested `limit`.

This guarantees that during zero-shot evaluation, active learning queries, or feature caching, each taxon category receives equal representation.

### Dense Taxon Mapping

External iNaturalist identifiers are arbitrary, non-contiguous integers (e.g., Western Honey Bee is `47219`, Gray Wolf is `47144`). Machine learning loss functions (`CrossEntropyLoss`, `ClassBalancedLoss`) require dense integer class indices $y \in [0, K-1]$.

In `src/taxon_vision/pipelines/extract_features.py`, `build_taxon_index_map` iterates through catalog entries and alias records from `config/taxa_catalog.yaml`, mapping external identifiers to continuous dense indices:

$$\phi: \text{TaxonID} \rightarrow \{0, 1, \dots, K-1\}$$

Any observation in the manifest whose `taxon_id` cannot be mapped through $\phi$ is rejected prior to feature extraction.

### Stratified Dataset Partitioning

During offline representation caching (`extract_features.py`), extracted feature vectors $Z \in \mathbb{R}^{N \times D}$ and integer labels $y \in \mathbb{Z}^N$ are split into training and validation sets via `create_stratified_embedding_split`:

```python
train_idx, val_idx = train_test_split(
    indices,
    test_size=val_split,
    stratify=labels.numpy(),
    random_state=random_seed,
)
```

Stratification enforces identical class ratios in both splits. This invariant prevents rare classes from having zero validation instances, which would otherwise result in undefined PR-AUC or F1-scores.

The function calculates the empirical sample frequency per class:

$$n_c = \sum_{i \in \mathcal{D}_{\text{train}}} \mathbb{I}(y_i = c)$$

and packages the tensors into an `EmbeddingSplit` dataclass (`train_embeddings`, `train_labels`, `val_embeddings`, `val_labels`, `samples_per_class`). The `samples_per_class` vector is directly consumed by `ClassBalancedLoss` to calculate effective sample weights:

$$W_c = \frac{1 - \beta}{1 - \beta^{n_c}}$$

---

## DVC Dataset Versioning & Remote Synchronization

TaxonVision tracks datasets and model artifacts using Data Version Control (DVC). Manifests, intermediate embeddings, and model weights are tracked outside Git while maintaining strict version synchronization.

### Pipeline Dependency Graph in `dvc.yaml`

The DVC pipeline configuration (`dvc.yaml`) formalizes the data ingestion stage, its code dependencies, parameters, and outputs:

```yaml
stages:
  ingestion:
    cmd: pixi run --frozen -e dev python scripts/ingest_data.py
    deps:
      - scripts/ingest_data.py
      - src/taxon_vision/data/inat_client.py
      - src/taxon_vision/data/license_filter.py
    params:
      - config/params.yaml:
        - ingestion.taxa_to_fetch
        - ingestion.per_page
    outs:
      - data/manifests
  train:
    cmd: pixi run --frozen -e dev python -m taxon_vision.models.trainer --epochs 5 --extractor dinov3
    deps:
      - data/manifests
      - src/taxon_vision/models/
    params:
      - config/params.yaml:
        - training.epochs
        - training.extractor
    outs:
      - models/checkpoints/head.pt
  export_onnx:
    cmd: pixi run --frozen -e dev python -m taxon_vision.inference.onnx_exporter
    deps:
      - models/checkpoints/head.pt
      - src/taxon_vision/inference/onnx_exporter.py
    outs:
      - models/checkpoints/model.onnx
```

When running `dvc repro`, DVC inspects the cryptographic MD5 hashes of dependencies and the parameters in `config/params.yaml`. If neither code nor parameters have changed, DVC skips re-ingestion and reuses the cached manifest directory.

### Cryptographic Lineage Tracking in `dvc.lock`

Execution results and exact content hashes are committed to Git in `dvc.lock`:

```yaml
schema: '2.0'
stages:
  ingestion:
    cmd: pixi run --frozen -e dev python scripts/ingest_data.py
    deps:
    - path: scripts/ingest_data.py
      hash: md5
      md5: 252699c9b9439376ffff4c510e27a1fe
    - path: src/taxon_vision/data/inat_client.py
      hash: md5
      md5: 6f925d7c31fd3d16da8bc5640ca4c470
    - path: src/taxon_vision/data/license_filter.py
      hash: md5
      md5: 9d53e65484b212283079875587a0dca3
    params:
      config/params.yaml:
        ingestion.per_page: 200
        ingestion.taxa_to_fetch:
        - 47120
        - 47119
        - 47114
    outs:
    - path: data/manifests
      hash: md5
      md5: e1ead79bbe4bac3b0b80cf0d11a68d6b.dir
```

To establish an unbroken audit trail between data versions and model checkpoints, `taxon_vision.models.training.tracking._get_dvc_dataset_hash()` reads `dvc.lock` at training time, extracts the MD5 checksum of `data/manifests`, and logs it to MLflow as the `dvc_dataset_hash` run parameter. This ensures any deployed ONNX model can be traced back to the exact manifest snapshot used to train it.

### DagsHub Remote Synchronization & Secret Service Integration

DVC remote storage is configured in `.dvc/config`:

```ini
[core]
    remote = origin
['remote "origin"']
    url = s3://dvc
    endpointurl = https://dagshub.com/foersben/taxon-vision-mlops.s3
```

The remote points to DagsHub's S3-compatible storage tier (`s3://dvc`). To avoid storing AWS access keys or DagsHub tokens in plain text configuration files, TaxonVision delegates authentication to the host OS keyring via `secret-tool` (interfacing with FreeDesktop Secret Service or KeePassXC).

The recipes in `Justfile` manage this credential extraction strictly in-memory:

```just
# Push tracked datasets & models to DagsHub using in-memory Secret Service credentials
[group("mlops")]
dvc-push:
    @TOKEN=$(secret-tool lookup Title DagsHub 2>/dev/null || secret-tool lookup service dagshub 2>/dev/null); \
    if [ -z "$TOKEN" ]; then \
        echo "❌ Error: KeePassXC is locked or DagsHub token not found in Secret Service." >&2; \
        exit 1; \
    fi; \
    AWS_ACCESS_KEY_ID="$TOKEN" AWS_SECRET_ACCESS_KEY="$TOKEN" \
    pixi run --frozen -e dev dvc push

# Pull tracked datasets & models from DagsHub
[group("mlops")]
dvc-pull:
    @TOKEN=$(secret-tool lookup Title DagsHub 2>/dev/null || secret-tool lookup service dagshub 2>/dev/null); \
    if [ -z "$TOKEN" ]; then \
        echo "❌ Error: KeePassXC is locked or DagsHub token not found in Secret Service." >&2; \
        exit 1; \
    fi; \
    AWS_ACCESS_KEY_ID="$TOKEN" AWS_SECRET_ACCESS_KEY="$TOKEN" \
    pixi run --frozen -e dev dvc pull
```

Credentials are passed to the subshell environment only for the duration of the `dvc push` or `dvc pull` command, never touching the filesystem or shell history.

---

## Operations & Command Reference

The following table summarizes the operational commands and recipes for data ingestion, sampling, and versioning:

| Task / Purpose | Command / Just Recipe | Underlying Execution |
|---|---|---|
| Ingest observation metadata | `just ingest-data` | `pixi run --frozen -e dev python scripts/ingest_data.py` |
| Stream sample images to disk | `just fetch-sample` | `pixi run --frozen -e dev python -m taxon_vision.data.s3_streamer --limit 100 --out data/sample/` |
| Download catalog calibration samples | `just download-samples per_taxon="1"` | `pixi run --frozen -e dev python scripts/download_samples.py --per-taxon 1` |
| Reproduce DVC ingestion stage | `pixi run --frozen -e dev dvc repro ingestion` | Executes `dvc.yaml` ingestion stage if dependencies changed |
| Push manifests & models to DagsHub | `just dvc-push` | Authenticates via `secret-tool` and runs `dvc push` |
| Pull manifests & models from DagsHub | `just dvc-pull` | Authenticates via `secret-tool` and runs `dvc pull` |
| Run data component unit tests | `pixi run --frozen -e dev pytest tests/unit/test_data_components.py` | Tests `INatAPIClient`, `S3ImageStreamer`, and domain entities |
| Run licensing compliance tests | `pixi run --frozen -e dev pytest tests/unit/test_license_filter.py` | Validates permitted licenses and static domain anti-scraping rules |
