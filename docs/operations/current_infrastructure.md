---
type: Reference
title: Current Infrastructure & MLOps State
status: stable
stale_after: "2027-01-01T00:00:00Z"
version: 1.0
description: Details of the current server setup, CI/CD pipeline, and ML ecosystem.
tags: [infrastructure, mlops, state]
---

# Current Infrastructure & MLOps State

## Physical Server & Cluster Setup

* **Hardware:** Single-node bare-metal server (`hive-mind`) running Ubuntu. Equipped with an Intel i7-14700K CPU, 128GB RAM, and an RTX 5070 Ti GPU.
* **Security & Boot:** Full Disk Encryption (LUKS) is managed remotely via a custom `unlock_taxon.sh` Dropbear/SSH handover script.
* **Kubernetes (K3s) Architecture:** We run a lightweight `k3s` distribution directly on the bare-metal host. The cluster is integrated with the NVIDIA Container Toolkit to expose the host's GPU to Kubernetes pods. This setup acts as the central execution engine for our CI/CD workflows and heavy MLOps tasks.
* **Actions Runner Controller (ARC):** Instead of installing the GitHub Actions runner directly on the host OS, we use ARC to natively orchestrate runner pods within the `arc-systems` namespace.
    * **Runner Registration:** ARC authenticates with GitHub using a Personal Access Token to request short-lived registration tokens via the GitHub API. The controller dynamically spawns `Runner` pods based on a `RunnerDeployment` definition.
    * **Pod Configuration:** The runner pods are instantiated from the `summerwind/actions-runner` image. Our `RunnerDeployment` requests `nvidia.com/gpu: "1"` under resource limits to ensure the physical GPU is exclusively mounted into the runner container. The pods announce themselves to GitHub using the labels `k3s-pod`, `gpu`, and `taxon-vision`.

## Core MLOps Ecosystem

* **Package & Environment Management:** Strictly managed via `pixi`. We utilise a dual-environment strategy: `ci-dev` (CPU only) for continuous integration, and `dev` (GPU enabled) for local execution.
* **Task Automation:** All pipeline steps, testing, and linting are executed via `just` (as defined in our `Justfile`).
* **CI/CD Pipelines:** A robust `.github/workflows/ci.yml` triggers exclusively on pushes or pull requests to the `main` and `develop` branches. Deployments are secured via GitHub Environments (`taxon-vision-prod` and `taxon-vision-staging`), requiring manual administrator approval to prevent unauthorised execution on our bare-metal infrastructure.
* **Data Versioning & Tracking:** We rely on DVC for dataset immutability and MLflow (hosted on DagsHub) for experiment tracking.
    * **Credential Injection:** To allow the ARC runner pods to push/pull from DagsHub without hardcoding secrets in the repository, we created a Kubernetes Secret (`dagshub-credentials`) on the host cluster containing the credentials mapped directly from the host's `.zshrc`.
    * **Integration Mechanism:** The `RunnerDeployment` is configured with an `envFrom` block referencing this secret. When ARC spins up a new runner pod, Kubernetes automatically maps the keys (`MLFLOW_TRACKING_USERNAME`, `MLFLOW_TRACKING_PASSWORD`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`) directly into the pod's environment variables. This ensures that any `dvc pull` or MLflow logging executed inside the GitHub Action runner automatically picks up the credentials natively from the environment without needing explicit GitHub Secrets.

## Outstanding Implementation Tasks

* **Data Pipeline Execution:** Actual execution of DVC pipelines to ingest and transform iNaturalist/GBIF DarwinCore observation data.
* **Model Training & Benchmarking:** Implementation of the multi-backbone Pareto benchmarking (`BioCLIP-2`, `DINOv3`, `MobileNetV4`) as defined in the strategy report.
* **Inference Optimisation:** ONNX Runtime export with dynamic INT8/FP16 quantisation for low-latency inference.
* **Uncertainty Calibration:** Implementation of Split Conformal Prediction for coverage bounds and OOD detection logic.
* **Web Service & Dashboard:** Development of the unified FastAPI machine REST API and the server-rendered Jinja2/HTMX operator dashboard.
* **Observability Stack:** Deployment of the Prometheus telemetry stack and Grafana dashboards for monitoring concept drift and latency budgets.
