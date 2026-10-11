---
type: Reference
title: Current Infrastructure & MLOps State
status: stable
stale_after: "2027-01-01T00:00:00Z"
version: 1.1
description: Details of the server setup, k3s cluster, ARC runners, and step-by-step configuration runbook.
tags: [infrastructure, mlops, state, k3s, arc]
---

# Current Infrastructure & MLOps State

This document outlines the hardware, cluster configuration, and MLOps tooling powering
TaxonVision, including the step-by-step instructions for reproducing and maintaining the setup.

## Physical Server & Host Configuration

### Host Hardware Profile

* **Node Identity:** Single-node bare-metal server (`hive-mind`) running Ubuntu 26.04.1 LTS (codename `resolute`).
* **Processor (CPU):** Intel Core i7-14700K (20 physical cores, 28 threads).
* **System Memory (RAM):** 128 GB DDR5.
* **Graphics Hardware (GPU):** NVIDIA GeForce RTX 5070 Ti (16 GB VRAM, Blackwell, compute capability 12.0) with NVIDIA driver 615.71.09 (CUDA 13.4 user-mode driver).
* **Storage:** NVMe SSD storage with Full Disk Encryption (LUKS).

### Remote Boot & Remote LUKS Decryption Setup

Because the host uses Full Disk Encryption (LUKS), remote reboots require network-accessible
decryption before the root filesystem mounts.

* **Dropbear SSH in Initramfs:** Configured inside the initial ramdisk on an isolated port.
* **Key Authorization:** Authorized SSH keys are placed into `/etc/dropbear/initramfs/authorized_keys`.
* **Unlock Automation:** The root disk is unlocked over SSH via `cryptroot-unlock`, wrapped locally by `scripts/unlock_taxon.sh`:

```bash
# Unlock remote host via Dropbear SSH wrapper
./scripts/unlock_taxon.sh
```

### NVIDIA Drivers & Container Toolkit Setup

To make the RTX 5070 Ti accessible to container engines, the proprietary NVIDIA driver
and the NVIDIA Container Toolkit are installed directly on the Ubuntu host. The verified
baseline is driver `615.71.09` reporting CUDA `13.4` (`nvidia-smi`). Because the CUDA user-mode
driver is backward compatible, containers may ship an older CUDA runtime; for the Blackwell
GPU this runtime must be CUDA 12.8 or newer, since earlier PyTorch builds contain no kernels
for compute capability 12.0:

```bash
# 1. Install the recommended proprietary NVIDIA graphics driver
sudo apt-get update
sudo ubuntu-drivers install

# 2. Add NVIDIA Container Toolkit repository
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# 3. Install NVIDIA Container Toolkit
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# 4. Configure host container runtime and restart containerd
sudo nvidia-ctk runtime configure --runtime=containerd
sudo systemctl restart containerd
```

## Kubernetes (K3s) Cluster Setup

The execution cluster runs a lightweight `k3s` distribution directly on bare metal,
integrated with the NVIDIA Container Toolkit to pass the host GPU to Kubernetes workloads.

### Installing K3s with NVIDIA Runtime Integration

1. **Install K3s:** Deploy K3s with Traefik disabled (ingress is managed via custom services)
   and enable default containerd socket access:

```bash
curl -sfL https://get.k3s.io | INSTALL_K3S_EXEC="--disable traefik" sh -
```

1. **Configure K3s Containerd Template for NVIDIA Runtime:** Create or modify
   `/var/lib/rancher/k3s/agent/etc/containerd/config.toml.tmpl` to register the `nvidia` runtime:

```toml
[plugins."io.containerd.grpc.v1.cri".containerd.runtimes."nvidia"]
  runtime_type = "io.containerd.runc.v2"
[plugins."io.containerd.grpc.v1.cri".containerd.runtimes."nvidia".options]
  BinaryName = "/usr/bin/nvidia-container-runtime"
```

1. **Restart K3s Service:**

```bash
sudo systemctl restart k3s
```

1. **Deploy NVIDIA Kubernetes Device Plugin & Time-Slicing Configuration:** Deploy the device plugin daemonset and apply the time-slicing ConfigMap so K3s exposes 4 virtual GPU slices for the physical RTX 5070 Ti:

```bash
kubectl apply -f deploy/k8s/nvidia-time-slicing-config.yaml
```

Patch the device plugin daemonset to read the configuration:

```bash
kubectl patch daemonset nvidia-device-plugin-daemonset -n kube-system --patch '
spec:
  template:
    spec:
      containers:
      - name: nvidia-device-plugin-ctr
        env:
        - name: CONFIG_FILE
          value: /etc/config/config.yaml
        volumeMounts:
        - name: config
          mountPath: /etc/config
      volumes:
      - name: config
        configMap:
          name: nvidia-device-plugin-config
'
```

1. **Verify GPU Capacity:**

```bash
kubectl get nodes "-o=custom-columns=NAME:.metadata.name,GPU:.status.allocatable.nvidia\.com/gpu"
```

The output confirms 4 allocatable `nvidia.com/gpu` virtual units on the `hive-mind` node, permitting concurrent multi-tenant execution across Web API inference, PyTorch batch training, and CI test runners without exclusive locking.

## Actions Runner Controller (ARC) Setup

Instead of running long-lived GitHub Actions runner binaries directly on the host OS,
runners are deployed as ephemeral Kubernetes pods in the `arc-systems` namespace using
Actions Runner Controller (ARC).

### Step 1: Deploy cert-manager Prerequisite

ARC requires `cert-manager` for webhook admission certificates:

```bash
kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.16.0/cert-manager.yaml
kubectl rollout status deployment -n cert-manager cert-manager --timeout=120s
```

### Step 2: Install ARC via Helm

1. Add the ARC Helm repository:

```bash
helm repo add actions-runner-controller https://actions-runner-controller.github.io/actions-runner-controller
helm repo update
```

1. Create the `arc-systems` namespace and register the GitHub Personal Access Token (PAT):

```bash
kubectl create namespace arc-systems

# Store GitHub PAT with repo/admin:org scope
kubectl create secret generic controller-manager \
    -n arc-systems \
    --from-literal=github_token="<YOUR_GITHUB_PAT>"
```

1. Install the controller manager:

```bash
helm install arc actions-runner-controller/actions-runner-controller \
    --namespace arc-systems \
    --set syncPeriod=1m
```

### Step 3: Deploy the GPU RunnerDeployment Manifest

The runner pods are instantiated from the `summerwind/actions-runner` image, requesting
exclusive GPU access and mounting DagsHub credentials from a Kubernetes Secret.

Create and apply `arc-runner-deployment.yaml`:

```yaml
apiVersion: actions.summerwind.dev/v1alpha1
kind: RunnerDeployment
metadata:
  name: taxon-vision-gpu-runner
  namespace: arc-systems
spec:
  replicas: 1
  template:
    metadata:
      labels:
        app: taxon-vision-runner
    spec:
      repository: foersben/taxon-vision-mlops
      labels:
        - k3s-pod
        - gpu
        - taxon-vision
      envFrom:
        - secretRef:
            name: dagshub-credentials
      resources:
        limits:
          nvidia.com/gpu: "1"
        requests:
          nvidia.com/gpu: "1"
          cpu: "4"
          memory: "16Gi"
```

Apply the manifest:

```bash
kubectl apply -f arc-runner-deployment.yaml
kubectl get pods -n arc-systems -l app=taxon-vision-runner
```

## Secret Management & DagsHub Integration

We use DVC for dataset versioning and MLflow on DagsHub for experiment tracking.
To avoid hardcoding secrets in repository workflows, credentials are injected at the
pod level via Kubernetes Secrets.

### Creating the DagsHub Kubernetes Secret

The secret `dagshub-credentials` is provisioned on the host cluster in the `arc-systems`
namespace. The values are mapped directly from the local developer environment:

```bash
kubectl create secret generic dagshub-credentials \
    -n arc-systems \
    --from-literal=MLFLOW_TRACKING_USERNAME="$DAGSHUB_USERNAME" \
    --from-literal=MLFLOW_TRACKING_PASSWORD="$DAGSHUB_TOKEN" \
    --from-literal=AWS_ACCESS_KEY_ID="$DAGSHUB_AWS_ACCESS_KEY_ID" \
    --from-literal=AWS_SECRET_ACCESS_KEY="$DAGSHUB_AWS_SECRET_ACCESS_KEY"
```

### Environment Variable Injection in Runners

When ARC instantiates a runner pod, Kubernetes evaluates `envFrom: [secretRef: {name: dagshub-credentials}]`.
The keys are mapped into the container environment variables:

* `MLFLOW_TRACKING_USERNAME`
* `MLFLOW_TRACKING_PASSWORD`
* `AWS_ACCESS_KEY_ID`
* `AWS_SECRET_ACCESS_KEY`

This allows any script executing `dvc pull` or `mlflow.start_run()` to authenticate
transparently without needing GitHub Actions Repository Secrets.

### DVC Remote Configuration

The repository DVC configuration routes to the DagsHub S3-compatible bucket (`s3://dvc` at `https://dagshub.com/foersben/taxon-vision-mlops.s3`):

```bash
# S3-compatible remote on DagsHub
pixi run -e dev dvc remote add -d dagshub s3://dvc
pixi run -e dev dvc remote modify dagshub endpointurl https://dagshub.com/foersben/taxon-vision-mlops.s3
```

Local developers never commit credentials to `.dvc/config.local`. Instead, `just dvc-push` queries the token in-memory from KeePassXC via Secret Service:

```bash
# Push manifests and artifacts using ephemeral Secret Service tokens
just dvc-push
```

## Core MLOps Ecosystem

* **Package & Environment Management:** Strictly managed via `pixi`. We utilise a dual-environment strategy: `ci-dev` (CPU only) for continuous integration, and `dev` (GPU enabled) for the remote GPU runners; model training is never executed on the developer workstation.
* **Task Automation:** All pipeline steps, testing, and linting are executed via `just` (as defined in our `Justfile`).
* **CI/CD Pipelines:** A robust `.github/workflows/ci.yml` triggers exclusively on pushes or pull requests to the `main` and `develop` branches. Deployments are secured via GitHub Environments (`taxon-vision-prod` and `taxon-vision-staging`), requiring manual administrator approval to prevent unauthorised execution on our bare-metal infrastructure.
* **Data Versioning & Tracking:** We rely on DVC for dataset immutability and MLflow (hosted on DagsHub) for experiment tracking.

## Production CI/CD Architecture: GitHub Actions + Jenkins Hybrid

The system implements a production two-layer CI/CD architecture:

* **Layer 1 - GitHub Actions (lightweight gates):** Lint, type checks, unit tests (CPU), license audit, OKF validation, conformal audit, Docker builds, and DVC push/pull run on ARC runner pods.
* **Layer 2 - Jenkins (heavy compute & continuous delivery):** A self-hosted Jenkins controller on the bare-metal host (`hive-mind`) receives webhooks through a zero-open-ports Cloudflare Tunnel. Jenkins dispatches ephemeral agent pods inside `k3s` via the `jenkins-agent` ServiceAccount across isolated pipeline stages:
    * `Prepare Toolchain` - Provisions standalone `git` and `kubernetes-client` via Pixi, mounting the host NVMe Rattler cache for sub-second dependency resolution.
    * `Static Quality & Invariants` - Executes Ruff linting, MyPy type checks, license compliance auditing, and OKF graph validation concurrently.
    * `Unit & Integration Tests` - Runs pytest suites with 78% line coverage threshold and verifies split conformal coverage error bounds.
    * `Documentation Strict Build` - Validates OKF knowledge graph and verifies doc integrity via `zensical build --strict`.
    * `Model Training & Checkpoint (GPU)` - Runs on `main` using bare-metal GPU time-slicing on `hive-mind`, pulling DVC datasets, training with early stopping and best weight restoration, exporting ONNX, and logging to DagsHub MLflow.
    * `Deployment Rollout` - Automatically applies [deploy/k8s/api-deployment.yaml](file:///home/benni/Documents/antigravity_workspace/taxon-vision-mlops/deploy/k8s/api-deployment.yaml) and monitors zero-downtime rolling update status in the `taxon-vision` namespace.

## Outstanding Implementation Tasks

* **Data Pipeline Execution:** Actual execution of DVC pipelines to ingest and transform iNaturalist/GBIF DarwinCore observation data.
* **Model Training & Benchmarking:** Implementation of the multi-backbone Pareto benchmarking (`BioCLIP-2`, `DINOv3`, `MobileNetV4`) as defined in the strategy report.
* **Inference Optimisation:** ONNX Runtime export with static INT8 ONNX PTQ for low-latency inference.
* **Uncertainty Calibration:** Verification of Split Conformal Prediction coverage bounds and OOD detection logic on full observation splits.
* **Web Service & Dashboard:** Validation and enhancement of the unified FastAPI machine REST API and the server-rendered Jinja2/HTMX operator dashboard.
* **Observability Stack:** Deployment of the Prometheus telemetry stack and Grafana dashboards for monitoring concept drift and latency budgets.

## Local Development Stack

For local development and debugging, a `docker-compose.local.yaml` is provided at the
repository root. It spins up the infrastructure components that cannot run inside Pixi:

* **MinIO** (ports `9000` / `9001`) - S3-compatible local object store for DVC remotes
  and S3 streamer development without hitting real AWS.
* **Prometheus** (port `9090`) - scrapes the local FastAPI `/metrics` endpoint at
  `host.docker.internal:8000`. Configuration lives in `config/prometheus.yaml`.
* **Grafana** (port `3000`) - connected to the local Prometheus instance for building
  and testing dashboards before exporting to production.

Start the stack with:

```bash
docker compose -f docker-compose.local.yaml up -d
```

The MLflow local tracking server is managed via Pixi (not Docker) to keep it inside the
correct Python environment:

```bash
pixi run -e dev mlflow
```

This starts MLflow on `http://localhost:5000`, persisting runs to `./mlruns` and
`./mlflow.db`.
