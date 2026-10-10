---
type: Reference
title: Deployment Runbook & Containerization
status: stable
stale_after: "2027-01-01T00:00:00Z"
version: 1.1
description: Operational instructions for containerization, zero-downtime rollouts, Cloudflare Tunnel ingress, and dataset scaling horizons.
tags: [docker, deployment, operations, cloudflare, scaling]
generated: {by: process:scaffold-init, at: "2026-10-02T10:00:00Z"}
verified: {by: process:agent-jules, at: "2026-10-10T17:30:00Z"}
---

# Deployment Runbook & Containerization

## Section 1: Local Development & Verification

The local development environment requires initialization of the dependent infrastructure and application services.

Start the local infrastructure stack (MinIO, Prometheus, Grafana) via Docker Compose:

```bash
docker compose -f docker-compose.local.yaml up -d
```

Start the Pixi-managed FastAPI service:

```bash
pixi run -e dev api
```

Start the local MLflow tracking server:

```bash
pixi run -e dev mlflow
```

## Section 2: Safe Production Deployment Invariants

Production deployments enforce strict invariants to guarantee uninterrupted service and correct operation.

* Zero-downtime rolling deployment strategy: Kubernetes `RollingUpdate` with `maxSurge: 1` and `maxUnavailable: 0`.
* Inference engine startup warmup: Readiness probe validating ONNX Runtime session execution with dummy input tensor (latency threshold < 20 ms) before receiving traffic.
* Calibration cache invariant: Split Conformal Prediction non-conformity threshold cache ($q_{\text{val}}$) must be validated and loaded into memory before the pod transitions to `Ready`.
* Automated canary verification & rollback: Prometheus metric gates monitoring p95 latency (< 50 ms) and HTTP 5xx error rate (< 0.5%). Automated rollback if thresholds are violated.

## Section 3: Edge Ingress Topology via Cloudflare Tunnel

The system utilizes a zero-open-ports architecture. The `cloudflared` daemon operates via outbound-only multiplexed QUIC/HTTP2 tunnels to Cloudflare Edge PoPs, completely isolating the internal Kubernetes/Docker network from public inbound scanning.

The `cloudflared` daemon configuration routes the public hostname to the internal service endpoint:

```yaml
tunnel: <TUNNEL_ID>
credentials-file: /etc/cloudflared/credentials.json

ingress:
  - hostname: taxon.benjamin-foerster.eu
    service: http://localhost:30080
  - service: http_status:404
```

The packet flow visualizes the path from client to backend:

```mermaid
flowchart LR
    A[Client] --> B[Cloudflare Edge PoP (WAF & Access Filter)]
    B --> C[Cloudflare Tunnel (cloudflared)]
    C --> D[FastAPI ASGI / ONNX Runtime Service]
```

## Section 4: Restricted Private Access Control Topology

The preview/operator service is exposed to a selected external audience using two distinct Cloudflare defense layers.

* Layer A (Programmatic API Access Key via WAF):
    * Custom Cloudflare WAF Expression filtering HTTP request headers: `http.request.headers["X-Taxon-Access-Key"][0] eq "REDACTED_ACCESS_KEY"` (or custom HMAC query token).
    * Requests lacking the valid key are dropped immediately at the Cloudflare Edge PoP with HTTP 403 Forbidden, protecting the backend from compute starvation.
* Layer B (Operator Dashboard Identity Gate via Cloudflare Zero Trust):
    * Cloudflare Access Application wrapping operator paths (`/ui`, `/triage`, `/docs`).
    * Identity provider whitelist (One-Time PIN via email whitelist or Google Workspace SSO).
    * Prevents unauthorized browser access to the HTMX triage dashboard while allowing authenticated taxonomists.

## Section 5: Dataset Scaling Horizons

The architectural breakdown details how TaxonVision-MLOps scales across data horizons.

| Horizon | Scale | Infrastructure & Architecture Details | Validation & Verification |
| --- | --- | --- | --- |
| Dev Baseline (Current) | 518 observations, 10 taxa | Local SQLite & disk cache | Rapid offline unit/conformal testing |
| Horizon 1 (Pilot Perimeter) | ~5,000 observations, 10 curated target taxa | Single GPU training, DVC versioning on DagsHub | Strict CC-BY/CC0 verification |
| Horizon 2 (Regional Production) | ~50,000 observations, 100 taxa | Multi-GPU Distributed Data Parallel (DDP), S3/MinIO bucket versioning | Automated concept drift detection |
| Horizon 3 (Global Scale) | 100,000+ exemplars, ~50 TB raw imagery | Distributed zero-copy streaming with Ray Data and PyTorch WebDataset, fine-tuning foundation vision backbones (DINOv2, BioCLIP-2) | Conformal calibration on stratified holdouts |

## Section 6: Cluster Bootstrapping vs. Continuous Delivery Lifecycle

The deployment infrastructure separates Day-0 cluster prerequisites from Day-2 continuous deployment pipelines:

* **Day-0 Cluster Bootstrap (One-Time Setup):**
    * Configures host-level infrastructure, hardware device plugins, and authentication boundaries on `hive-mind` (`192.168.178.30`).
    * Applied once by cluster administrators using `kubectl apply`:
        * [deploy/k8s/jenkins-agent-rbac.yaml](file:///home/benni/Documents/antigravity_workspace/taxon-vision-mlops/deploy/k8s/jenkins-agent-rbac.yaml): Provisions `jenkins-ci` and `taxon-vision` namespaces, machine ServiceAccount `jenkins-agent`, and RBAC bindings.
        * [deploy/k8s/nvidia-time-slicing-config.yaml](file:///home/benni/Documents/antigravity_workspace/taxon-vision-mlops/deploy/k8s/nvidia-time-slicing-config.yaml): Partitions the physical RTX 5070 Ti into 4 virtual GPU slices in `kube-system`.
    * Storing these manifests in Git provides Infrastructure as Code (IaC) without requiring repetitive manual execution.
* **Day-2 Continuous Delivery (Automated via Jenkins):**
    * Operates entirely within the CI/CD pipeline triggered by Git commits and PR events.
    * Eliminates brittle host-level cronjobs, systemd file watchers, or local `git pull` shell loops on the server.
    * Jenkins checks out code via authenticated GitHub PAT, executes test gates, builds documentation, and applies [deploy/k8s/api-deployment.yaml](file:///home/benni/Documents/antigravity_workspace/taxon-vision-mlops/deploy/k8s/api-deployment.yaml) directly via the in-cluster `jenkins-agent` ServiceAccount.

## Section 7: NVIDIA Device Plugin GPU Time-Slicing Specification

Consumer and workstation GPUs (such as our NVIDIA GeForce RTX 5070 Ti) lack hardware Multi-Instance GPU (MIG) slicing. Without time-slicing, requesting `nvidia.com/gpu: 1` places an exclusive lock on the entire card, starving co-located workloads.

### Time-Slicing Configuration Architecture

The cluster exposes 4 virtual GPU slots using the official NVIDIA Kubernetes Device Plugin:

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: nvidia-device-plugin-config
  namespace: kube-system
data:
  config.yaml: |
    version: v1
    sharing:
      timeSlicing:
        resources:
          - name: nvidia.com/gpu
            replicas: 4
```

The `nvidia-device-plugin-daemonset` in `kube-system` mounts this ConfigMap at `/etc/config/config.yaml` and reads environment variable `CONFIG_FILE=/etc/config/config.yaml`.

### Cluster Verification Command

Validate allocatable GPU slices on `hive-mind`:

```bash
kubectl get node hive-mind -o jsonpath='{.status.allocatable.nvidia\.com/gpu}{"\n"}'
# Output: 4
```

### Application VRAM Guardrails

Because hardware time-slicing multiplexes compute cores while sharing device memory globally, PyTorch batch training pipelines enforce a strict allocation cap:

```python
import os
import torch

if torch.cuda.is_available():
    fraction = float(os.getenv("TAXON_CUDA_MEMORY_FRACTION", "0.7"))
    torch.cuda.set_per_process_memory_fraction(fraction)
```

This reserves 30% (4.8 GB) of VRAM for live ONNX Runtime FastAPI inference and host display outputs.

## Section 8: Automated Kubernetes Continuous Deployment Pipeline

When commits are pushed to `main` or `feature/phase1-foundations`, the declarative pipeline in [Jenkinsfile](file:///home/benni/Documents/antigravity_workspace/taxon-vision-mlops/Jenkinsfile) executes an automated zero-downtime rolling update:

### Pipeline Execution Flow

* **Step 1 (Toolchain Provisioning):** The container runner installs `kubernetes-client` (from conda-forge, exposing `kubectl`) and standalone `git` into `/root/.pixi/bin`.
* **Step 2 (Verification Gates):** Executes Ruff linting, MyPy type checks, License compliance audits, OKF validation, unit tests (with 78% line coverage threshold), integration tests, and strict Zensical build.
* **Step 3 (Deployment Rollout):**
    * Applies [deploy/k8s/api-deployment.yaml](file:///home/benni/Documents/antigravity_workspace/taxon-vision-mlops/deploy/k8s/api-deployment.yaml) using `jenkins-deployer-role` permissions.
    * Triggers zero-downtime rolling update: `kubectl rollout restart deployment/taxon-vision-api -n taxon-vision`.
    * Awaits readiness probe verification: `kubectl rollout status deployment/taxon-vision-api -n taxon-vision --timeout=120s`.
