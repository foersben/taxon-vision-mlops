---
type: Reference
title: GPU Resource Scheduling & Multiplexing
status: stable
stale_after: "2027-01-01T00:00:00Z"
version: 1.1
description: Architectural strategies for managing single-GPU scheduling conflicts between Training, CI/CD, and Web Inference workloads, enforcing the 4-slot time-slicing invariant and staging CPU discipline.
tags: [infrastructure, gpu, k3s, scheduling, time-slicing, deadlock-prevention]
generated: {by: "process:docs-librarian", at: "2026-11-10T16:00:00Z"}
verified: {by: "process:docs-librarian", at: "2026-11-10T16:00:00Z"}
---

# GPU Resource Scheduling & Multiplexing

## The Single-GPU Scheduling Bottleneck

Our bare-metal server (`hive-mind`) is equipped with a single, highly capable NVIDIA RTX 5070 Ti. Because consumer and prosumer tier GPUs do not support hardware-level Multi-Instance GPU (MIG) partitioning (a feature reserved for datacenter cards like the A100), Kubernetes natively treats the `nvidia.com/gpu: "1"` resource request as an exclusive lock.

By default, if a heavy model training pod claims the GPU, Kubernetes will actively block any other pod (such as an ARC runner executing CI tests, or a Web API pod serving the vRAG interface) from spinning up. They will languish in a `Pending` state until the training pod completes and relinquishes the lock. Given that model training can take days, this default behaviour is unacceptable for maintaining a responsive web interface or an agile CI/CD pipeline.

Below are the three architectural strategies available to resolve this resource contention, culminating in our recommended hybrid approach.

## Strategy 1: NVIDIA Time-Slicing & VRAM Budgeting (Recommended)

This strategy bypasses the exclusive lock mechanism by configuring the NVIDIA Kubernetes device plugin to present a single physical GPU as multiple virtual GPUs (e.g., `replicas: 4`). This tricks the Kubernetes scheduler into allowing multiple pods to mount the GPU concurrently.

* **Context Switching & Compute:** The NVIDIA driver handles the multiplexing at the hardware level. The Web API, ARC Runner, and Training pods all run simultaneously, with the GPU rapidly context-switching compute cycles between them.
* **The VRAM Danger:** While compute cycles are multiplexed safely, Video RAM (VRAM) is shared globally across all pods. If the training pod aggressively allocates 100% of the VRAM, the Web API pod will instantly crash with an Out Of Memory (OOM) error when a user requests an inference.
* **The Solution (Budgeting):** To safely execute this strategy, we enforce strict bounds at the software level. The Web API relies on highly optimised ONNX Runtime INT8 models, requiring a minuscule memory footprint (< 1 GB). Conversely, PyTorch training scripts are explicitly capped using `torch.cuda.set_per_process_memory_fraction(0.7)`. This guarantees that training can never exceed 70% of the VRAM (11.2 GB), permanently reserving the remaining 30% (4.8 GB) to guarantee the stability of the Web API and system displays.

## Strategy 2: Preemptive PriorityClasses (The Checkpoint-and-Kill Approach)

If workloads require 100% of the GPU memory without compromise, we can abandon Time-Slicing in favour of strict Kubernetes PriorityClasses and scale-to-zero paradigms (e.g., Knative).

* **Priority Assignment:** The Web API pod is assigned an ultra-high priority (`1000`), while Training pods are assigned a low priority (`500`).
* **Eviction Dynamics:** When a user accesses the web interface, Knative attempts to spin up the Web API pod. Recognising that the GPU is locked by the Training pod, the Kubernetes scheduler immediately evicts (kills) the Training pod to free the resource. The Web API serves the user's request. After a period of inactivity, the Web API scales back to zero, and the Training pod is automatically rescheduled.
* **Implementation Burden:** This strategy demands that all training scripts be engineered with extreme fault tolerance. They must aggressively save state checkpoints (weights, optimizer states, dataloader indices) to persistent volumes at high frequencies. Otherwise, every web request would result in catastrophic loss of training progress.

## Strategy 3: Native Kubernetes Queuing (Batch Scheduling)

For workloads that do not require real-time responsiveness, we can implement a native Kubernetes batch scheduler, such as Kueue or Volcano.

* **Job Serialization:** Instead of deploying long-running pods, workloads are submitted as discrete Jobs to a central queue. The queue manager evaluates the available cluster resources. If an ARC runner is executing a deployment, a newly submitted Training job will simply wait patiently in the queue.
* **Limitations:** While this elegantly solves contention between backend tasks (like preventing CI/CD pipelines and Model Training from colliding), it is fundamentally incompatible with a live Web API. Users expect a web interface to load in milliseconds, not to wait hours for a training run to finish.

## Fallback Strategy for Development

For environments lacking GPU hardware (such as local development machines), workloads are configured with a fallback to CPU emulation. This allows the API and pipelines to function gracefully, albeit at higher latencies, without failing instantly on missing CUDA drivers.

## The Hybrid Recommendation

To satisfy the competing demands of high-throughput training, agile CI/CD, and real-time user inference on a single GPU, we adopt a hybrid architecture:

* **Enable Time-Slicing:** Configure the `k3s` NVIDIA plugin for Time-Slicing via [deploy/k8s/nvidia-time-slicing-config.yaml](file:///home/benni/Documents/antigravity_workspace/taxon-vision-mlops/deploy/k8s/nvidia-time-slicing-config.yaml), allowing the Web API to coexist permanently with background tasks via standard Kubernetes GPU resource allocation (`nvidia.com/gpu: 1`).
* **Enforce VRAM Fractions:** Hardcode per-process memory limits in training via `configure_cuda_memory_budget()` in [src/taxon_vision/models/training/runner.py](file:///home/benni/Documents/antigravity_workspace/taxon-vision-mlops/src/taxon_vision/models/training/runner.py), paired with ultra-light ONNX quantization for FastAPI endpoints.
* **Queue Heavy Backends:** Implement Kueue exclusively for batch jobs. This ensures that while the Web API is always alive via Time-Slicing, multiple heavy backend tasks do not crash each other, but instead queue up politely.
* **Node Isolation:** Configure node taints and tolerations for dedicated ML worker nodes to ensure regular non-ML pods do not accidentally schedule onto scarce GPU resources.

## Kubernetes Time-Slicing Manifest Specification

The cluster-level time-slicing configuration is deployed via a ConfigMap targeted at the NVIDIA Kubernetes Device Plugin:

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

Applying this ConfigMap exposes 4 assignable `nvidia.com/gpu` slots in `k3s`. Under our production operational model on `hive-mind`, these 4 slots are strictly allocated:

$$\underbrace{1_{\text{ARC Runner}}}_{\text{arc-systems}} + \underbrace{2_{\text{Production Pods}}}_{\text{taxon-vision}} + \underbrace{1_{\text{Jenkins GPU Agent}}}_{\text{jenkins-ci (training)}} = 4_{\text{Total Hardware Slots}}$$

* **Slot 0 (ARC GitHub Actions Runner):** Dedicated to the standing self-hosted runner pod in `namespace: arc-systems` (`gpu-runner-deployment-xxfjd-8994t`), providing GPU acceleration for on-cluster GitHub Actions workflows.
* **Slots 1 & 2 (Production Serving Replicas):** Dedicated to the two high-availability production inference replicas in `namespace: taxon-vision` ([`deploy/k8s/overlays/production/patch-replicas.yaml`](file:///home/benni/Documents/antigravity_workspace/taxon-vision-mlops/deploy/k8s/overlays/production/patch-replicas.yaml)).
* **Slot 3 (Ephemeral Jenkins GPU Training Agent):** Claimed on-demand by the dynamic Jenkins training pod ([`deploy/k8s/jenkins-gpu-pod.yaml`](file:///home/benni/Documents/antigravity_workspace/taxon-vision-mlops/deploy/k8s/jenkins-gpu-pod.yaml)) during stage `Model Training & Checkpoint (GPU)`. The pod is terminated immediately upon pipeline completion to release the slot.

### Staging CPU Invariant & Deadlock Elimination

A critical architectural pitfall in single-node GPU time-slicing is **scheduling starvation deadlock**. If the staging environment (`namespace: taxon-vision-staging`) were permitted to claim a GPU slot, all 4 physical slots would be continuously occupied:

$$1_{\text{ARC}} + 2_{\text{Prod}} + 1_{\text{Staging}} = 4_{\text{Slots Occupied}}$$

Under this state, any incoming Jenkins CI build requesting `nvidia.com/gpu: 1` to train a candidate model would hang indefinitely in `Pending` (`0/1 nodes are available: 1 Insufficient nvidia.com/gpu`), causing the build to stall and report the agent as offline.

To resolve this contention at the architectural root:

* **Zero-Allocation CPU Serving:** Under Rule 02, our INT8-quantised ONNX serving hot path executes in $< 15\,\text{ms}$ on the Intel Core i7-14700K Performance cores without GPU acceleration.
* **CPU-Only Staging Overlay:** [`deploy/k8s/overlays/staging/patch-replicas.yaml`](file:///home/benni/Documents/antigravity_workspace/taxon-vision-mlops/deploy/k8s/overlays/staging/patch-replicas.yaml) strips `nvidia.com/gpu` entirely, constraining the staging pod to `cpu: "2"` and `memory: "4Gi"`.
* **Deterministic Scheduling Guarantee:** Staging consumes zero GPU slots, guaranteeing that the 4th GPU slot is permanently available for ephemeral Jenkins model training runs.

## Application VRAM Budgeting Mechanics

Because NVIDIA time-slicing does not isolate device memory, PyTorch training pipelines declare their allocation cap prior to CUDA memory initialization:

```python
import os
import torch

if torch.cuda.is_available():
    fraction = float(os.getenv("TAXON_CUDA_MEMORY_FRACTION", "0.7"))
    torch.cuda.set_per_process_memory_fraction(fraction)
```

CLI execution also exposes `--cuda-memory-fraction`:

```bash
pixi run -e dev python -m taxon_vision.pipelines.train --cuda-memory-fraction 0.7
```

On CPU-only development environments, `configure_cuda_memory_budget()` gracefully degrades to a no-op, preserving portability across Darwin, Linux CPU, and CUDA workstations.
