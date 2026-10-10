---
type: Reference
title: GPU Resource Scheduling & Multiplexing
status: stable
stale_after: "2027-01-01T00:00:00Z"
version: 1.0
description: Architectural strategies for managing single-GPU scheduling conflicts between Training, CI/CD, and Web Inference workloads.
tags: [infrastructure, gpu, k3s, scheduling]
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

## The Hybrid Recommendation

To satisfy the competing demands of high-throughput training, agile CI/CD, and real-time user inference on a single GPU, we adopt a hybrid architecture:

* **Enable Time-Slicing:** Configure the `k3s` NVIDIA plugin for Time-Slicing via [deploy/k8s/nvidia-time-slicing-config.yaml](file:///home/benni/Documents/antigravity_workspace/taxon-vision-mlops/deploy/k8s/nvidia-time-slicing-config.yaml), allowing the Web API to coexist permanently with background tasks.
* **Enforce VRAM Fractions:** Hardcode per-process memory limits in training via `configure_cuda_memory_budget()` in [src/taxon_vision/models/training/runner.py](file:///home/benni/Documents/antigravity_workspace/taxon-vision-mlops/src/taxon_vision/models/training/runner.py), paired with ultra-light ONNX quantization for FastAPI endpoints.
* **Queue Heavy Backends:** Implement Kueue exclusively for batch jobs. This ensures that while the Web API is always alive via Time-Slicing, multiple heavy backend tasks do not crash each other, but instead queue up politely.

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

Applying this ConfigMap exposes 4 assignable `nvidia.com/gpu` slots in `k3s`:

* **Slot 1 (Web API):** Dedicated to [deploy/k8s/api-deployment.yaml](file:///home/benni/Documents/antigravity_workspace/taxon-vision-mlops/deploy/k8s/api-deployment.yaml).
* **Slot 2 (Batch Training):** Claimed during DVC pipeline execution.
* **Slots 3-4 (Ephemeral / CI):** Disposable runner pods or local interactive research.

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
pixi run -e dev python -m taxon_vision.models.trainer --cuda-memory-fraction 0.7
```

On CPU-only development environments, `configure_cuda_memory_budget()` gracefully degrades to a no-op, preserving portability across Darwin, Linux CPU, and CUDA workstations.
