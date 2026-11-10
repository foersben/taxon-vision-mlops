---
type: Agent Rule
title: "Rule 07: Hardware Invariants"
status: stable
stale_after: "2027-01-01T00:00:00Z"
version: 1.0
description: "Hardware representation, zero-hallucination policy, and infrastructure rationalization standards."
tags: [hardware, governance, infrastructure]
generated: {by: process:scaffold-init, at: "2026-10-25T07:45:00Z"}
verified: {by: process:scaffold-init, at: "2026-10-25T07:45:00Z"}
trigger: always_on
severity: critical
---

# Mandates: Hardware Invariants

Hardware specifications and infrastructure descriptions must adhere strictly to the following architectural invariants across all documentation, reports, and code comments.

## 1. Functional Necessity Principle

Hardware specifications must **only** be mentioned if they directly dictate an architectural equation, a mathematical resource ceiling, or a systems-level parameter (e.g. CPU thread affinity masks, CUDA memory fractions, or filesystem block allocation flags). If a hardware attribute has no direct consequence on the software architecture or execution configuration, **omit it completely**.

## 2. Immutable Hardware Ground Truth (Zero-Hallucination Policy)

Never invent, alter, upgrade, or generalize the hardware platform. Never substitute cloud GPU types (e.g. A100, H100, V100, L4, T4) or other consumer cards (e.g. RTX 3090, RTX 4090).

When hardware must be stated, use **only** the following exact verified parameters:

| Component | Canonical Specification | Permitted Architectural Reason to Mention | Strictly Prohibited Trivia (Never Mention) |
| :--- | :--- | :--- | :--- |
| **GPU** | NVIDIA GeForce RTX 5070 Ti (16 GB GDDR7 VRAM) | Setting PyTorch memory budget ($f=0.7 \implies 11.2\text{ GB}$ training, leaving $4.8\text{ GB}$ for ONNX serving & system reserve); Blackwell architecture (compute capability 12.0) requiring CUDA runtime $\ge 12.8$. | Partner brands (Zotac, MSI, ASUS), cooler type, clock speeds, display outputs, PCIe slot index. |
| **CPU** | Intel Core i7-14700K (20 cores / 28 threads: 8 P-cores, 12 E-cores) | Explicit CPU thread affinity pinning via `taskset`: hot numerical math on 16 P-core threads (`0-15`), async I/O, Arrow streaming, and telemetry on 12 E-core threads (`16-27`). | Motherboard model (ASRock), chipset (B760M), form factor (mATX), boost frequencies (5.5 GHz / 4.3 GHz), CPU socket type. |
| **RAM** | 128 GB DDR5 | High-throughput in-memory feature embedding caching and LanceDB vector index capacity planning in system memory. | Brand names (Crucial), slot topologies (`A2/B2`), timings, CAS latency. |
| **Storage** | 2 TB PCIe Gen4 NVMe SSD | NVMe throughput, BTRFS transparent `zstd:3` compression, non-CoW (`chattr +C`) allocation for database and DVC storage to eliminate physical block fragmentation. | SSD model names (FireCuda 530), controller chips (Phison PS5018-E18), TBW endurance ratings (`2,550 TBW`), slot identifiers (`M2_1`). |
| **OS / Host** | Ubuntu 26.04 LTS on LUKS-encrypted root storage | Kernel-level cgroups v2 process isolation, initramfs Dropbear remote unlocking, NVIDIA driver baseline (615.71.09 / CUDA UMD 13.4). | Desktop chassis, power supply, fan curves, peripheral connections. |

## 3. Ban on Cloud Superstition & Hyperbolic Formulations

Never justify bare-metal or on-premise infrastructure using unscientific or hyperbolic assertions about public cloud volatility.

### Forbidden Formulations

* **Never claim:** "Cloud infrastructure is fundamentally too volatile or unstable for deep vision models." (Public cloud providers offer deterministic, enterprise-grade GPU instances).
* **Never claim:** "Hypervisor overhead or jitter obscures true latency metrics and invalidates benchmarking." (Modern bare-metal and pass-through cloud instances have negligible virtualization jitter).
* **Never claim:** "Bare-metal hardware prevents physical clock-speed variance." (Consumer CPUs and GPUs throttle based on thermal envelopes; determinism stems from fixed seeds, thread pinning, and containerized runtime locking).

### Mandatory Systems-Engineering Justifications

Always ground on-premise infrastructure choices in verifiable unit economics and system sovereignty:

* **Total Cost of Ownership (TCO) & Unit Economics:** Persistent dedicated cloud GPU instances with comparable modern architectures incur continuous, high hourly costs ($1.00-$2.50+/hr). Dedicated on-premise silicon amortizes upfront capital expenditure within 2 to 3 months for continuous CI/CD, local tuning sweeps, and regression testing.
* **Zero-Egress Data Economics:** Staging and versioning multi-gigabyte embedding splits and thousands of image crops against a local, in-cluster S3-compatible store (SeaweedFS) eliminates cloud network egress penalties entirely.
* **Direct Kernel & Resource Control:** A dedicated host allows unrestricted Linux kernel tuning: cgroups v2 resource capping, transparent BTRFS compression, direct CPU thread isolation via `taskset`, and custom time-slicing profiles across local Kubernetes namespaces without cloud hypervisor quotas.

## 4. Standardized Phrasing Templates

### High-Level Infrastructure Summary

> "To ensure cost-effective continuous training, regression testing, and low-latency serving without cloud egress overhead, the platform targets a dedicated compute node equipped with an Intel Core i7-14700K processor (8 Performance-cores, 12 Efficient-cores), 128 GB DDR5 RAM, and an NVIDIA GeForce RTX 5070 Ti GPU (16 GB VRAM). Computational workloads are segregated at the OS level: compute-intensive neural training and inference loops are pinned to the Performance cores, while asynchronous I/O and telemetry are handled by the Efficient cores."

### Storage & Filesystem Description

> "Local storage I/O is backed by a 2 TB PCIe Gen4 NVMe drive running a BTRFS filesystem with transparent `zstd:3` compression. To prevent block fragmentation caused by Copy-on-Write (CoW) allocation during continuous DVC chunk mutations and database writes, persistent directories for SeaweedFS, SQLite, and MLflow are initialized with the non-CoW attribute (`chattr +C`), guaranteeing contiguous block allocation across the NVMe bus."

### GPU Time-Slicing & Memory Partitioning

> "Because single consumer GPUs lack hardware Multi-Instance GPU (MIG) partitioning, the cluster employs the NVIDIA Kubernetes device plugin in time-slicing mode to expose four virtual GPU slots. Memory isolation is enforced in software by setting PyTorch's per-process memory fraction to $0.7$, reserving $11.2\text{ GB}$ VRAM for training pods and guaranteeing an isolated $4.8\text{ GB}$ envelope for the active INT8 ONNX serving process."
