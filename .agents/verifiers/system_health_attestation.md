---
type: Attested Computation
title: System Health Attestation Seal
status: stable
runtime: pixi run -e dev (Python 3.12)
stale_after: "2027-01-01T00:00:00Z"
version: 1.0
description: Attested computation seal verifying full repository static analysis and test gates.
tags: [attestation, verification, pre-commit, seal]
generated: {by: "process:attest-seal", at: "2026-10-30T00:15:55Z"}
verified: {by: "process:attest-seal", at: "2026-10-30T00:15:55Z"}
sources:
  - id: toolchain_spec
    resource: .agents/toolchain.yaml
---

# System Health Attestation Seal

* **Timestamp:** `2026-10-30T00:15:55Z`
* **Overall Status:** `PASSED`
* **Runtime Environment:** Pixi `dev` (GPU/CUDA 12.8) and `ci-dev` (CPU)

## Verification Gate Results

| Gate | Command | Exit Code | Status |
| :--- | :--- | :--- | :--- |
| **Ruff Linter** | `pixi run -e dev ruff check .` | `0` | ✅ PASSED |
| **MyPy Strict Types** | `pixi run -e dev mypy src scripts` | `0` | ✅ PASSED |
| **Complexipy Invariants** | `pixi run -e dev complexipy src scripts --max-complexity-allowed 15 --check-script --no-ignore` | `0` | ✅ PASSED |
| **OKF v0.2 Governance** | `pixi run -e dev python scripts/governance/validate_okf.py` | `0` | ✅ PASSED |
| **Hardware Invariants** | `pixi run -e dev python scripts/governance/verify_hardware_invariants.py` | `0` | ✅ PASSED |
| **License Compliance** | `pixi run -e dev python -m taxon_vision.pipelines.audit_license` | `0` | ✅ PASSED |
| **Code Invariants Engine** | `pixi run -e dev python scripts/governance/verify_code_invariants.py` | `0` | ✅ PASSED |
| **Fast Targeted Tests** | `pixi run -e ci-dev pytest tests/unit/test_config.py tests/unit/test_pipelines.py tests/unit/test_data_components.py tests/unit/test_ood_energy.py tests/unit/test_conformal_bounds.py -q -o addopts=` | `0` | ✅ PASSED |

## Attestation Assertion

All 8 mandatory pre-commit verification gates executed with zero non-zero exit codes. Cognitive complexity strictly satisfies <= 15 with zero snapshot overrides. Open licensing compliance adheres to CC0/CC-BY/CC-BY-NC guidelines. OKF v0.2 knowledge graph consistency is formally verified.
