import re

with open('docs/latex/technical_report/chapters/02_bare_metal_host_and_toolchain_interlock.tex', 'r') as f:
    content = f.read()

# Make changes to Section 2.1
new_desc_21 = r"""Central to the computational capacity of the \texttt{hive-mind} node is the AMD Ryzen 9 9950X CPU (16 physical cores, 32 threads, 4.3 GHz base / 5.7 GHz boost) and 64 GB DDR5-6000 ECC RAM, operating alongside the NVIDIA GeForce RTX 5070 Ti hardware accelerator. This graphics processing unit represents the vanguard of the Blackwell architecture, characterised by Streaming Multiprocessor (SM) 120 and Compute Capability 12.0. Equipped with 16 Gigabytes of GDDR7 Video Random Access Memory (VRAM), the accelerator guarantees unprecedented memory bandwidth for high-resolution image batch processing."""

content = re.sub(
    r'Central to the computational capacity of the \\texttt\{hive-mind\} node is the integration of the NVIDIA GeForce RTX 5070 Ti hardware accelerator\. This graphics processing unit represents the vanguard of the Blackwell architecture, characterised by Streaming Multiprocessor \(SM\) 120 and Compute Capability 12\.0\. Equipped with 16 Gigabytes of GDDR7 Video Random Access Memory \(VRAM\), the accelerator guarantees unprecedented memory bandwidth for high-resolution image batch processing\.',
    new_desc_21,
    content,
    flags=re.DOTALL
)

# Make changes to Section 2.2
new_desc_22 = r"""The project configuration, declared within \texttt{pyproject.toml}, establishes a four-environment specification designed to cater seamlessly to the diverse execution contexts. The \texttt{dev} environment targets the full developer workstation environment (\texttt{features = ["dev", "gpu"]}) heavily augmented with GPU dependencies. It explicitly requests PyTorch 2.7.0, Torchvision 0.22.0, and Torchaudio 2.7.0, statically linked against the \texttt{cu128} (CUDA 12.8) distribution, which is imperative for compatibility with the SM 120 architecture of the RTX 5070 Ti. Additionally, the \texttt{dev} environment encompasses TensorRT, ONNX Runtime GPU dependencies, Optuna, and MLflow to facilitate downstream model exportation and inference optimisation.

Conversely, the \texttt{ci-dev} environment is defined as a minimal CPU continuous integration runner (\texttt{features = ["dev", "cpu"]}). It substitutes the heavy CUDA binaries with PyTorch 2.5.1 sourced from the CPU-only channel. This architectural bifurcation ensures that ephemeral CI runners, which lack physical GPU accelerators, can execute the rapid test loops (\texttt{pytest}, \texttt{ruff}, \texttt{mypy}, pre-commit gates) deterministically without incurring the bandwidth and storage penalties associated with downloading multi-gigabyte CUDA libraries.

Beyond development contexts, the platform defines the \texttt{-e ci} environment (\texttt{features = ["cpu"]}) to serve as a minimal CPU container serving environment for lightweight Docker deployments, explicitly distinct from the CI test execution environment (\texttt{ci-dev}). Furthermore, the \texttt{default} environment provides a minimal GPU runtime (\texttt{features = ["gpu"]})."""

content = re.sub(
    r'The project configuration, declared within \\texttt\{pyproject\.toml\}, establishes a dual-environment specification designed to cater seamlessly to both the hardware-accelerated local development context and the resource-constrained Continuous Integration \(CI\) runners\..*?associated with downloading multi-gigabyte CUDA libraries\.',
    new_desc_22,
    content,
    flags=re.DOTALL
)

with open('docs/latex/technical_report/chapters/02_bare_metal_host_and_toolchain_interlock.tex', 'w') as f:
    f.write(content)
