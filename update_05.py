with open('docs/latex/technical_report/chapters/05_containerisation_dockerfile_and_compose_manifests.tex', 'r') as f:
    content = f.read()

old_content = r"""The installation directive \texttt{RUN pixi install --frozen -e ci} instructs Pixi to instantiate the \texttt{ci} environment defined within the \texttt{pixi.lock} file. The \texttt{--frozen} flag enforces strict adherence to the cryptographic hashes and exact package versions recorded in the lockfile, guaranteeing deterministic reproducibility and thwarting supply chain substitution attacks.

The \texttt{ci} environment is explicitly formulated to exclude the \texttt{gpu} and \texttt{dev} feature sets. As verified within the \texttt{pixi.toml} and \texttt{pixi.lock} specifications, the \texttt{ci} environment resolves exclusively to CPU-optimised binaries:
\begin{itemize}
    \item \texttt{torch-2.5.1+cpu-cp312-cp312-linux\_x86\_64.whl}
    \item \texttt{torchvision-0.20.1+cpu-cp312-cp312-linux\_x86\_64.whl}
    \item \texttt{onnxruntime-1.31.0}
\end{itemize}

This exclusion of the \texttt{gpu} feature set purges the container of the \texttt{nvidia-cuda-runtime-cu12}, \texttt{nvidia-cudnn-cu12}, and associated libcudart dependencies. The resulting inference environment is lean, containing only the specific instruction sets required to execute the INT8-quantised ONNX models against the host CPU."""

new_content = r"""The container construction enforces a strict architectural divergence between the minimal runtime container (\texttt{-e ci}) and the CI test execution environment (\texttt{-e ci-dev}). The \texttt{ci} environment specifies \texttt{features = ["cpu"]}, deliberately omitting the multi-gigabyte CUDA libraries to optimize for lightweight Docker deployments. The \texttt{ci-dev} environment specifies \texttt{features = ["dev", "cpu"]} to facilitate rapid CPU continuous integration test execution (\texttt{pytest}, \texttt{ruff}, \texttt{mypy}, pre-commit gates). Additionally, local environments include \texttt{dev} (\texttt{features = ["dev", "gpu"]}) for full developer workstation setup, and \texttt{default} (\texttt{features = ["gpu"]}) for a minimal GPU runtime.

The installation directive \texttt{RUN pixi install --frozen -e ci} instructs Pixi to instantiate the \texttt{ci} environment defined within the \texttt{pixi.lock} file. The \texttt{--frozen} flag enforces strict adherence to the cryptographic hashes and exact package versions recorded in the lockfile, guaranteeing deterministic reproducibility and thwarting supply chain substitution attacks.

The \texttt{ci} environment is explicitly formulated to exclude the \texttt{gpu} and \texttt{dev} feature sets. As verified within the \texttt{pyproject.toml} and \texttt{pixi.lock} specifications, the \texttt{ci} environment resolves exclusively to CPU-optimised binaries:
\begin{itemize}
    \item \texttt{torch-2.5.1+cpu-cp312-cp312-linux\_x86\_64.whl}
    \item \texttt{torchvision-0.20.1+cpu-cp312-cp312-linux\_x86\_64.whl}
    \item \texttt{onnxruntime-1.31.0}
\end{itemize}

This exclusion of the \texttt{gpu} feature set purges the container of the \texttt{nvidia-cuda-runtime-cu12}, \texttt{nvidia-cudnn-cu12}, and associated libcudart dependencies. The resulting inference environment is lean, containing only the specific instruction sets required to execute the INT8-quantised ONNX models against the host CPU."""

content = content.replace(old_content, new_content)

with open('docs/latex/technical_report/chapters/05_containerisation_dockerfile_and_compose_manifests.tex', 'w') as f:
    f.write(content)
