with open('docs/latex/technical_report/figures/fig_02_host_and_cluster_topology.tikz', 'r') as f:
    content = f.read()

old_label = r"Physical Bare-Metal Host: \texttt{hive-mind} (Ubuntu 26.04.1 LTS, AMD/Intel x86\_64, RTX 5070 Ti 16GB, NVMe)"
new_label = r"Physical Bare-Metal Host: \texttt{hive-mind} (Ubuntu 26.04.1 LTS, AMD Ryzen 9 9950X 16C/32T, 64\,GB DDR5 ECC, RTX 5070 Ti 16GB, NVMe)"

content = content.replace(old_label, new_label)

with open('docs/latex/technical_report/figures/fig_02_host_and_cluster_topology.tikz', 'w') as f:
    f.write(content)
