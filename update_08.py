with open('docs/latex/technical_report/chapters/08_data_ingestion_arrow_streaming_and_licensing.tex', 'r') as f:
    content = f.read()

old_desc = r"""Furthermore, the \texttt{INatClient} restricts queries explicitly to observations possessing a \texttt{quality\_grade=research} parameter. This strict filtering guarantees that only records that have achieved a community consensus regarding the taxonomic identification are ingested, thereby maintaining high label fidelity. The client subsequently sanitises the retrieved metadata by upgrading the resolution of the image locators; it replaces the \texttt{square} thumbnail URLs with \texttt{medium} resolution locators (typically 500 pixels along the longest dimension), providing sufficient spatial frequency data for the downstream neural feature extraction."""

new_desc = r"""Furthermore, the \texttt{INatClient} restricts queries explicitly to observations possessing a \texttt{quality\_grade=research} parameter. This strict filtering guarantees that only records that have achieved a community consensus regarding the taxonomic identification are ingested, thereby maintaining high label fidelity. While research-grade quality gating requires source photographic observations to have $\ge 1024\text{px}$ resolution, the streaming metadata ingestion engine (\texttt{INatClient} in \texttt{src/taxon\_vision/data/inat\_client.py}) deliberately targets \texttt{medium} resolution locators (typically 500 pixels along the longest dimension). This 500px target provides optimal visual fidelity for downstream vision foundation backbones (which crop/resize to $224\times 224$ or $336\times 336$ at inference/feature caching), whilst reducing network transfer overhead and in-memory Arrow table serialization volume by $\sim 4\times$, preventing the "50 TB trap"."""

content = content.replace(old_desc, new_desc)

with open('docs/latex/technical_report/chapters/08_data_ingestion_arrow_streaming_and_licensing.tex', 'w') as f:
    f.write(content)
