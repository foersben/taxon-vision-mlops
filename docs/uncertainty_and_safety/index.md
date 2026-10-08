
# Chapter 4: Uncertainty & Safety

Uncalibrated softmax probabilities are mathematically deceptive and pose an unacceptable epistemic risk in biological monitoring. This chapter outlines the rigorous safety mechanisms that guarantee the reliability of TaxonVision-MLOps predictions.

We document the Split Conformal Prediction theory that provides mathematical coverage guarantees, ensuring that the true species is captured within a dynamic prediction set. We also detail the Energy-Based Out-Of-Distribution (OOD) detection systems that filter out anomalous or empty frames natively in the feature space. When ambiguity is detected, the system safely triggers active learning pipelines and human-in-the-loop (HITL) triage to incrementally improve the taxonomic knowledge graph.

## Thematic Sections

* **Conformal Prediction Theory**: Establishing distribution-free mathematical coverage bounds.
* **OOD & Empty Frame Detection**: Utilizing energy-based scoring for robust anomaly rejection.
* **Active Learning & Human Review**: Query strategies and triage routing for ambiguous predictions.
