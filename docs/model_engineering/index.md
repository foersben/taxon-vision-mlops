
# Chapter 3: Model Engineering & Inference

This chapter focuses on the mathematical heart of the platform: the neural architectures and their highly optimized execution runtimes.

We cover the entire lifecycle of the classification engine, beginning with the selection and construction of the vision backbones (e.g., BioCLIP-2, DINOv3). The documentation walks through the GPU-accelerated training pipeline, the implementation of Class-Balanced Loss, and the rigorous Pareto benchmarking used to evaluate model candidates. Finally, it explores the critical optimization techniques - specifically static ONNX Post-Training Quantization (PTQ) and forward-hooked CAM-required to serve these models within a strict sub-25 ms latency budget.

## Thematic Sections

* **Vision Model Architecture**: Structure of the foundational feature extractors and linear heads.
* **Training Pipeline**: End-to-end execution of the model training and tuning loops.
* **Backbone Pareto Benchmarks**: Empirical analysis of accuracy, latency, and operational cost.
* **ONNX Runtime Optimization**: Static INT8 PTQ quantization for maximizing inference throughput.
