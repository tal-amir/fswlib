# Fourier Sliced-Wasserstein (FSW) Embedding

This package provides an implementation of the **Fourier Sliced-Wasserstein (FSW) embedding** for multisets and measures, as introduced in our [ICLR 2025 paper](https://iclr.cc/virtual/2025/poster/30562):

> **Fourier Sliced-Wasserstein Embedding for Multisets and Measures**  
> Tal Amir, Nadav Dym  
> *International Conference on Learning Representations (ICLR), 2025*

It is designed for integration with PyTorch-based pipelines and supports optional modules such as FSW-GNN.

- 🔍 FSW embedding for learning from unordered sets and distributions  
- 🧩 Modular design with optional submodules (e.g., FSW-GNN)  
- ⚡ Optional CUDA acceleration  
- 🔬 Designed for reproducible research  

---

## Installation

### 🔧 Basic installation

To install the package:

```bash
pip install fsw
```

This module includes an optional custom CUDA backend, which is ~2× faster than the pure PyTorch version when used on sparse graphs/weights. To compile the backend, run
```bash
fsw-build
```
