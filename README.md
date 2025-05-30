# Fourier Sliced-Wasserstein (FSW) embedding — a PyTorch-based library

This package provides an implementation of the **Fourier Sliced-Wasserstein (FSW) embedding** for multisets and measures, introduced in our [ICLR 2025 paper](https://iclr.cc/virtual/2025/poster/30562):

> **Fourier Sliced-Wasserstein Embedding for Multisets and Measures**  
> Tal Amir, Nadav Dym  
> *International Conference on Learning Representations (ICLR), 2025*

---

## 🔧 Installation

To install the package:

```bash
pip install fswlib
```

This package includes an optional custom CUDA implementation. When working with sparse weight matrices (e.g., sparse graphs), it can be approximately 2× faster than the pure-PyTorch implementation.  
To compile it, run:

```bash
fswlib-build
```

---

## 👨🏻‍🔧 Maintainer

This library is maintained by [**Tal Amir**](https://tal-amir.github.io)  
Contact: [talamir@technion.ac.il](mailto:talamir@technion.ac.il)


---

## 📄 Citation

If you use this library in your research, please cite our paper:

```bibtex
@inproceedings{amir2025fsw,
  title={Fourier Sliced-{W}asserstein Embedding for Multisets and Measures},
  author={Tal Amir and Nadav Dym},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2025}
}
```

---

## 🔗 Links

- **Paper**: [ICLR 2025](https://iclr.cc/virtual/2025/poster/30562)  
- **Code**: [GitHub repository](https://github.com/tal-amir/fswlib)
