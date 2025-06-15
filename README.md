# fswlib: A PyTorch Library for the Fourier Sliced-Wasserstein (FSW) Embedding

This package provides an implementation of the **Fourier Sliced-Wasserstein (FSW) embedding**, introduced in our ICLR 2025 paper [1].

- [1] Tal Amir & Nadav Dym. "Fourier Sliced-Wasserstein Embedding for Multisets and Measures." *International Conference of Learning Representations (ICLR)*, 2025. URL: https://iclr.cc/virtual/2025/poster/30562

---

## 📦 Requirements

- **Python** ≥ 3.10.3 (released March 2022)  
- **PyTorch** ≥ 2.1.0 (released October 2023)  
- **NumPy** ≥ 1.24.4 (released June 2023)  
  
The core package has been tested on **Linux** and **Windows**.  
It may also run on **macOS**, though this has not been verified.  


---

## 🔧 Installation

To install the package:

```bash
pip install fswlib
```
The core package runs on both **CPU** and **CUDA-enabled GPUs**, using PyTorch's standard CUDA backend.  

In addition, it includes an optional **custom CUDA extension** that can provide up to 2× speedup for sparse weight matrices (e.g., sparse graphs). This extension is currently supported only on **Linux**.

  
To compile the optional extension, run:

```bash
fswlib-build
```


---

## 📘 Usage Example

Below is a basic usage example of the `FSWEmbedding` class.  

For more examples, see the `examples/` [directory](https://github.com/tal-amir/fswlib/tree/main/examples) of the GitHub repository.  
Full API documentation is available at [https://tal-amir.github.io/fswlib](https://tal-amir.github.io/fswlib).


```python
import torch
from fswlib import FSWEmbedding

# Configuration
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = torch.float32
d = 15     # Dimension of multiset elements
n = 50     # Multiset size
m = 123    # Embedding output dimension

# Create FSW embedding module for multisets/measures over ℝ^d
embed = FSWEmbedding(d_in=d, d_out=m, device=device, dtype=dtype)

# --- Single input multiset ---
X = torch.randn(size=(n, d), device=device, dtype=dtype)
W = torch.rand(n, device=device, dtype=dtype)  # Optional weights

X_emb = embed(X, W)  # Embeds a weighted multiset
X_emb = embed(X)     # Embeds X assuming uniform weights

# --- A batch of input multisets ---
batch_dims = (5,3,7,9)
Xb = torch.randn(size=batch_dims+(n,d), device=device, dtype=dtype)
Wb = torch.rand(batch_dims+(n,), device=device, dtype=dtype)
Xb_emb = embed(Xb, Wb)

print(f"Dimension of multiset elements: {d}")
print(f"Embedding dimension: {m}")
print(f"\nOne multiset X of size {n}:")
print("X shape:", X.shape)
print("embed(X) shape:", X_emb.shape)

batch_dim_str = "×".join(str(b) for b in batch_dims)
print(f"\nBatch of {batch_dim_str} multisets, each of size {n}:")
print("Xb shape:", Xb.shape)
print("embed(Xb) shape:", Xb_emb.shape)
```

Output:
```
Dimension of multiset elements: 15
Embedding dimension: 123

One multiset X of size 50:
X shape: torch.Size([50, 15])
embed(X) shape: torch.Size([123])

Batch of 5×3×7×9 multisets, each of size 50:
Xb shape: torch.Size([5, 3, 7, 9, 50, 15])
embed(Xb) shape: torch.Size([5, 3, 7, 9, 123])
```

The example below illustrates the difference between the core embedding, which is invariant to the input multiset size, and an embedding that explicitly encodes it.
```python
# --- Encoding multiset size (total mass) ---
# By default, the embedding is invariant to the input multiset size, since it
# treats inputs as *probability measures*.
# Set `encode_total_mass = True` to make the embedding encode the size of the
# input multisets, or, more generally, the total mass (i.e. sum of weights).
embed_total_mass_invariant = FSWEmbedding(d_in=d, d_out=m, device=device, dtype=dtype)
embed_total_mass_aware =     FSWEmbedding(d_in=d, d_out=m, encode_total_mass=True, device=device, dtype=dtype)

# Two multisets of different size but identical element proportions
X = torch.rand(3, d, device=device, dtype=dtype)
v1, v2, v3 = X[0], X[1], X[2]

X1 = torch.stack([v1, v2, v3])
X2 = torch.stack([v1, v1, v2, v2, v3, v3])

# Embedding *without* total mass encoding
X1_emb = embed_total_mass_invariant(X1)
X2_emb = embed_total_mass_invariant(X2)

# Embedding *with* total mass encoding
X1_emb_aware = embed_total_mass_aware(X1)
X2_emb_aware = embed_total_mass_aware(X2)

# Measure the differences
diff_invariant = torch.norm(X1_emb - X2_emb).item()
diff_aware = torch.norm(X1_emb_aware - X2_emb_aware).item()

print("Two different-size multisets with identical element proportions:")
print("X₁ = {v1, v2, v3},   X₂ = {v1, v1, v2, v2, v3, v3}")
print("Embedding difference: ‖Embed(X₁) − Embed(X₂)‖₂")
print(f"With total mass encoding:     {diff_aware}")
print(f"Without total mass encoding:  {diff_invariant:.2e}")
```

Output:
```
Two different-size multisets with identical element proportions:
X₁ = {v1, v2, v3},   X₂ = {v1, v1, v2, v2, v3, v3}
Embedding difference: ‖Embed(X₁) − Embed(X₂)‖₂
With total mass encoding:     3.0
Without total mass encoding:  5.09e-07
```

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

---

## 👨🏻‍🔧 Maintainer

This library is maintained by **Tal Amir**  
Homepage: [https://tal-amir.github.io](https://tal-amir.github.io)  
EMail: [talamir@technion.ac.il](mailto:talamir@technion.ac.il)

