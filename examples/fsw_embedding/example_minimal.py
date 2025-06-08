import torch
from fswlib import FSWEmbedding

# Configuration
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = torch.float32
d = 15     # Input element dimension
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

# Output:

# Dimension of multiset elements: 15
# Embedding dimension: 123
#
# One multiset X of size 50:
# X shape: torch.Size([50, 15])
# embed(X) shape: torch.Size([123])
#
# Batch of 5×3×7×9 multisets, each of size 50:
# Xb shape: torch.Size([5, 3, 7, 9, 50, 15])
# embed(Xb) shape: torch.Size([5, 3, 7, 9, 123])


# --- Encoding multiset size (total mass) ---
# By default, the embedding is invariant to the input multiset size, since it treats inputs as *probability measures*.
# Set `encode_total_mass = True` to make the embedding encode the size of the input multisets, or, more generally,
# the total mass (i.e. sum of weights).
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

print()
print("Two different-size multisets with identical element proportions:")
print("X₁ = {v1, v2, v3},   X₂ = {v1, v1, v2, v2, v3, v3}")
print("Embedding difference: ‖Embed(X₁) − Embed(X₂)‖₂")
print(f"With total mass encoding:     {diff_aware}")
print(f"Without total mass encoding:  {diff_invariant:.2e}")


# Output:

# Two different-size multisets with identical element proportions:
# X₁ = {v1, v2, v3},   X₂ = {v1, v1, v2, v2, v3, v3}
# Embedding difference: ‖Embed(X₁) − Embed(X₂)‖₂
# With total mass encoding:     3.0
# Without total mass encoding:  5.09e-07
