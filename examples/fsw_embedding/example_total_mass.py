import torch
from fswlib import FSWEmbedding

# Configuration
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = torch.float32
d = 15     # Input element dimension
n = 50     # Multiset size
m = 5    # Embedding output dimension

# By default, the embedding is invariant to the input multiset size, since it treats inputs as *probability measures*.
# Set `encode_total_mass = True` to make the embedding encode the size of the input multisets, or, more generally,
# the total mass (i.e. sum of weights).
embed_total_mass_invariant = FSWEmbedding(d_in=d, d_out=m, device=device, dtype=dtype)
embed_total_mass_aware =     FSWEmbedding(d_in=d, d_out=m, encode_total_mass=True,
                                          total_mass_encoding_method='homogeneous_legacy',
                                          total_mass_encoding_function='identity',
                                          device=device, dtype=dtype)
config2 = {
    "d_in":d, "d_out":m, "encode_total_mass":True,
                                          "total_mass_encoding_method":'homogeneous_legacy',
                                          "total_mass_encoding_function":'identity',
                                          "device":device, "dtype":dtype
}
embed_total_mass_aware = FSWEmbedding.from_config(config2)

# Two multisets with identical proportions but different cardinalities
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

ratio_invariant = X2_emb / X1_emb
ratio_aware = X2_emb_aware / X1_emb_aware

print()
print("Two different-size multisets with identical element proportions:")
print("X₁ = {a,b,c},   X₂ = {a,a,b,b,c,c}")
print("Embedding difference: ‖Embed(X₁) − Embed(X₂)‖₂")
print(f"With total mass encoding:     {diff_aware}")
print(f"Without total mass encoding:  {diff_invariant:.2e}")

print(f'Ratio invariant: {ratio_invariant}')
print(f'Ratio aware: {ratio_aware}')

# Output:

# Two different-size multisets with identical element proportions:
# X₁ = {v₁, v₂, v₃},   X₂ = {v₁, v₁, v₂, v₂, v₃, v₃}
# Embedding difference: ‖Embed(X₁) − Embed(X₂)‖₂
# With total mass encoding:     3.0
# Without total mass encoding:  5.09e-07
