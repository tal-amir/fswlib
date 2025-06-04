import torch

from fswlib import FSWEmbedding

dtype=torch.float32
device = 'cuda' if torch.cuda.is_available() else 'cpu'

d = 15  # dimension of input multiset elements
n = 50  # multiset size
m = 123 # embedding output dimension

# If False, input multisets are treated as uniform distributions over their elements,
# making the embedding invariant to the multiset size.
encode_total_mass = True

# Generate an embedding module
embed = FSWEmbedding(d_in=d, d_out=m, encode_total_mass=encode_total_mass, device=device, dtype=dtype)

print(f"Dimension of multiset elements: {d}\nEmbedding dimension: {m}")

print(f'\nOne input multiset X of size {n}:')

# Generate a multiset
X = torch.randn(size=(n,d), dtype=dtype, device=device)
print('Shape of X: ', X.shape)

X_emb = embed(X)
print('Shape of embed(X): ', X_emb.shape)

# Supports any number of batch dimensions:
batch_dims = (5,3,4)
batch_dim_str = "×".join(str(d) for d in batch_dims)
print(f'\nA batch Xb of {batch_dim_str} input multisets, each is of size {n}: ')

# Generate a batch of multisets
Xb = torch.randn(size=batch_dims+(n,d), dtype=dtype, device=device)
print('Shape of Xb: ', Xb.shape)

Xb_emb = embed(Xb)
print('Shape of embed(Xb): ', Xb_emb.shape)

# Output:

# Dimension of multiset elements: 15
# Embedding dimension: 123
#
# One input multiset X of size 50:
# Shape of X:  torch.Size([50, 15])
# Shape of embed(X):  torch.Size([123])
#
# A batch Xb of 5×3×4 input multisets, each is of size 50:
# Shape of Xb:  torch.Size([5, 3, 4, 50, 15])
# Shape of embed(Xb):  torch.Size([5, 3, 4, 123])

print('Device: ', embed.device, 'dtype: ', embed.dtype)
embed.to('cpu')
print('Device: ', embed.device, 'dtype: ', embed.dtype)
print('d_in: ', embed.d_in, 'd_out: ', embed.d_out)
