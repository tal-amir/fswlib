# Graph Separation
# This code generates the the plots of Figurs 2 and 3.

import sys
import os
import pathlib

# Determine the root of the project (one level up from this file)
project_root = pathlib.Path(__file__).resolve().parent.parent

# Path to the src directory
fsw_embedding_dir = project_root / 'src'

# Add to sys.path if not already there
if str(fsw_embedding_dir) not in sys.path:
    sys.path.insert(0, str(fsw_embedding_dir))


import numpy as np

import torch
import torch.nn as nn

from fsw_embedding import FSW_embedding

import time

def reldiff(x,y):
    return np.abs(x-y)/min(x,y)

dtype=torch.float64

# Note: This estimation is much slower than the rest of the code when n is large, and is not always accurate.
#       I just used it as a sanity check.
do_monte_carlo = False

do_compile = False
serialize_slices_chunk = None

# d: ambient dimension
# n: maximal multiset size
# m: embedding dimension

## Interesting test settings:

setting = 4

if setting == 1:
    # super fast
    d = 20
    n = 100
    m = 1000
    #do_monte_carlo = True
elif setting == 2:
    # takes ~40 s
    d = 100
    n = 50000
    m = 1000
elif setting == 3:
    # takes 20-30s to calculate four embeddings
    d = 3 
    n = 100000
    m = 300
elif setting == 4:
    # takes ~2.5 minutes. sometimes yields up to 10% error due to low embedding dimension
    d = 3
    n = 1000000
    m = 150
    #serialize_slices_chunk = 15
elif setting == 5: # Could take 10 minutes
    d = 3
    n = 1000000
    m = 400



# deterministic_freqs spreads the frequencies in a way that better approximates their distribution than random sampling
# optimize_slices uses mutual coherence minimization to make the slices better spread
# so far most settings both techniques did not yield dramatic improvement of performance
embed = SW_embedding(d, m, deterministic_freqs=False, optimize_slices = False)
embed2 = SW_embedding(d, m, deterministic_freqs=False, optimize_slices = False)

if do_compile:
    embed = torch.compile(embed)
    embed2 = torch.compile(embed2)


X1 = torch.randn(size=(n,d), dtype=dtype)
X2 = 1.1 * torch.randn(size=(n,d), dtype=dtype)

# Approx. 20% of the weights will be zero.
W1 = torch.nn.functional.relu(torch.rand(size=(n,), dtype=dtype)-0.2)
W2 = torch.nn.functional.relu(torch.rand(size=(n,), dtype=dtype)-0.2)

# The weights do not have to be normalized, as they are normalized in the forward call.
# I normalized them here for convenience.
W1 = torch.nn.functional.normalize(W1, p=1.0, dim=-1, eps=0)
W2 = torch.nn.functional.normalize(W2, p=1.0, dim=-1, eps=0)

print()
print('d=%d  n=%d  m=%d' % (d, n, m) )
print()
print('Calculating SW embeddings... ', end='', flush=True)
print()

t_start = time.time();

X1_emb = embed(X1,W1, serialize_slices_chunk=serialize_slices_chunk); print('1', end='', flush=True)
X2_emb = embed(X2,W2, serialize_slices_chunk=serialize_slices_chunk); print('2', end='', flush=True)

X1_emb2 = embed2(X1,W1, serialize_slices_chunk=serialize_slices_chunk); print('3', end='', flush=True)
X2_emb2 = embed2(X2,W2, serialize_slices_chunk=serialize_slices_chunk); print('4', end='', flush=True)

print(flush=True)

t_end = time.time();


## Monte-carlo estimation of the SW distance.     
if do_monte_carlo:
    import ot

    print('Calculating Monte-Carlo estimation...', flush=True)

    # Monte-carlo estimation of the SW distance.     
    est1 = ot.sliced.sliced_wasserstein_distance(X1, X2, a=W1, b=W2, n_projections=50000, p=2, projections=None, seed=None, log=False)
    est2 = ( torch.norm(X1_emb-X2_emb) / np.sqrt(float(embed.m)) )

    rel_diff = reldiff(est1, est2)
    print()

    print('Sliced Wasserstein of (X1,W1), (X2,W2) estimates:  Embedding: %.5g\t Sampling: %.5g\t Rel. diff.: %g' % (est1, est2, rel_diff), flush=True)
else:
    print('Skipping Monte-Carlo estimation')
    print()

## Comparison between two independent embeddings

est1 = ( torch.norm(X1_emb-X2_emb) / np.sqrt(float(embed.m)) )
est2 = ( torch.norm(X1_emb2-X2_emb2) / np.sqrt(float(embed.m)) )

rel_diff = reldiff(est1, est2)

print('Sliced Wasserstein of (X1,W1), (X2,W2) estimates:  Embedding 1: %.5g\t Embedding 2: %.5g\t Rel. diff.: %g' % (est1, est2, rel_diff))


## Comparison with ground truth

SWass_est = torch.norm(X1_emb) / np.sqrt(float(embed.m))
SWass_true = torch.norm(X1 * torch.sqrt(W1.reshape((n,1)))) / np.sqrt(d)
rel_err = rel_diff = reldiff(SWass_est, SWass_true)

print('Sliced Wasserstein of (X1,W1) to delta_0:          Embedding: %.5g\t Ground truth: %.5g\t Rel. error: %g' % (SWass_est, SWass_true, rel_err))

print()
print('Avg. embedding calculation time: %.2f s' % ((t_end-t_start)/4))

print()
