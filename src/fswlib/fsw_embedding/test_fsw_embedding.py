# Graph Separation
# This code generates the the plots of Figurs 2 and 3.

def main():
    import numpy as np

    import torch
    import torch.nn as nn

    from fswlib import FSWEmbedding

    import time

    def reldiff(x,y):
        return (torch.norm(x-y)/torch.norm(x)).item()

    def relerr(x,y):
        return (torch.norm(x-y)/torch.norm(x)).item()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float64

    # Note: This estimation is much slower than the rest of the code when n is large, and is not always accurate.
    #       I just used it as a sanity check.
    do_monte_carlo = False

    do_compile = False
    serialize_num_slices = None

    # d: ambient dimension
    # n: maximal multiset size
    # m: embedding dimension

    ## Interesting test settings:

    setting = 1

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
        #serialize_num_slices = 15
    elif setting == 5: # Could take 10 minutes
        d = 3
        n = 1000000
        m = 400



    # deterministic_freqs spreads the frequencies in a way that better approximates their distribution than random sampling
    # optimize_slices uses mutual coherence minimization to make the slices better spread
    # so far most settings both techniques did not yield dramatic improvement of performance
    embed = FSWEmbedding(d, m, dtype=dtype, device=device)
    embed2 = FSWEmbedding(d, m, dtype=dtype, device=device)

    # embed.to('cpu', torch.float32)
    # print('Embed type: ', embed.dtype, 'device: ', embed.device)
    # embed.to('cpu', dtype=torch.float64)
    # print('Embed type: ', embed.dtype, 'device: ', embed.device)
    # embed.to(device = 'cuda', dtype=torch.float32)
    # print('Embed type: ', embed.dtype, 'device: ', embed.device)
    # embed.to(torch.float64)
    # print('Embed type: ', embed.dtype, 'device: ', embed.device)


    if do_compile:
        embed = torch.compile(embed)
        print('compile 1')
        embed2 = torch.compile(embed2)
        print('compile 2')


    X1 = torch.randn(size=(n,d), dtype=dtype, device=device)
    X2 = 1.1 * torch.randn(size=(n,d), dtype=dtype, device=device)

    # Approx. 20% of the weights will be zero.
    W1 = torch.nn.functional.relu(torch.rand(size=(n,), dtype=dtype, device=device)-0.2)
    W2 = torch.nn.functional.relu(torch.rand(size=(n,), dtype=dtype, device=device)-0.2)

    # The weights do not have to be normalized, as they are normalized in the forward call.
    # I normalized them here for convenience.
    W1 = torch.nn.functional.normalize(W1, p=1.0, dim=-1, eps=0)
    W2 = torch.nn.functional.normalize(W2, p=1.0, dim=-1, eps=0)

    print()
    print('d=%d  n=%d  m=%d' % (d, n, m) )
    print()
    print('Device: %s' %(device))
    print('Calculating SW embeddings... ', end='', flush=True)
    print()

    t_start = time.time();

    X1_emb = embed(X1,W1, serialize_num_slices=serialize_num_slices); print('1', end='', flush=True)
    X2_emb = embed(X2,W2, serialize_num_slices=serialize_num_slices); print('2', end='', flush=True)

    X1_emb2 = embed2(X1,W1, serialize_num_slices=serialize_num_slices); print('3', end='', flush=True)
    X2_emb2 = embed2(X2,W2, serialize_num_slices=serialize_num_slices); print('4', end='', flush=True)

    print(flush=True)

    t_end = time.time();


    ## Monte-carlo estimation of the SW distance.
    if do_monte_carlo:
        import ot

        print('Calculating Monte-Carlo estimation...', flush=True)

        # Monte-carlo estimation of the SW distance.
        est1 = ot.sliced.sliced_wasserstein_distance(X1, X2, a=W1, b=W2, n_projections=50000, p=2, projections=None, seed=None, log=False)
        est2 = ( torch.norm(X1_emb-X2_emb) / np.sqrt(float(embed.d_out)) )

        rel_diff = reldiff(est1, est2)
        print()

        print('Sliced Wasserstein of (X1,W1), (X2,W2) estimates:  Embedding: %.5g\t Sampling: %.5g\t Rel. diff.: %g' % (est1, est2, rel_diff), flush=True)
    else:
        print('Skipping Monte-Carlo estimation')
        print()

    ## Comparison between two independent embeddings

    est1 = ( torch.norm(X1_emb-X2_emb) / np.sqrt(float(embed.d_out)) )
    est2 = ( torch.norm(X1_emb2-X2_emb2) / np.sqrt(float(embed.d_out)) )

    rel_diff = reldiff(est1, est2)

    print('Sliced Wasserstein of (X1,W1), (X2,W2) estimates:  Embedding 1: %.5g\t Embedding 2: %.5g\t Rel. diff.: %g' % (est1, est2, rel_diff))


    ## Comparison with ground truth

    SWass_est = torch.norm(X1_emb) / np.sqrt(float(embed.d_out))
    SWass_true = torch.norm(X1 * torch.sqrt(W1.reshape((n,1)))) / np.sqrt(d)
    rel_err = rel_diff = reldiff(SWass_est, SWass_true)

    print('Sliced Wasserstein of (X1,W1) to delta_0:          Embedding: %.5g\t Ground truth: %.5g\t Rel. error: %g' % (SWass_est, SWass_true, rel_err))

    print()
    print('Avg. embedding calculation time: %.2f s' % ((t_end-t_start)/4))

    print()


    embed = FSWEmbedding(d, d_out=15, collapse_freqs=False, device=device, dtype=dtype)
    print('Serialization err: ', relerr( embed(X1,W1), embed(X1,W1, serialize_num_slices = 1)  ))
    print('Serialization err: ', relerr( embed(X1,W1), embed(X1,W1, serialize_num_slices = 2)  ))
    print('Serialization err: ', relerr( embed(X1,W1), embed(X1,W1, serialize_num_slices = 3)  ))
    print('Serialization err: ', relerr( embed(X1,W1), embed(X1,W1, serialize_num_slices = 5)  ))
    print('Serialization err: ', relerr( embed(X1,W1), embed(X1,W1, serialize_num_slices = 14)  ))
    print('Serialization err: ', relerr( embed(X1,W1), embed(X1,W1, serialize_num_slices = 15)  ))
    print('Serialization err: ', relerr( embed(X1,W1), embed(X1,W1, serialize_num_slices = 16)  ))

    embed = FSWEmbedding(d, nSlices=15, nFreqs=37, collapse_freqs=False, device=device, dtype=dtype)
    print('Serialization err: ', relerr( embed(X1,W1), embed(X1,W1, serialize_num_slices = 1)  ))
    print('Serialization err: ', relerr( embed(X1,W1), embed(X1,W1, serialize_num_slices = 2)  ))
    print('Serialization err: ', relerr( embed(X1,W1), embed(X1,W1, serialize_num_slices = 3)  ))
    print('Serialization err: ', relerr( embed(X1,W1), embed(X1,W1, serialize_num_slices = 5)  ))
    print('Serialization err: ', relerr( embed(X1,W1), embed(X1,W1, serialize_num_slices = 14)  ))
    print('Serialization err: ', relerr( embed(X1,W1), embed(X1,W1, serialize_num_slices = 15)  ))
    print('Serialization err: ', relerr( embed(X1,W1), embed(X1,W1, serialize_num_slices = 16)  ))

    embed = FSWEmbedding(d, nSlices=15, nFreqs=37, collapse_freqs=True, device=device, dtype=dtype)
    print('Serialization err: ', relerr( embed(X1,W1), embed(X1,W1, serialize_num_slices = 1)  ))
    print('Serialization err: ', relerr( embed(X1,W1), embed(X1,W1, serialize_num_slices = 2)  ))
    print('Serialization err: ', relerr( embed(X1,W1), embed(X1,W1, serialize_num_slices = 3)  ))
    print('Serialization err: ', relerr( embed(X1,W1), embed(X1,W1, serialize_num_slices = 5)  ))
    print('Serialization err: ', relerr( embed(X1,W1), embed(X1,W1, serialize_num_slices = 14)  ))
    print('Serialization err: ', relerr( embed(X1,W1), embed(X1,W1, serialize_num_slices = 15)  ))
    print('Serialization err: ', relerr( embed(X1,W1), embed(X1,W1, serialize_num_slices = 16)  ))



    # x = torch.tensor([4.0, 2.0, 1.5, 0.5], requires_grad=True)
    # out = torch.sin(x) * torch.cos(x) + x.pow(2)
    # # Pass tensor of ones, each for each item in x
    # out.backward(torch.ones_like(x))
    # print('aaaaa', x.grad)


    # Test graph mode:

    n = 100
    d = 15

    if True:
        m=23
        nSlices=None
        nFreqs=None
    else:
        m=None
        nSlices=19
        nFreqs=17

    X = torch.randn(size=(2,3,n,d), dtype=dtype, device=device)
    Wx = torch.nn.functional.relu(torch.rand(size=(2,3,n-3,n), dtype=dtype, device=device)-0.2)

    #X.requires_grad = True
    #Wx.requires_grad = True

    embed = FSWEmbedding(d, d_out=m, nSlices=nSlices, nFreqs=nFreqs, device=device, dtype=dtype)
    #embed = FSWEmbedding(d, nSlices=15, nFreqs=7, collapse_freqs=False, device=device, dtype=dtype)

    Xx = X.unsqueeze(dim=-3).expand( tuple(Wx.shape) + (X.shape[-1],))

    emb1 = embed(Xx, Wx)

    emb1a = embed(Xx,Wx)
    emb1b = embed(Xx,Wx.to_sparse())
    emb1c = embed(Xx,Wx.to_sparse(), serialize_num_slices=1)
    emb1d = embed(Xx,Wx, serialize_num_slices=1)

    emb2a = embed(X,Wx.to_sparse(), graph_mode=True, serialize_num_slices=1)

    print('=========================================')
    emb2b = embed(X,Wx.to_sparse(), graph_mode=True)
    #emb2c = embed(X,Wx, graph_mode=True)

    print('Graph mode error 1a: ', relerr(emb1,emb1a))
    print('Graph mode error 1b: ', relerr(emb1,emb1b))
    print('Graph mode error 1c: ', relerr(emb1,emb1c))
    print('Graph mode error 1d: ', relerr(emb1,emb1d))
    print('Graph mode error 2a: ', relerr(emb1,emb2a))
    print('Graph mode error 2b: ', relerr(emb1,emb2b))
    #print('Graph mode error 2c: ', relerr(emb1,emb2c))

    if False:
        from swe_works import SW_embed as SWE
        embedw = SWE(d, m=d, device=device, dtype=dtype)

        sd = embed.state_dict()
        embedw.load_state_dict(sd) # your error

        emb1aw = embedw(Xx,Wx.to_sparse())
        emb1bw = embedw(Xx,Wx.to_sparse(), serialize_num_slices=7)
        emb1cw = embedw(Xx,Wx, serialize_num_slices=7)

        emb2aw = embedw(X,Wx.to_sparse(), graph_mode=True, serialize_num_slices=7)
        emb2bw = embedw(X,Wx.to_sparse(), graph_mode=True)
        emb2cw = embedw(X,Wx, graph_mode=True)

        print('\nGraph mode error 1a: ', relerr(emb1aw,emb1a))
        print('Graph mode error 1b: ', relerr(emb1bw,emb1b))
        print('Graph mode error 1c: ', relerr(emb1cw,emb1c))
        print('Graph mode error 2a: ', relerr(emb2aw,emb2a))
        print('Graph mode error 2b: ', relerr(emb2bw,emb2b))
        print('Graph mode error 2c: ', relerr(emb2cw,emb2c))

    # exit()

    print('=========================================')

    x = torch.randn(size=(5,), dtype=torch.float64, device=device, requires_grad=True)
    y = torch.sinc(x)
    dy = torch.autograd.grad(y, x, torch.ones_like(y), create_graph=True, retain_graph=True)[0]
    #y, dy = sw_embedding.sinc_dsinc(x)

    print('dsinc err: ', relerr( (torch.cos(np.pi*x) - torch.sinc(x)) / x, dy) )


    print('=========================================')

    # Testing gradient
    n = 100
    d = 64

    m = 100

    nSlices = None
    nFreq = None

    # m = None
    # nSlices = 20
    # nFreqs =  50

    graph_mode = False

    test_grad_X = True
    test_grad_W = True
    test_grad_slices = True
    test_grad_freqs = True
    test_grad_bias = True

    size = (n-1,n) if graph_mode else (n,)

    X1 = torch.randn(size=(n,d), dtype=dtype, device=device)
    W1 = torch.nn.functional.relu(torch.rand(size=size, dtype=dtype, device=device)-0.2)

    X2 = X1.clone()
    W2 = W1.clone().to_sparse()

    W2.values()[0] = 0
    W1[tuple(W2.indices()[:,0])] = 0

    X1.requires_grad = test_grad_X
    X2.requires_grad = test_grad_X

    W1.requires_grad = test_grad_W
    W2.requires_grad = test_grad_W

    # Embed 1
    embed = FSWEmbedding(d_in=d, d_out=m, nSlices=nSlices, nFreqs = nFreqs, device=device, dtype=dtype, learnable_slices=test_grad_slices, learnable_freqs=test_grad_freqs, minimize_slice_coherence=True, freqs_init='spread')

    emb1 = embed(X1, W1, graph_mode=graph_mode)
    S1 = emb1.norm()
    S1.backward()

    proj_grad1 = embed.projVecs.grad.clone() if test_grad_slices else None
    freqs_grad1 = embed.freqs.grad.clone() if test_grad_freqs else None
    bias_grad1 = embed.bias.grad.clone() if test_grad_bias else None

    # Embed 2
    embed.zero_grad()

    emb2 = embed(X2, W2, graph_mode=graph_mode)
    S2 = emb2.norm()
    S2.backward()

    proj_grad2 = embed.projVecs.grad.clone() if test_grad_slices else None
    freqs_grad2 = embed.freqs.grad.clone() if test_grad_freqs else None
    bias_grad2 = embed.bias.grad.clone() if test_grad_bias else None

    # Compare gradients

    print()
    print('Sparsity error:   ', relerr(emb1,emb2))
    print()

    if test_grad_X:
        print('Grad X error:     ', reldiff(X1.grad, X2.grad))

    if test_grad_W:
        print('Grad W error:     ', reldiff(   W2.grad.to_dense()[W2.grad.to_dense() != 0], W1.grad.to_dense()[W2.grad.to_dense() != 0]    ))

    if test_grad_slices:
        print('Grad projs error: ', reldiff(proj_grad1, proj_grad2))

    if test_grad_freqs:
        print('Grad freqs error: ', reldiff(freqs_grad1, freqs_grad2))

    if test_grad_bias:
        print('Grad bias error: ', reldiff(bias_grad1, bias_grad2))

    # W1 = FSWEmbedding.project_W(W1, 1e-8)
    # W2 = FSWEmbedding.project_W(W2, 1e-8)

    print()
    print('=========================================')


    from torch.autograd import gradcheck

    # gradcheck takes a tuple of tensors as input, check if your gradient
    # evaluated with these tensors are close enough to numerical
    # approximations and returns True if they all verify this condition.
    input = (X1, W1+1e-5, None, graph_mode)
    test = gradcheck(embed, input, eps=1e-6, atol=1e-6)
    print('Gradcheck test result: ', test)

    print('=========================================')


    state_dict = embed.state_dict()
    for x in state_dict:
        print('Name: ', x, ' Shape: ', state_dict[x].shape)

    print()


    exit()

    print('=========================================')



    # Stress-testing graph mode


    # print(state_dict)

    # print('projVecs shape: ', state_dict['projVecs'].shape)
    # state_dict['projVecs'][:] = 0

    bs = 20
    n = 3000
    deg = 20
    d = 64
    m = 64

    X = torch.randn(size=(bs,n,d), dtype=dtype, device=device)

    diags = torch.rand(size=(deg,n), dtype=dtype)
    offsets = torch.tensor(range(deg))
    pattern = torch.sparse.spdiags(diags, offsets, shape=(n,n)).to(device=device)

    pattern = pattern.unsqueeze(dim=0)
    pattern = torch.cat((pattern,)*bs, dim=0).coalesce()
    pattern.values()[:] = torch.rand(size=pattern.values().shape, dtype=dtype, device=device)

    W = pattern

    X.requires_grad = True
    W.requires_grad = True
    embed = FSWEmbedding(d=d, m=m, device=device, dtype=dtype, learnable_slices=True, learnable_freqs=True)

    t = time.time()
    emb = embed(X,W, graph_mode=True, serialize_num_slices=1)

    S = emb.norm()
    S.backward()
    tElapsed = time.time()-t

    print('Grad X shape: ', X.grad.shape)
    assert X.grad.isnan().any() == False


    print('\n%d graphs, each with %d nodes of degree %g. Vertex feature dimension: %d. Embedding dimension: %d' % (bs, n, deg, d, m))
    print('Time elapsed: %g\n' % (tElapsed))

    ########################################################


    ########################################################
    # n = 1000

    # W = torch.nn.functional.relu(torch.rand(size=(n,n), dtype=dtype)-0.2)
    # W = W.to_sparse()

    # S = sw_embedding.sparse_expand_as(W.sum(dim=-1).unsqueeze(-1), W)
    # print('S shape: ', S.shape)

    # print(W.to_dense())
    # print('W sum: ', W.sum(dim=-1).to_dense())
    # print(S.to_dense())

    # print('W nnz: ', len(W.values()) )
    # print('S nnz: ', len(S.values()) )

    # W2 = sw_embedding.sparse_mul_expand(W, W.sum(dim=-1).unsqueeze(-1))
    # print('W2 shape: ', W2.shape)
    # print('W2 error: ', torch.norm( W2.to_dense() - W.to_dense() * W.to_dense().sum(dim=-1,keepdim=True) ))

    # exit()

    # W_orig = W.to_dense()

    # W = W * ( W.sum(dim=-1).to_dense().unsqueeze(-1).reciprocal() )
    # # Here W is sparse and normalied

    # sums = -W.sum(dim=-1).unsqueeze(-1)
    # W_walled = torch.cat((W, sums), dim=-1).coalesce()

    # inds = W_walled.indices()
    # vals = W_walled.values().cumsum(dim=0)
    # subset = (inds[-1,:] < n)

    # W_sparse_cumsum = torch.sparse_coo_tensor(indices=inds[:,subset], values=vals[subset], size=W.shape)

    # B = nn.functional.normalize(W_orig,dim=-1,p=1.0).cumsum(dim=-1)
    # B[W.to_dense()==0] = 0

    # print('Numerical error: ', torch.norm(W_sparse_cumsum.to_dense()-B).item())

    ####################################################################




    X1 = torch.randn(size=(n,d), dtype=dtype, device=device)
    W1 = torch.nn.functional.relu(torch.rand(size=(n,), dtype=dtype, device=device)-0.2).to_sparse()

    if True:
        X1.requires_grad = True
        X2.requires_grad = True

        W1.requires_grad = True
        W2.requires_grad = True

    embed = FSWEmbedding(d=d, m=m, device=device, dtype=dtype, learnable_freqs=True, learnable_slices=True)
    E = embed(X1,W1)

# if __name__ == "__main__":
#     main()
