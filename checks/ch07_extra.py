"""Additional computed toys for chapter 7 (search structures, KV cache, end-to-end).

Imported by ch07_examples.py, which checks every number printed here.
Everything is seeded so the output is reproducible on one machine.
"""
import math
import numpy as np


def corpus(seed=0, n=4000, d=32, nclus=20, nq=200):
    """Clustered Gaussian corpus with a decaying spectrum, plus queries from the same law."""
    rng = np.random.default_rng(seed)
    spec = 1.0 / np.sqrt(np.arange(1, d + 1))           # per-axis scale, decaying
    centres = rng.standard_normal((nclus, d)) * 3.0 * spec
    lab = rng.integers(0, nclus, n + nq)
    X = centres[lab] + rng.standard_normal((n + nq, d)) * spec
    return X[:n].astype(np.float64), X[n:].astype(np.float64)


def exact_topk(X, Q, k):
    S = Q @ X.T
    return np.argsort(-S, axis=1)[:, :k], S


def recall(ids, true):
    k = true.shape[1]
    return float(np.mean([len(set(a) & set(b)) / k for a, b in zip(ids, true)]))


def kmeans(X, K, iters, rng):
    C = X[rng.choice(len(X), K, replace=False)].copy()
    for _ in range(iters):
        d2 = (X ** 2).sum(1)[:, None] - 2 * X @ C.T + (C ** 2).sum(1)[None]
        a = np.argmin(d2, 1)
        for j in range(K):
            m = a == j
            if m.any():
                C[j] = X[m].mean(0)
    d2 = (X ** 2).sum(1)[:, None] - 2 * X @ C.T + (C ** 2).sum(1)[None]
    return C, np.argmin(d2, 1)


def haar(d, rng):
    G = rng.standard_normal((d, d))
    Qm, R = np.linalg.qr(G)
    return Qm @ np.diag(np.sign(np.diag(R)))


def ivf_curve(X, Q, k=10, nlist=32, seed=1):
    rng = np.random.default_rng(seed)
    C, a = kmeans(X, nlist, 25, rng)
    true, _ = exact_topk(X, Q, k)
    out = []
    order = np.argsort(-(Q @ C.T), axis=1)          # cells by centroid score
    for nprobe in [1, 2, 4, 8, 16, 32]:
        ids, frac = [], []
        for qi, q in enumerate(Q):
            cells = order[qi, :nprobe]
            cand = np.where(np.isin(a, cells))[0]
            s = X[cand] @ q
            ids.append(cand[np.argsort(-s)[:k]])
            frac.append(len(cand) / len(X))
        out.append((nprobe, float(np.mean(frac)), recall(ids, true)))
    return out


def scalar_codes(X, Q, bits, rng):
    """Random Haar rotation, then a uniform b-bit quantizer per coordinate, range +-3 sigma."""
    d = X.shape[1]
    R = haar(d, rng)
    Y = X @ R.T
    sig = Y.std()
    A = 3 * sig
    L = 2 ** bits
    step = 2 * A / L
    idx = np.clip(np.floor((Y + A) / step), 0, L - 1)
    Yh = -A + (idx + 0.5) * step
    return Yh @ R                                   # back to original coordinates


def pq_codes(X, m, K, rng):
    n, d = X.shape
    s = d // m
    Xh = np.empty_like(X)
    for j in range(m):
        sub = X[:, j * s:(j + 1) * s]
        C, a = kmeans(sub, K, 20, rng)
        Xh[:, j * s:(j + 1) * s] = C[a]
    return Xh


def end_to_end(X, Q, k=10, seed=2):
    rng = np.random.default_rng(seed)
    true, S = exact_topk(X, Q, k)
    d = X.shape[1]
    rows = []
    for bits in [1, 2, 4]:
        Xh = scalar_codes(X, Q, bits, rng)
        nbytes = d * bits / 8
        ids = np.argsort(-(Q @ Xh.T), axis=1)[:, :k]
        # rerank: take 5k ADC candidates, rescore exactly
        cand = np.argsort(-(Q @ Xh.T), axis=1)[:, :5 * k]
        rr = np.array([c[np.argsort(-(X[c] @ q))[:k]] for c, q in zip(cand, Q)])
        rows.append(("scalar", nbytes, recall(ids, true), recall(rr, true)))
    for m in [4, 8, 16]:
        Xh = pq_codes(X, m, 256, rng)
        ids = np.argsort(-(Q @ Xh.T), axis=1)[:, :k]
        cand = np.argsort(-(Q @ Xh.T), axis=1)[:, :5 * k]
        rr = np.array([c[np.argsort(-(X[c] @ q))[:k]] for c, q in zip(cand, Q)])
        rows.append(("pq", float(m), recall(ids, true), recall(rr, true)))
    return rows


def adc_vs_sdc(X, Q, rng, m=8, K=16):
    """Mean squared score error of ADC <q, xh> and SDC <qh, xh> against <q, x>."""
    n, d = X.shape
    s = d // m
    Xh = np.empty_like(X)
    Qh = np.empty_like(Q)
    for j in range(m):
        C, a = kmeans(X[:, j * s:(j + 1) * s], K, 20, rng)
        Xh[:, j * s:(j + 1) * s] = C[a]
        sub = Q[:, j * s:(j + 1) * s]
        d2 = (sub ** 2).sum(1)[:, None] - 2 * sub @ C.T + (C ** 2).sum(1)[None]
        Qh[:, j * s:(j + 1) * s] = C[np.argmin(d2, 1)]
    S = Q @ X.T
    adc = np.mean((Q @ Xh.T - S) ** 2)
    sdc = np.mean((Qh @ Xh.T - S) ** 2)
    return adc, sdc, float(np.mean(S ** 2))


def kv_toy(seed=4, T=64, d=16, outlier=8.0, bits=4):
    """Keys with one outlier channel. Per-token vs per-channel asymmetric uniform quantization."""
    rng = np.random.default_rng(seed)
    K = rng.standard_normal((T, d))
    K[:, 0] = K[:, 0] * 0.5 + outlier             # channel 0: large DC offset
    q = rng.standard_normal(d)
    q[0] = 1.0

    def quant(M, axis):
        lo = M.min(axis=axis, keepdims=True)
        hi = M.max(axis=axis, keepdims=True)
        L = 2 ** bits - 1
        sc = (hi - lo) / L
        return lo + np.round((M - lo) / sc) * sc

    Kt = quant(K, 1)                                # per token: one range per key vector
    Kc = quant(K, 0)                                # per channel: one range per channel over tokens
    mse_t = np.mean((Kt - K) ** 2)
    mse_c = np.mean((Kc - K) ** 2)
    mse_t_small = np.mean((Kt - K)[:, 1:] ** 2)
    mse_c_small = np.mean((Kc - K)[:, 1:] ** 2)

    def attn(Km):
        l = Km @ q / math.sqrt(d)
        p = np.exp(l - l.max())
        return p / p.sum()

    p0 = attn(K)
    kl = lambda p: float(np.sum(p0 * np.log(p0 / p)))
    return dict(mse_t=mse_t, mse_c=mse_c, mse_t_small=mse_t_small, mse_c_small=mse_c_small,
                kl_t=kl(attn(Kt)), kl_c=kl(attn(Kc)))


def value_avg_toy(seed=5, T=64, d=16, sigma=0.1):
    """Value error is averaged by attention: E||sum p_s e_s||^2 = sigma^2 d sum p_s^2."""
    rng = np.random.default_rng(seed)
    l = rng.standard_normal(T)
    p = np.exp(l) / np.exp(l).sum()
    E = rng.standard_normal((20000, T, d)) * sigma
    o = np.einsum("s,nsd->nd", p, E)
    return float(np.mean((o ** 2).sum(1))), sigma ** 2 * d * float(np.sum(p ** 2)), float(np.sum(p ** 2)), d * sigma ** 2
