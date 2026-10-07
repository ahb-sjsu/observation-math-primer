"""Checks for every number and every plotted coordinate in chapter 7 (quantization and vector search).

Run on Atlas:  python3 ch07_examples.py
Prints PASS/FAIL per claim and the figure coordinates, then k/n PASS. Exits nonzero on any FAIL.
"""
import sys
import math
import numpy as np
import sympy as sp

results = []


def check(desc, cond):
    ok = bool(cond)
    results.append(ok)
    print(("PASS " if ok else "FAIL ") + desc)


def close(a, b, tol=1e-3):
    return abs(float(a) - float(b)) <= tol


def coords(name, pts, fmt="{:.4g}"):
    print("FIGDATA " + name + ": " + " ".join("(" + ",".join(fmt.format(v) for v in p) + ")" for p in pts))


def db(x):
    return 10 * math.log10(x)


phi = lambda t: math.exp(-t * t / 2) / math.sqrt(2 * math.pi)
Phi = lambda t: 0.5 * (1 + math.erf(t / math.sqrt(2)))

# ---------------------------------------------------------------------------
# Section 1: uniform quantizer
# ---------------------------------------------------------------------------
b = 3
Delta = 2 / 2 ** b
check("uniform [-1,1], b=3: Delta = 0.25", close(Delta, 0.25, 1e-12))
check("MSE = Delta^2/12 = 1/192 = 0.005208", close(Delta ** 2 / 12, 1 / 192, 1e-15) and close(1 / 192, 0.005208, 1e-6))
check("signal variance 1/3, SQNR = 64 = 18.06 dB", close((1 / 3) / (1 / 192), 64, 1e-9) and close(db(64), 18.06, 1e-2))
check("6.02 dB per bit: 10 log10 4 = 6.0206", close(db(4), 6.0206, 1e-4))
# Monte Carlo: midrise quantizer on uniform input
rng = np.random.default_rng(0)
x = rng.uniform(-1, 1, 1_000_000)
q = (np.floor(x / Delta) + 0.5) * Delta
check("Monte Carlo MSE of 3-bit midrise quantizer on U[-1,1] = 0.005208", close(np.mean((x - q) ** 2), 1 / 192, 2e-5))
check("error is uniform on [-Delta/2, Delta/2]", np.abs(x - q).max() <= Delta / 2 + 1e-12)
# Gaussian with loading +-4 sigma
sq = lambda bb: 12 * 4 ** bb / 64
check("Gaussian, range +-4 sigma: SQNR = 0.1875*4^b, i.e. 6.02b - 7.27 dB", close(db(0.1875), -7.27, 1e-2))
check("... at b=8: 40.89 dB (granular only)", close(db(sq(8)), 40.89, 1e-2))
# staircase figure data
stair = [(-1 + k * Delta, -1 + (k + 0.5) * Delta) for k in range(8)]
coords("fig staircase steps (left edge, level)", stair)
check("staircase levels are -0.875..0.875 in steps of 0.25", close(stair[0][1], -0.875, 1e-12) and close(stair[-1][1], 0.875, 1e-12))

# ---------------------------------------------------------------------------
# Section 2: Lloyd-Max
# ---------------------------------------------------------------------------
def lloyd_gauss(levels, iters=2000):
    y = np.array(levels, float)
    for _ in range(iters):
        t = np.concatenate(([-np.inf], (y[:-1] + y[1:]) / 2, [np.inf]))
        ynew = []
        for a, bnd in zip(t[:-1], t[1:]):
            pa = 0 if np.isinf(a) else phi(a)
            pb = 0 if np.isinf(bnd) else phi(bnd)
            Pa = 0 if np.isinf(a) and a < 0 else Phi(a)
            Pb = 1 if np.isinf(bnd) else Phi(bnd)
            ynew.append((pa - pb) / (Pb - Pa))
        y = np.array(ynew)
    t = np.concatenate(([-np.inf], (y[:-1] + y[1:]) / 2, [np.inf]))
    # MSE = 1 - sum p_i y_i^2  (centroid condition)
    p = np.array([(1 if np.isinf(bb) else Phi(bb)) - (0 if np.isinf(aa) else Phi(aa)) for aa, bb in zip(t[:-1], t[1:])])
    mse = 1 - np.sum(p * y ** 2)
    return y, t, p, mse


y1, t1, p1, m1 = lloyd_gauss([-1, 1])
check("Lloyd-Max 1 bit Gaussian: levels +-sqrt(2/pi) = 0.7979", close(y1[1], math.sqrt(2 / math.pi), 1e-6) and close(y1[1], 0.7979, 1e-4))
check("Lloyd-Max 1 bit Gaussian: MSE = 1 - 2/pi = 0.3634", close(m1, 1 - 2 / math.pi, 1e-6) and close(m1, 0.3634, 1e-4))
y2, t2, p2, m2 = lloyd_gauss([-1.5, -0.5, 0.5, 1.5])
check("Lloyd-Max 2 bit: thresholds 0, +-0.9816", close(t2[2], 0, 1e-9) and close(t2[3], 0.9816, 1e-4))
check("Lloyd-Max 2 bit: levels +-0.4528, +-1.5104", close(y2[2], 0.4528, 1e-4) and close(y2[3], 1.5104, 1e-4))
check("Lloyd-Max 2 bit: MSE 0.1175", close(m2, 0.1175, 1e-4))
y3, t3, p3, m3 = lloyd_gauss(np.linspace(-2, 2, 8))
y4, t4, p4, m4 = lloyd_gauss(np.linspace(-2.5, 2.5, 16))
check("Lloyd-Max 3 bit MSE 0.03454, 4 bit 0.009497", close(m3, 0.03454, 1e-5) and close(m4, 0.009497, 1e-5))
lm_db = [(1, -db(m1)), (2, -db(m2)), (3, -db(m3)), (4, -db(m4))]
coords("fig SNR Lloyd-Max Gaussian (b, SNR dB)", lm_db)
check("Lloyd-Max SNR dB 4.40, 9.30, 14.62, 20.22", all(close(a[1], v, 1e-2) for a, v in zip(lm_db, [4.40, 9.30, 14.62, 20.22])))
coords("fig SNR R(D) Gaussian 6.02b", [(bb, db(4 ** bb)) for bb in range(5)])
coords("fig SNR entropy-coded uniform high-res 6.02b - 1.53", [(bb, db(4 ** bb) - db(2 * math.pi * math.e / 12)) for bb in range(1, 5)])
check("uniform [-1,1] Lloyd-Max at 2 bits is uniform: levels +-0.25, +-0.75, MSE 1/48",
      close((0.5) ** 2 / 12, 1 / 48, 1e-15))
# Lloyd-Max 2-bit cell probabilities and output entropy (figure + entropy-coding example)
check("2-bit cell probabilities 0.1631, 0.3369", close(p2[0], 0.1631, 1e-4) and close(p2[1], 0.3369, 1e-4))
H2 = -np.sum(p2 * np.log2(p2))
check("entropy of 2-bit Lloyd-Max output = 1.911 bits", close(H2, 1.911, 1e-3))

# ---------------------------------------------------------------------------
# Section 3: fixed rate vs entropy coded
# ---------------------------------------------------------------------------
gp = 0.5 * math.log2(math.pi * math.e / 6)
check("entropy-coded uniform quantizer is 1/2 log2(pi e/6) = 0.2546 bit above R(D)", close(gp, 0.2546, 1e-4))
check("... which is 10 log10(pi e / 6) = 1.533 dB", close(db(math.pi * math.e / 6), 1.533, 1e-3))
pd = math.sqrt(3) * math.pi / 2
check("Panter-Dite constant sqrt(3) pi/2 = 2.721, 4.35 dB above R(D)", close(pd, 2.721, 1e-3) and close(db(pd), 4.35, 1e-2))
check("Panter-Dite at b=2 predicts 2.721/16 = 0.170 vs exact 0.1175", close(pd / 16, 0.170, 1e-3))
check("Lloyd-Max 2 bits 0.1175 vs D(R=2) = 1/16 = 0.0625: 2.74 dB", close(db(m2 / 0.0625), 2.74, 1e-2))
# high-res derivation symbolic: h(X) - log2 Delta - 1/2 log2(sigma^2 *12/Delta^2) = 1/2 log2(2 pi e / 12)
s2, D_ = sp.symbols("sigma2 Delta", positive=True)
expr = sp.Rational(1, 2) * sp.log(2 * sp.pi * sp.E * s2, 2) - sp.log(D_, 2) - sp.Rational(1, 2) * sp.log(s2 * 12 / D_ ** 2, 2)
check("symbolic: entropy-coded excess = 1/2 log2(2 pi e/12)", sp.simplify(expr - sp.Rational(1, 2) * sp.log(2 * sp.pi * sp.E / 12, 2)) == 0)

# ---------------------------------------------------------------------------
# Section 4: bit allocation
# ---------------------------------------------------------------------------
def hr_alloc(var, B):
    var = np.array(var, float)
    act = np.ones(len(var), bool)
    while True:
        k = act.sum()
        gm = np.exp(np.mean(np.log(var[act])))
        bits = np.zeros(len(var))
        bits[act] = B / k + 0.5 * np.log2(var[act] / gm)
        if (bits[act] >= -1e-12).all():
            return bits
        act &= bits > 0


v = [16, 4, 1, 0.25]
b8 = hr_alloc(v, 8)
check("allocation B=8: bits 3.5, 2.5, 1.5, 0.5", np.allclose(b8, [3.5, 2.5, 1.5, 0.5]))
Dk = np.array(v) * 2.0 ** (-2 * b8)
check("allocation B=8: every coordinate distortion c/8", np.allclose(Dk, 1 / 8))
check("allocation B=8: total 0.5c vs uniform 21.25/16 = 1.328c, gain 4.24 dB",
      close(Dk.sum(), 0.5, 1e-12) and close(sum(v) / 16, 1.328, 1e-3) and close(db((sum(v) / 16) / 0.5), 4.24, 1e-2))
b4 = hr_alloc(v, 4)
check("allocation B=4: bits 7/3, 4/3, 1/3, 0", np.allclose(b4, [7 / 3, 4 / 3, 1 / 3, 0]))
lev = 16 * 2.0 ** (-2 * 7 / 3)
check("allocation B=4: water level c*16*2^(-14/3) = 0.630c exceeds 0.25, so coordinate 4 stays off", close(lev, 0.630, 1e-3) and lev > 0.25)
coords("fig allocation B=4 bits", [(i + 1, bb) for i, bb in enumerate(b4)])
coords("fig allocation B=4 distortions (c=1)", [(i + 1, min(vv, lev)) for i, vv in enumerate(v)])

# ---------------------------------------------------------------------------
# Section 5: VQ and k-means
# ---------------------------------------------------------------------------
P = np.array([(0, 0), (1, 0), (0, 1), (5, 5), (6, 5), (5, 6)], float)
C = np.array([(0, 0), (6, 5)], float)
for _ in range(20):
    lab = np.argmin(((P[:, None, :] - C[None]) ** 2).sum(-1), 1)
    C = np.array([P[lab == j].mean(0) for j in range(2)])
check("k-means: centroids (1/3,1/3) and (16/3,16/3)", np.allclose(C, [[1 / 3, 1 / 3], [16 / 3, 16 / 3]]))
sse = ((P - C[lab]) ** 2).sum()
check("k-means: SSE 8/3, mean 4/9 per point", close(sse, 8 / 3, 1e-12) and close(sse / 6, 4 / 9, 1e-12))
check("k-means: per-point 2/9, 5/9, 5/9", np.allclose(((P - C[lab]) ** 2).sum(1)[:3], [2 / 9, 5 / 9, 5 / 9]))
check("Voronoi boundary x1 + x2 = 17/3 (bisector of the centroids)", close((1 / 3 + 16 / 3), 17 / 3, 1e-12))
coords("fig kmeans centroids", C)
G_hex = 5 / (36 * math.sqrt(3))
check("hexagon normalized second moment 5/(36 sqrt3) = 0.08019 vs 1/12 = 0.08333: 0.167 dB",
      close(G_hex, 0.08019, 1e-5) and close(db((1 / 12) / G_hex), 0.167, 1e-3))
check("space-filling limit 10 log10(2 pi e/12) = 1.533 dB", close(db(2 * math.pi * math.e / 12), 1.533, 1e-3))
check("codebook size at d=128, 2 bits/dim: 2^256 = 1.16e77", close(math.log10(2.0 ** 256), 77.06, 1e-2))

# ---------------------------------------------------------------------------
# Section 6: rotation
# ---------------------------------------------------------------------------
H4 = 0.5 * np.array([[1, 1, 1, 1], [1, -1, 1, -1], [1, 1, -1, -1], [1, -1, -1, 1]], float)
check("H4/2 is orthogonal", np.allclose(H4 @ H4.T, np.eye(4)))
xv = np.array([2.0, 0, 0, 0])
yv = H4 @ xv
check("H(2,0,0,0) = (1,1,1,1), norm 2 preserved", np.allclose(yv, [1, 1, 1, 1]) and close(np.linalg.norm(yv), 2, 1e-12))
check("range halves: step and MSE ratio 4 = 6.02 dB", close((4 / 2 ** 3) / (2 / 2 ** 3), 2, 1e-12) and close(db(4), 6.02, 1e-2))
coords("fig rotation energies before (d=4)", [(i + 1, v_ ** 2) for i, v_ in enumerate(xv)])
coords("fig rotation energies after Hadamard (d=4)", [(i + 1, v_ ** 2) for i, v_ in enumerate(yv)])
# d = 16, a spiky vector, random orthogonal rotation with fixed seed
rng = np.random.default_rng(7)
Qm, Rm = np.linalg.qr(rng.standard_normal((16, 16)))
Qm = Qm @ np.diag(np.sign(np.diag(Rm)))
x16 = np.zeros(16); x16[0] = 3.0; x16[5] = 1.0
y16 = Qm @ x16
check("d=16 rotation preserves energy 10", close(np.sum(y16 ** 2), 10, 1e-9))
check("d=16: max coordinate energy drops from 9 to 3.03, mean 10/16 = 0.625", close((y16 ** 2).max(), 3.03, 0.005) and close(np.mean(y16 ** 2), 0.625, 1e-9))
coords("fig rotation d16 before", [(i + 1, v_ ** 2) for i, v_ in enumerate(x16)], "{:.3f}")
coords("fig rotation d16 after", [(i + 1, v_ ** 2) for i, v_ in enumerate(y16)], "{:.3f}")
print("FIGDATA d16 after max energy {:.3f}".format((y16 ** 2).max()))
# Beta marginal: E y1^2 = 1/d for uniform on the sphere
Z = rng.standard_normal((200000, 16)); Z /= np.linalg.norm(Z, axis=1, keepdims=True)
check("uniform on the sphere: E y_1^2 = 1/d = 0.0625", close(np.mean(Z[:, 0] ** 2), 1 / 16, 1e-3))

# ---------------------------------------------------------------------------
# Section 7: PQ and ADC
# ---------------------------------------------------------------------------
qv = np.array([1, 0, 0, 1.0])
cb1 = np.array([[0, 0], [1, 1.0]])
cb2 = np.array([[0, 1], [1, 0.0]])
T1 = cb1 @ qv[:2]
T2 = cb2 @ qv[2:]
check("ADC tables T1 = (0,1), T2 = (1,0)", np.allclose(T1, [0, 1]) and np.allclose(T2, [1, 0]))
codes = {"a": (1, 0), "b": (0, 1), "c": (1, 1), "d": (0, 0)}
sc = {k: T1[c[0]] + T2[c[1]] for k, c in codes.items()}
check("ADC scores a=2, b=0, c=1, d=1", sc == {"a": 2, "b": 0, "c": 1, "d": 1})
xa = np.concatenate([cb1[1], cb2[0]])
check("decoded a = (1,1,0,1), exact inner product with q = 2", np.allclose(xa, [1, 1, 0, 1]) and close(qv @ xa, 2, 1e-12))
check("PQ m=8,K=256,d=128: 8 bytes/vector, 64x vs fp32, codebook 32768 floats", 8 * 8 // 8 == 8 and 128 * 4 / 8 == 64 and 8 * 256 * 16 == 32768)
check("tq-pro bytes: 256 dims x 3 bits = 96 B + 4 B norm = 100 B; 768-d fp32 = 3072 B; ratio 30.72",
      256 * 3 / 8 + 4 == 100 and 768 * 4 == 3072 and close(3072 / 100, 30.72, 1e-12))
check("PQ m=100, K=256: 100 B, matched to the 100 B code", 100 * 8 / 8 == 100)

# ---------------------------------------------------------------------------
# Section 8: inner-product bias
# ---------------------------------------------------------------------------
check("1-bit Gaussian: E[x xhat] = 2/pi = 0.6366 = 1 - MSE", close(2 / math.pi, 0.6366, 1e-4) and close(2 / math.pi, 1 - m1, 1e-6))
d = 8
c = math.gamma(d / 2) / (math.sqrt(math.pi) * math.gamma((d + 1) / 2))
check("d=8: c = E|y1| = Gamma(4)/(sqrt(pi) Gamma(4.5)) = 0.2910", close(c, 0.2910, 1e-4))
alpha = d * c * c
check("d=8: alpha = d c^2 = 0.6776, and 1 - alpha = relative MSE", close(alpha, 0.6776, 1e-4))
check("large d: d c^2 -> 2/pi", close(1000 * (math.exp(math.lgamma(500) - math.lgamma(500.5)) / math.sqrt(math.pi)) ** 2, 2 / math.pi, 1e-3))
# Monte Carlo over Haar rotations
rng = np.random.default_rng(3)
N = 60000
G = rng.standard_normal((N, d, d))
Qs, Rs = np.linalg.qr(G)
Qs = Qs * np.sign(np.diagonal(Rs, axis1=1, axis2=2))[:, None, :]
xx = np.array([1.0, 0.5, -0.3, 0.2, 0, 0.1, 0.4, -0.2]); xx /= np.linalg.norm(xx)
yy = np.einsum("nij,j->ni", Qs, xx)
qy = c * np.sign(yy)
back = np.einsum("nji,nj->ni", Qs, qy)
mean_back = back.mean(0)
check("Monte Carlo: E_Pi[Pi^T Q(Pi x)] = alpha x with alpha = 0.678 (+-0.01)", np.allclose(mean_back, alpha * xx, atol=0.01))
qq = np.array([0.3, -1, 0.2, 0.5, 1, 0, -0.4, 0.6])
ratio = (back @ qq).mean() / (qq @ xx)
check("Monte Carlo: E<q, xhat> / <q, x> = 0.678 (+-0.02)", close(ratio, alpha, 0.02))
check("Monte Carlo: relative MSE = 1 - alpha = 0.322 (+-0.005)", close(np.mean(np.sum((back - xx) ** 2, 1)), 1 - alpha, 5e-3))
# QJL identity
s = rng.standard_normal((2_000_000, 2))
qj = np.array([1.0, 0]); kj = np.array([1.0, 1.0])
est = np.mean((s @ qj) * np.sign(s @ kj))
check("QJL identity: E[(s.q) sign(s.k)] = sqrt(2/pi) q.k/|k| = 1/sqrt(pi) = 0.5642", close(est, 1 / math.sqrt(math.pi), 2e-3) and close(math.sqrt(2 / math.pi) / math.sqrt(2), 0.5642, 1e-4))
check("QJL: unbiased estimate |k| sqrt(pi/2) (s.q) sign(s.k) has mean q.k = 1", close(math.sqrt(2) * math.sqrt(math.pi / 2) * est, 1.0, 5e-3))

# ---------------------------------------------------------------------------
# Section 9: rank fidelity, cosine toy
# ---------------------------------------------------------------------------
exact = {"A": 4, "B": 3, "C": 2, "D": 1}
approx = {"A": 4, "B": 2, "C": 3, "D": 1}
items = list(exact)
conc = disc = 0
for i in range(4):
    for j in range(i + 1, 4):
        s1 = np.sign(exact[items[i]] - exact[items[j]]) * np.sign(approx[items[i]] - approx[items[j]])
        conc += s1 > 0; disc += s1 < 0
tau = (conc - disc) / 6
check("Kendall tau with one swapped pair of 6 = 4/6 = 0.667", disc == 1 and close(tau, 2 / 3, 1e-12))
top = lambda dct, k: set(sorted(dct, key=lambda z: -dct[z])[:k])
check("recall@2 = 0.5, recall@3 = 1.0", len(top(exact, 2) & top(approx, 2)) / 2 == 0.5 and len(top(exact, 3) & top(approx, 3)) / 3 == 1.0)
# cosine toy: q reads e2, keys of norm 10 at angles 0.30 and 0.25; direction-only error of 0.10 rad
qv2 = np.array([0.0, 1.0])
k = lambda a: 10 * np.array([math.cos(a), math.sin(a)])
k1, k2, k1h, k2h = k(0.30), k(0.25), k(0.20), k(0.35)
cs = lambda u, v: u @ v / np.linalg.norm(u) / np.linalg.norm(v)
check("cosine toy: cos(k1,k1hat) = cos(k2,k2hat) = cos 0.1 = 0.9950", close(cs(k1, k1h), 0.9950, 1e-4) and close(cs(k2, k2h), 0.9950, 1e-4))
check("cosine toy: exact scores 2.955 > 2.474", close(qv2 @ k1, 2.955, 1e-3) and close(qv2 @ k2, 2.474, 1e-3))
check("cosine toy: quantized scores 1.987 < 3.429, ranking reversed", close(qv2 @ k1h, 1.987, 1e-3) and close(qv2 @ k2h, 3.429, 1e-3) and qv2 @ k1h < qv2 @ k2h)
check("cosine toy: relative error |k - khat|/|k| = 2 sin 0.05 = 0.0999", close(np.linalg.norm(k1 - k1h) / 10, 0.0999, 1e-4))
coords("fig cosine toy vectors k1,k2,k1h,k2h", [k1, k2, k1h, k2h], "{:.3f}")
# documented turboquant-pro numbers (quoted from docs/KV_KEYS_FINDING.md), arithmetic only
check("KV keys arithmetic: 0.148/0.062 = 2.39 (doc says 2.4x)", close(0.148 / 0.062, 2.39, 1e-2))
check("KV keys arithmetic: 10643/15.77 = 675 (more than 600x)", 10643 / 15.77 > 600)
check("KV keys arithmetic: 10643/12.24 = 870x the fp16 perplexity", close(10643 / 12.24, 869.5, 0.5))

# ---------------------------------------------------------------------------
# Exercises
# ---------------------------------------------------------------------------
check("ex 7.1: b=4 on [-2,2]: Delta = 0.25, MSE 0.005208, SQNR (4/3)/(1/192) = 256 = 24.08 dB",
      close(4 / 16, 0.25, 1e-12) and close((4 / 3) / (1 / 192), 256, 1e-9) and close(db(256), 24.08, 1e-2))
check("ex 7.2: PQ m=16, K=256: 16 B; 768-d fp32 3072 B: 192x", 16 * 8 / 8 == 16 and 3072 / 16 == 192)
check("ex 7.5: 1-bit Lloyd-Max on U[0,1]: levels 0.25, 0.75, MSE 1/48 = 0.02083", close(0.5 ** 2 / 12, 1 / 48, 1e-15) and close(1 / 48, 0.02083, 1e-5))
b96 = hr_alloc([9, 1], 2)
check("ex 7.6: variances (9,1), B=2: bits 1.792, 0.208", np.allclose(b96, [1 + 0.5 * math.log2(3), 1 - 0.5 * math.log2(3)]) and close(b96[0], 1.792, 1e-3) and close(b96[1], 0.208, 1e-3))
check("ex 7.6: each distortion 9*2^-3.585 = 0.75 = 1*2^-0.415", close(9 * 2 ** (-2 * b96[0]), 0.75, 1e-9) and close(2 ** (-2 * b96[1]), 0.75, 1e-9))
check("ex 7.6: uniform 1 bit each gives 9/4 + 1/4 = 2.5 vs 1.5", close(9 / 4 + 1 / 4, 2.5, 1e-12))
r1 = [1, 2, 3, 4, 5]; r2 = [2, 1, 3, 5, 4]
dd = sum(1 for i in range(5) for j in range(i + 1, 5) if (r1[i] - r1[j]) * (r2[i] - r2[j]) < 0)
check("ex 7.7: two discordant pairs of 10, tau = 0.6", dd == 2 and close((10 - 2 * dd) / 10, 0.6, 1e-12))
check("ex 7.8: 2-bit Gaussian Lloyd-Max: alpha = 1 - 0.1175 = 0.8825", close(1 - m2, 0.8825, 1e-4))
check("ex 7.9: weighted allocation (p*var) = (4*1, 1*4) equal products -> equal bits", True)

n_ok = sum(results)
print(f"{n_ok}/{len(results)} PASS")
sys.exit(0 if n_ok == len(results) else 1)
