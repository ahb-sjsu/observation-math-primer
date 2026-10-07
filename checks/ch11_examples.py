"""Checks every number in chapter 11 (high-dimensional geometry, concentration and hubs),
its exercise answers and the coordinates of its figures.

Run on Atlas:  /home/claude/env/bin/python3 ch11_examples.py
Prints PASS/FAIL per claim, FIGDATA lines with the plotted coordinates, then 'k/n PASS';
exits nonzero on any FAIL. Every simulation is seeded and runs in seconds.
"""
import math
import sys
import numpy as np
import sympy as sp
from scipy import stats
from scipy.special import gammaln, ndtri
from scipy.stats import norm, chi2, poisson, beta, binom

results = []


def check(desc, cond):
    ok = bool(cond)
    results.append(ok)
    print(("PASS" if ok else "FAIL") + "  " + desc)


def figdata(name, pts, fmt="({:.4g},{:.4g})"):
    print(f"FIGDATA {name}| " + " ".join(fmt.format(*p) for p in pts))


def close(a, b, tol=5e-4):
    return abs(float(a) - float(b)) < tol


def skew(c):
    c = np.asarray(c, dtype=float)
    m = c.mean(); s = c.std()
    return ((c - m) ** 3).mean() / s ** 3


def sqdist(X, Y=None):
    Y = X if Y is None else Y
    a = (X * X).sum(1); b = (Y * Y).sum(1)
    D = a[:, None] + b[None, :] - 2.0 * X @ Y.T
    return np.maximum(D, 0.0)


def nk_counts(X, k):
    """Corpus->corpus k-occurrence counts (each row queries the others, self excluded)."""
    n = X.shape[0]
    D = sqdist(X)
    np.fill_diagonal(D, np.inf)
    idx = np.argpartition(D, k, axis=1)[:, :k]
    return np.bincount(idx.ravel(), minlength=n), D


# ======================================================== Section 1: norms concentrate
d = 100
check("E||x||^2 = d = 100, Var = 2d = 200, sd = 14.14", chi2.mean(d) == 100 and chi2.var(d) == 200 and close(np.sqrt(200), 14.142, 1e-3))
check("relative sd sqrt(2/d) = 0.1414 at d = 100", close(np.sqrt(2 / 100), 0.1414, 1e-4))
cheb = 200 / 30 ** 2
exact = chi2.sf(130, d) + chi2.cdf(70, d)
print("exact P(|chi2_100-100|>=30) =", exact)
check("Chebyshev bound 200/900 = 0.222", close(cheb, 0.2222, 1e-4))
check("exact P(|chi2_100 - 100| >= 30) = 0.0334", close(exact, 0.0334, 5e-4))
Er = np.exp(0.5 * np.log(2) + gammaln((d + 1) / 2) - gammaln(d / 2))
sdr = np.sqrt(d - Er ** 2)
print("E||x|| d=100", Er, "sd", sdr)
check("E||x|| = 9.975 and sd ||x|| = 0.7062 at d = 100", close(Er, 9.975, 5e-4) and close(sdr, 0.7062, 5e-4))
# sd of ||x|| tends to 1/sqrt2
Er4 = np.exp(0.5 * np.log(2) + gammaln((10001) / 2) - gammaln(10000 / 2))
check("sd/mean of ||x|| at d=100 is 7.1 percent", close(sdr / Er, 0.0708, 5e-4))
check("Chebyshev 0.222 is about 6.7 times the exact 0.0334", close(cheb / exact, 6.66, 0.02))
check("sd ||x|| -> 1/sqrt(2) = 0.7071 (d = 10000 gives 0.7071)", close(np.sqrt(10000 - Er4 ** 2), 0.7071, 1e-4))
# Chebyshev-on-norm bound P(| ||x|| - sqrt d | >= beta) <= 2/beta^2
check("norm bound 2/beta^2: beta = 3 gives 0.222", close(2 / 9, 0.2222, 1e-4))
# Monte Carlo of the norm bound and exact value, d = 400, beta = 3
rng = np.random.default_rng(11)
R = np.sqrt((rng.standard_normal((20000, 400)) ** 2).sum(1))
mc = np.mean(np.abs(R - 20) >= 3)
ex400 = chi2.sf(23 ** 2, 400) + chi2.cdf(17 ** 2, 400)
print("d=400 beta=3: exact", ex400, "MC", mc)
check("d = 400, beta = 3: exact P = 2.25e-5", close(ex400 * 1e5, 2.25, 0.005))
check("d = 400, beta = 3: exact P = 2.2e-5 (far below 0.222), MC agrees", ex400 < 3e-5 and ex400 > 1.5e-5 and mc < 3e-4)
# d = 2: P(||x|| <= 1) = 1 - e^{-1/2}
check("d = 2: P(||x||<=1) = 1 - e^{-1/2} = 0.3935", close(chi2.cdf(1, 2), 1 - np.exp(-0.5)) and close(1 - np.exp(-0.5), 0.3935, 1e-4))
print("log10 P(||x||<=1), d=100:", chi2.logcdf(1, 100) / np.log(10))
check("d = 100: P(||x||<=1) < 1e-79", chi2.logcdf(1, 100) / np.log(10) < -79)

# Figure 1: chi densities of ||x||, d = 1, 10, 100, 400


def chipdf(r, k):
    return np.exp((k - 1) * np.log(r) - r * r / 2 - (k / 2 - 1) * np.log(2) - gammaln(k / 2))


for k in [1, 10, 100, 400]:
    c0 = np.sqrt(k)
    lo, hi = max(0.02, c0 - 3.2), c0 + 3.2
    rs = np.round(np.linspace(lo, hi, 33), 3)
    pts = [(r, chipdf(r, k)) for r in rs]
    figdata(f"fig1_chi_d{k}", pts, "({:.3f},{:.4f})")
    mode = np.sqrt(k - 1) if k > 1 else 0.0
    print(f"chi d={k}: mean {np.exp(0.5*np.log(2)+gammaln((k+1)/2)-gammaln(k/2)):.4f} mode {mode:.4f} peak {chipdf(max(mode,1e-9), k) if k>1 else chipdf(0.02,1):.4f}")
check("fig1: chi densities integrate to 1 (d = 10, 100, 400)",
      all(close(np.trapezoid(chipdf(np.linspace(1e-6, 60, 200001), k), np.linspace(1e-6, 60, 200001)), 1, 1e-4) for k in [10, 100, 400]))
check("fig1: peak heights 0.569 (d=10), 0.565 (d=100), 0.564 (d=400), near 1/sqrt(pi) = 0.564",
      close(chipdf(3.0, 10), 0.569, 5e-4) and close(chipdf(np.sqrt(99), 100), 0.565, 5e-4) and close(chipdf(np.sqrt(399), 400), 0.564, 5e-4) and close(1 / np.sqrt(np.pi), 0.564, 5e-4))

# ======================================================== Section 2: shell and equator
check("shell fraction 1-(0.95)^100 = 0.9941", close(1 - 0.95 ** 100, 0.9941, 1e-4))
check("Ex: shell d=50 eps=0.02: 1-0.98^50 = 0.6358, bound 1-e^-1 = 0.6321", close(1 - 0.98 ** 50, 0.6358, 1e-4) and close(1 - np.exp(-1), 0.6321, 1e-4))
check("shell fraction 1-(0.95)^3 = 0.1426", close(1 - 0.95 ** 3, 0.1426, 1e-4))
check("(1-eps)^d <= e^{-eps d}: 0.95^100 = 0.00592 <= e^-5 = 0.00674", close(0.95 ** 100, 0.00592, 1e-5) and 0.95 ** 100 <= np.exp(-5) and close(np.exp(-5), 0.00674, 1e-5))
V = lambda k: np.exp(k / 2 * np.log(np.pi) - gammaln(k / 2 + 1))
check("V(2) = pi, V(3) = 4pi/3", close(V(2), np.pi, 1e-12) and close(V(3), 4 * np.pi / 3, 1e-12))
check("V(5) = 8 pi^2/15 = 5.2638 is the largest", close(V(5), 8 * np.pi ** 2 / 15, 1e-12) and close(V(5), 5.2638, 1e-4) and V(5) == max(V(k) for k in range(1, 60)))
check("V(20) = pi^10/10! = 0.02581", close(V(20), np.pi ** 10 / 3628800, 1e-12) and close(V(20), 0.02581, 1e-5))
# cosine of two random directions: cos^2 ~ Beta(1/2, (d-1)/2), E cos^2 = 1/d
for k in [3, 10, 100, 1000]:
    check(f"E cos^2 = 1/d for d={k}", close(beta.mean(0.5, (k - 1) / 2), 1 / k, 1e-12))
p03 = beta.sf(0.09, 0.5, 99 / 2)
print("P(|cos|>0.3), d=100:", p03)
check("P(|cos| > 0.3) at d = 100 is 0.0023, Chebyshev gives 0.111", close(p03, 0.0023, 1e-4) and close(0.01 / 0.09, 0.1111, 1e-4))
rng = np.random.default_rng(12)
U = rng.standard_normal((200000, 100)); W = rng.standard_normal((200000, 100))
cs = (U * W).sum(1) / np.linalg.norm(U, axis=1) / np.linalg.norm(W, axis=1)
check("MC: sd of cos at d=100 near 0.1, tail near 0.0023", close(cs.std(), 0.1, 2e-3) and abs(np.mean(np.abs(cs) > 0.3) - p03) < 6e-4)
check("d = 3: cosine is uniform on [-1,1] (density 1/2), P(|cos|>0.3) = 0.7", close(beta.sf(0.09, 0.5, 1), 0.7, 1e-12))
# ball equator: fraction of the d=100 ball volume with |x1| <= 0.2 ; x1^2 ~ Beta(1/2,(d+1)/2)
eqb = beta.cdf(0.04, 0.5, 101 / 2)
print("ball d=100 |x1|<=0.2:", eqb)
check("fraction of the d=100 unit ball with |x1| <= 0.2 is 0.9572", close(eqb, 0.9572, 5e-4))


def cospdf(t, k):
    return np.exp(gammaln(k / 2) - 0.5 * np.log(np.pi) - gammaln((k - 1) / 2)) * (1 - t * t) ** ((k - 3) / 2)


for k in [10, 100, 1000]:
    w = 4.5 / np.sqrt(k)
    ts = np.round(np.linspace(-min(w, 0.99), min(w, 0.99), 31), 4)
    figdata(f"fig2_cos_d{k}", [(t, cospdf(t, k)) for t in ts], "({:.4f},{:.4f})")
print("cos peaks", cospdf(0, 10), cospdf(0, 100), cospdf(0, 1000), np.sqrt(1000 / 2 / np.pi))
check("fig2: peak of cos density near sqrt(d/(2 pi)): d=1000 gives 12.61, d=100 gives 3.959",
      close(cospdf(0, 1000), 12.61, 5e-3) and close(cospdf(0, 100), 3.959, 5e-3))

# ======================================================== Section 3: pairwise distances
check("||x-y||^2 = 2 chi2_d: mean 2d = 200, var 8d = 800 at d = 100", 2 * 100 == 200 and 4 * chi2.var(100) == 800)
check("relative sd sqrt(8d)/(2d) = sqrt(2/d): 1, 0.1414, 0.01414 at d=2,100,10000",
      close(np.sqrt(8 * 2) / 4, 1) and close(np.sqrt(800) / 200, 0.1414, 1e-4) and close(np.sqrt(80000) / 20000, 0.01414, 1e-5))
# Figure 3: median DMAX/DMIN over 100 queries against n = 1000 data points
ratio_pts = []
for dd in [2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000]:
    rng = np.random.default_rng(300 + dd)
    X = rng.standard_normal((1000, dd)); Q = rng.standard_normal((100, dd))
    Dq = np.sqrt(sqdist(Q, X))
    r = Dq.max(1) / Dq.min(1)
    ratio_pts.append((dd, np.median(r)))
figdata("fig3_ratio", ratio_pts, "({:d},{:.3f})")
rp = dict(ratio_pts)
print("ratio", rp)
check("fig3: ratio decreases with d", all(ratio_pts[i][1] > ratio_pts[i + 1][1] for i in range(len(ratio_pts) - 1)))
check("fig3: ratio at d=2 above 20, at d=100 between 1.4 and 1.8, at d=10000 below 1.07",
      rp[2] > 20 and 1.4 < rp[100] < 1.8 and rp[10000] < 1.07)
# normal approximation of the d=100 ratio: z = 3.09 for 1/1000
za = norm.isf(1e-3)
lo2, hi2 = 200 - za * np.sqrt(800), 200 + za * np.sqrt(800)
print("normal approx d=100", za, lo2, hi2, np.sqrt(hi2 / lo2))
check("normal approximation at d=100: 200 -+ 3.09*28.28 = 112.6, 287.4, ratio 1.60",
      close(za, 3.090, 1e-3) and close(lo2, 112.6, 0.05) and close(hi2, 287.4, 0.05) and close(np.sqrt(hi2 / lo2), 1.60, 5e-3))

# ======================================================== Section 4: Johnson-Lindenstrauss


def jl_k(n, eps):
    return 4 * np.log(n) / (eps ** 2 / 2 - eps ** 3 / 3)


print("jl k", jl_k(1000, 0.2), jl_k(1000, 0.5), jl_k(1e6, 0.2), jl_k(1e4, 0.1), jl_k(200, 0.5))
check("JL: n = 1000, eps = 0.2: eps^2/2 - eps^3/3 = 0.017333, k >= 1594.1 -> 1595",
      close(0.02 - 0.008 / 3, 0.017333, 1e-6) and close(jl_k(1000, 0.2), 1594.1, 0.05) and int(np.ceil(jl_k(1000, 0.2))) == 1595)
check("JL: n = 1000, eps = 0.5: 0.083333, k >= 331.6 -> 332", close(jl_k(1000, 0.5), 331.6, 0.05) and int(np.ceil(jl_k(1000, 0.5))) == 332)
check("JL: n = 1e6, eps = 0.2: k >= 3188.2 -> 3189", close(jl_k(1e6, 0.2), 3188.2, 0.05) and int(np.ceil(jl_k(1e6, 0.2))) == 3189)
check("Ex: n = 1e4, eps = 0.1: k >= 7894.6 -> 7895", close(jl_k(1e4, 0.1), 7894.6, 0.05) and int(np.ceil(jl_k(1e4, 0.1))) == 7895)
# Chernoff for chi-square, symbolic optimisation
lam, e, kk = sp.symbols('lambda epsilon k', positive=True)
logb = -lam * (1 + e) * kk - kk / 2 * sp.log(1 - 2 * lam)
lstar = sp.solve(sp.diff(logb, lam), lam)
check("Chernoff: optimal lambda = eps/(2(1+eps))", len(lstar) == 1 and sp.simplify(lstar[0] - e / (2 * (1 + e))) == 0)
check("Chernoff: optimum value = (k/2)(log(1+eps) - eps)",
      sp.simplify(sp.expand_log(logb.subs(lam, lstar[0]) - kk / 2 * (sp.log(1 + e) - e), force=True)) == 0)
logb2 = lam * (1 - e) * kk - kk / 2 * sp.log(1 + 2 * lam)
lstar2 = sp.solve(sp.diff(logb2, lam), lam)
lv = lambda x, ep: x * (1 - ep) * 100 - 50 * np.log(1 + 2 * x)
ok_low = True
for ep in [0.1, 0.3, 0.7]:
    xs_ = np.linspace(0, 20, 400001); vals = lv(xs_, ep)
    ok_low &= close(xs_[np.argmin(vals)], ep / (2 * (1 - ep)), 1e-3) and close(vals.min(), 50 * (np.log(1 - ep) + ep), 1e-6)
check("Chernoff lower tail: lambda = eps/(2(1-eps)), value (k/2)(log(1-eps)+eps) (numeric, k=100)", ok_low)
ee = np.linspace(1e-4, 0.999, 2000)
check("log(1+eps) - eps <= -eps^2/2 + eps^3/3 and log(1-eps)+eps <= -eps^2/2 on (0,1)",
      np.all(np.log1p(ee) - ee <= -ee ** 2 / 2 + ee ** 3 / 3 + 1e-15) and np.all(np.log1p(-ee) + ee <= -ee ** 2 / 2 + 1e-15))
ub = (1.2 * np.exp(-0.2)) ** 50
ex = chi2.sf(120, 100)
print("k=100 eps=0.2: bound", ub, "exact", ex)
check("k = 100, eps = 0.2: Chernoff bound 0.4132, exact P(chi2_100 >= 120) = 0.0844", close(ub, 0.4132, 5e-4) and close(ex, 0.0844, 5e-4))
check("union: n(n-1) n^-2 = 1 - 1/n", sp.simplify(sp.Symbol('n') * (sp.Symbol('n') - 1) / sp.Symbol('n') ** 2 - (1 - 1 / sp.Symbol('n'))) == 0)
# Figure 4: max pairwise distortion of squared distances vs k, n = 200 points in d = 1000
rng = np.random.default_rng(400)
nX, dX = 200, 1000
X = rng.standard_normal((nX, dX))
D0 = sqdist(X)
iu = np.triu_indices(nX, 1)
jl_pts = []; rms_pts = []
for k in [10, 20, 50, 100, 200, 400, 800, 1600]:
    mx = []
    for rep in range(5):
        G = np.random.default_rng(1000 * k + rep).standard_normal((dX, k)) / np.sqrt(k)
        Y = X @ G
        rr = sqdist(Y)[iu] / D0[iu]
        mx.append(np.max(np.abs(rr - 1)))
    jl_pts.append((k, np.median(mx)))
    rms_pts.append((k, np.sqrt(2 / k)))
figdata("fig4_measured", jl_pts, "({:d},{:.4f})")
figdata("fig4_rms", rms_pts, "({:d},{:.4f})")


def jl_eps(n, k):
    from scipy.optimize import brentq
    f = lambda x: 4 * np.log(n) / (x ** 2 / 2 - x ** 3 / 3) - k
    if f(0.999) > 0:
        return None
    return brentq(f, 1e-3, 0.999)


bound_pts = [(k, jl_eps(200, k)) for k in [128, 150, 200, 300, 400, 600, 800, 1200, 1600]]
figdata("fig4_bound", bound_pts, "({:d},{:.4f})")
mp = dict(jl_pts)
print("fig4 measured", mp)
check("fig4: measured max distortion stays below the JL bound wherever the bound is < 1",
      all(mp[k] < jl_eps(200, k) for k in [200, 400, 800, 1600]))
check("fig4: measured max distortion falls with k", all(jl_pts[i][1] > jl_pts[i + 1][1] for i in range(len(jl_pts) - 1)))
check("fig4: bound needs k >= 4 ln 200 / (1/6) = 127.2 to reach eps < 1", close(4 * np.log(200) * 6, 127.2, 0.05))
check("fig4: n=200 eps=0.5 needs k >= 254.3", close(jl_k(200, 0.5), 254.3, 0.05))
# rotation check: k = d with an orthogonal matrix preserves all distances exactly
Qm, Rm = np.linalg.qr(np.random.default_rng(401).standard_normal((dX, dX)))
check("orthogonal rotation preserves every pairwise squared distance", np.max(np.abs(sqdist(X @ Qm)[iu] / D0[iu] - 1)) < 1e-10)

# ======================================================== Section 5: N_k, hand example
P5 = np.array([[0, 0], [1.8, 0], [-2, 0], [0, 2.5], [0, -3]], dtype=float)
D5 = np.sqrt(sqdist(P5))
check("five points: distances A-B 1.8, B-D 3.081, B-E 3.499, C-D 3.202, C-E 3.606",
      close(D5[0, 1], 1.8) and close(D5[1, 3], 3.081, 5e-4) and close(D5[1, 4], 3.499, 5e-4) and close(D5[2, 3], 3.202, 5e-4) and close(D5[2, 4], 3.606, 5e-4))
c1, _ = nk_counts(P5, 1)
c2, _ = nk_counts(P5, 2)
check("five points: N_1 = (4,1,0,0,0)", list(c1) == [4, 1, 0, 0, 0])
check("five points: skew N_1 = 4.8/2.4^1.5 = 1.291", close(skew(c1), 1.291, 5e-4) and close(4.8 / 2.4 ** 1.5, 1.291, 5e-4))
check("Ex: five points: N_2 = (4,3,1,2,0), sum 10, skew 0", list(c2) == [4, 3, 1, 2, 0] and c2.sum() == 10 and abs(skew(c2)) < 1e-12)
check("Ex: N_2 variance 2", close(np.var(c2), 2.0, 1e-12))

# ======================================================== Sections 5-7: Gaussian simulations, n = 2000, k = 10
n, k = 2000, 10
m = n - 1
pnull = k / m
null_skew = (1 - 2 * pnull) / np.sqrt(m * pnull * (1 - pnull))
print("binomial null skew", null_skew, "poisson", 1 / np.sqrt(k))
check("null skewness: binomial(1999, 10/1999) gives 0.3138, Poisson(10) gives 1/sqrt10 = 0.3162",
      close(null_skew, 0.3138, 5e-4) and close(1 / np.sqrt(10), 0.3162, 5e-4))
# large-d model: p(z) = Phi((sqrt6 c - sqrt2 z)/2), c = Phi^{-1}(k/(n-1))
cq = ndtri(k / m)
pz = lambda z: norm.cdf((np.sqrt(6) * cq - np.sqrt(2) * z) / 2)
zg = np.linspace(-9, 9, 36001); wz = norm.pdf(zg); wz /= wz.sum()
P = pz(zg)
EN1 = (wz * m * P).sum()
EN2 = (wz * (m * P * (1 - P) + (m * P) ** 2)).sum()
EN3 = (wz * m * P * (1 - 3 * P + 3 * m * P + 2 * P * P - 3 * m * P * P + m * m * P * P)).sum()
vmod = EN2 - EN1 ** 2
smod = (EN3 - 3 * EN1 * EN2 + 2 * EN1 ** 3) / vmod ** 1.5
p0mod = (wz * (1 - P) ** m).sum()
print("model: c", cq, "mean", EN1, "skew", smod, "P(N=0)", p0mod)
check("model: c = Phi^-1(10/1999) = -2.5755", close(cq, -2.5755, 5e-4))
check("model: mean count = k = 10 exactly", close(EN1, 10.0, 1e-6))
check("model: limiting skewness 8.26", close(smod, 8.26, 0.01))
check("model: anti-hub (N=0) fraction 0.36", close(p0mod, 0.36, 0.01))
check("model: z=0 gives p = Phi(-3.154) = 0.000805, expected count 1.61",
      close(pz(0), 0.000805, 5e-6) and close(m * pz(0), 1.61, 0.01) and close(np.sqrt(6) * cq / 2, -3.154, 1e-3))
check("model: z=-2 gives p = Phi(-1.740) = 0.0409, expected count 81.8",
      close((np.sqrt(6) * cq + 2 * np.sqrt(2)) / 2, -1.740, 1e-3) and close(pz(-2), 0.0409, 2e-4) and close(m * pz(-2), 81.8, 0.2))
check("model: count ratio 81.8/1.61 is about 50", close(pz(-2) / pz(0), 50.8, 0.1))
check("model: d=100, two sd closer: ||x||^2 = 100 - 2 sqrt200 = 71.72, E||x-q||^2 = 171.72 vs 200",
      close(100 - 2 * np.sqrt(200), 71.72, 5e-3) and close(171.72 - 71.72, 100))
# exercise model numbers: n = 1001, k = 10
c1001 = ndtri(0.01)
pz1001 = lambda z: norm.cdf((np.sqrt(6) * c1001 - np.sqrt(2) * z) / 2)
print("ex model", c1001, pz1001(0), pz1001(-1), 1000 * pz1001(0), 1000 * pz1001(-1))
check("Ex: c = -2.326, p(0) = Phi(-2.849) = 0.00219 -> 2.19, p(-1) = Phi(-2.142) = 0.0161 -> 16.1",
      close(c1001, -2.326, 1e-3) and close(pz1001(0), 0.00219, 1e-5) and close(1000 * pz1001(-1), 16.1, 0.05)
      and close((np.sqrt(6) * c1001 + np.sqrt(2)) / 2, -2.142, 1e-3))

skew_pts = []
store = {}
for dd in [2, 3, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000]:
    sks = []; mxs = []; zf = []
    for rep in range(10):
        rng = np.random.default_rng(10000 * dd + rep)
        X = rng.standard_normal((n, dd))
        c, D = nk_counts(X, k)
        sks.append(skew(c)); mxs.append(c.max()); zf.append(np.mean(c == 0))
        if rep == 0:
            store[dd] = (X, c, D)
    skew_pts.append((dd, np.mean(sks)))
    store.setdefault("mx", {})[dd] = (min(mxs), max(mxs), np.std(sks))
    print(f"d={dd}: mean skew {np.mean(sks):.3f} sd {np.std(sks):.3f} max N median {np.median(mxs):.0f} range {min(mxs)}-{max(mxs)} anti-hub frac {np.mean(zf):.3f}")
figdata("fig6_skew", skew_pts, "({:d},{:.3f})")
sp_ = dict(skew_pts)
check("seed maxima: d=2 range 17-20, d=3 range 18-20, all at or below 22", store["mx"][2][:2] == (17, 20) and store["mx"][3][:2] == (18, 20))
check("seed scatter of skewness at d >= 200 between 0.9 and 2", all(0.9 < store["mx"][x][2] < 2 for x in [200, 500, 1000, 2000, 5000, 10000]))
check("fig6: d=10000 mean skew 7.78", close(sp_[10000], 7.778, 1e-3))
check("fig6: low-d skew is negative (d = 2, 3, 5)", sp_[2] < 0 and sp_[3] < 0 and sp_[5] < 0)
check("fig6: skew rises past the Poisson null by d = 10 and exceeds 5 by d = 100", sp_[10] > null_skew and sp_[100] > 5)
check("fig6: skew at d >= 200 between 5.5 and 8.26 (model limit)", all(5.5 < sp_[x] < smod for x in [200, 500, 1000, 2000, 5000, 10000]))

# Figure 5: survival function of N_10 for d=3 and d=1000 (rep 0), with Poisson(10)
_, c3, _ = store[3]
X1k, c1k, D1k = store[1000]
print("d=3 max", c3.max(), "d=1000 max", c1k.max(), "d=1000 frac zero", np.mean(c1k == 0), "skew d=3", skew(c3), "skew d=1000", skew(c1k))
xs3 = list(range(0, 21, 2)) + [c3.max(), c3.max() + 1]
sv3 = [(x, np.mean(c3 >= x)) for x in sorted(set(xs3))]
xs1k = [0, 1, 2, 5, 10, 20, 30, 50, 75, 100, 150, 200, 250, 300, int(c1k.max())]
sv1k = [(x, np.mean(c1k >= x)) for x in xs1k]
svp = [(x, poisson.sf(x - 1, 10)) for x in range(0, 31, 2)]
figdata("fig5_d3", sv3, "({:d},{:.5f})")
figdata("fig5_d1000", sv1k, "({:d},{:.5f})")
figdata("fig5_poisson", svp, "({:d},{:.3e})")
check("fig5: survival at 0 is 1 for both", sv3[0][1] == 1 and sv1k[0][1] == 1)
check("fig5: d=1000 sample has max N_10 = 386, 28.7% zeros, skew 7.37", c1k.max() == 386 and close(np.mean(c1k == 0), 0.287, 1e-3))
check("fig5: d=3 sample has max N_10 = 19, skew -0.38, under 1% zeros", c3.max() == 19 and close(skew(c3), -0.38, 5e-3) and np.mean(c3 == 0) < 0.01)
check("fig5: every plotted survival point is a fraction of 2000 points (or Poisson)", all(abs(v * 2000 - round(v * 2000)) < 1e-6 for _, v in sv3 + sv1k))

# Figure 7: N_10 against centrality z at d = 1000, with the model curve (n-1) p(z)
sqn = (X1k * X1k).sum(1)
zc = (sqn - 1000) / np.sqrt(2000)
rho_c = stats.spearmanr(c1k, -sqn).correlation
dk = np.sqrt(np.sort(D1k, axis=1)[:, k - 1])
rho_d = stats.spearmanr(c1k, -dk).correlation
print("d=1000 spearman(N, -||x||) =", rho_c, " spearman(N, -d_k) =", rho_d)
check("fig7: Spearman corr(N_10, -||x||) at d = 1000 is above 0.9", rho_c > 0.9)
check("corr(N_10, -d_10) on an isotropic Gaussian exceeds 0.8 (d = 1000)", rho_d > 0.8)
sel = np.random.default_rng(700).choice(n, 400, replace=False)
figdata("fig7_scatter", [(zc[i], c1k[i]) for i in sel], "({:.2f},{:d})")
figdata("fig7_model", [(z, m * pz(z)) for z in np.arange(-3.0, 3.01, 0.25)], "({:.2f},{:.2f})")
# model vs simulation in z bins
for lo_, hi_ in [(-3, -2), (-2, -1), (-1, 0), (0, 1), (1, 2)]:
    msk = (zc >= lo_) & (zc < hi_)
    zm = zc[msk].mean()
    print(f"z in [{lo_},{hi_}): n={msk.sum()} mean N {c1k[msk].mean():.2f} model at mean z {m*pz(zm):.2f}")
msk = (zc >= -2) & (zc < -1)
check("fig7: points with z in [-2,-1) average 36.1, 3.6 times the overall mean count 10", close(c1k[msk].mean(), 36.07, 0.01))
check("model at bin mean z: 28.9 for [-2,-1), 0.50 for [0,1)", close(m * pz(zc[msk].mean()), 28.86, 0.01) and close(m * pz(zc[(zc >= 0) & (zc < 1)].mean()), 0.50, 0.005))
check("Spearman corr(N_10, -||x||) = 0.93 and corr(N_10, -d_10) = 0.92", close(rho_c, 0.93, 5e-3) and close(rho_d, 0.92, 5e-3))
check("d=1000 sample anti-hub fraction 0.29 vs model 0.36", close(np.mean(c1k == 0), 0.29, 5e-3))
msk2 = (zc >= 0) & (zc < 1)
check("fig7: points with z in [0,1) average 1.04", close(c1k[msk2].mean(), 1.04, 0.01))

# local scaling on the d = 1000 sample: rank candidate x for query q by d_qx / sqrt(sigma_x)
Dn = np.sqrt(D1k)
sig = np.sort(Dn, axis=1)[:, k - 1]
Ls = Dn / np.sqrt(sig)[None, :]
np.fill_diagonal(Ls, np.inf)
idxls = np.argpartition(Ls, k, axis=1)[:, :k]
cls = np.bincount(idxls.ravel(), minlength=n)
print("local scaling: skew", skew(cls), "max", cls.max(), "before", skew(c1k), c1k.max())
check("local scaling on the d=1000 sample: skew 7.37 -> 2.54, max N_10 386 -> 113", close(skew(cls), 2.54, 5e-3) and cls.max() == 113 and close(skew(c1k), 7.37, 5e-3) and c1k.max() == 386)
# centering: Euclidean N_k unchanged by a shift
cshift, _ = nk_counts(X1k + 5.0, k)
check("Euclidean N_k is unchanged by centering or any shift", np.array_equal(cshift, c1k))
# linear isometric embedding leaves N_k unchanged (intrinsic dimension)
X5 = np.random.default_rng(705).standard_normal((n, 5))
Qe, _ = np.linalg.qr(np.random.default_rng(706).standard_normal((500, 5)))
c5, _ = nk_counts(X5, k); c5e, _ = nk_counts(X5 @ Qe.T, k)
check("Ex: a 5-dim Gaussian embedded isometrically in 500 dims has identical N_k", np.array_equal(c5, c5e))

# ======================================================== Section 7: Poisson null


def ceiling(nn, mu):
    c = 0
    while nn * poisson.sf(c, mu) >= 1:  # sf(c) = P(X >= c+1)
        c += 1
    return c


cst = ceiling(2000, 10)
print("ceiling n=2000 mu=10:", cst, [2000 * poisson.sf(c - 1, 10) for c in [21, 22, 23]])
check("ceiling n=2000, mu=10 is 22: 2000 P(X>=22) = 1.40, 2000 P(X>=23) = 0.59",
      cst == 22 and close(2000 * poisson.sf(21, 10), 1.40, 0.01) and close(2000 * poisson.sf(22, 10), 0.59, 0.01))
mx3 = max(store[2][1].max(), store[3][1].max())
check("d=2,3 samples: busiest count at or below the ceiling 22", mx3 <= 22)
check("d=1000 sample: hub excess 386/22 = 17.5", close(c1k.max() / 22, 17.5, 0.05))
print("hub excess d=1000:", c1k.max() / 22)
cs2 = ceiling(10000, 5)
print("ceiling n=1e4 mu=5:", cs2, 1e4 * poisson.sf(cs2 - 1, 5), 1e4 * poisson.sf(cs2, 5))
check("example n=10000, k=5: ceiling 15, 1e4 P(X>=15) = 2.26, 1e4 P(X>=16) = 0.69", cs2 == 15 and close(1e4 * poisson.sf(14, 5), 2.26, 0.01) and close(1e4 * poisson.sf(15, 5), 0.69, 0.01))
cs3 = ceiling(5000, 20)
print("ceiling n=5000 mu=20:", cs3, 5000 * poisson.sf(cs3 - 1, 20), 5000 * poisson.sf(cs3, 20))
check("Ex: hub excess 190/38 = 5", 190 / 38 == 5)
check("Ex: n=5000, k=20: ceiling 38, 5000 P(X>=38) = 1.09, 5000 P(X>=39) = 0.54", cs3 == 38 and close(5000 * poisson.sf(37, 20), 1.09, 0.01) and close(5000 * poisson.sf(38, 20), 0.54, 0.01))
check("Poisson skewness is 1/sqrt(mu): 5 gives 0.447, 20 gives 0.224",
      close(poisson.stats(5, moments='s'), 1 / np.sqrt(5), 1e-12) and close(1 / np.sqrt(5), 0.4472, 1e-4) and close(1 / np.sqrt(20), 0.2236, 1e-4))
# binomial -> Poisson: P(N=0) for Binomial(1999, 10/1999) vs e^-10
print("P(N=0) binom", binom.pmf(0, 1999, 10 / 1999), "poisson", np.exp(-10))
check("P(N=0): binomial 4.43e-5 vs Poisson e^-10 = 4.54e-5",
      close(binom.pmf(0, 1999, 10 / 1999) * 1e5, 4.43, 0.01) and close(np.exp(-10) * 1e5, 4.54, 0.01))

# ======================================================== Section 8: mean recall, slot shares
check("recall example: 0.9*0.98 + 0.1*0.62 = 0.944", close(0.9 * 0.98 + 0.1 * 0.62, 0.944, 1e-12))
check("recall example: anti-hub stratum to 0.40 gives 0.922, mean moves 0.022 while stratum moves 0.22",
      close(0.9 * 0.98 + 0.1 * 0.40, 0.922, 1e-12) and close(0.944 - 0.922, 0.022, 1e-12))


def lorenz(c, fracs):
    s = np.sort(c)[::-1].astype(float)
    cum = np.cumsum(s) / s.sum()
    return [(f, 0.0 if f == 0 else cum[int(round(f * len(s))) - 1]) for f in fracs]


fr = [0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
L3 = lorenz(c3, fr); L1k = lorenz(c1k, fr)
figdata("fig8_d3", L3, "({:.2f},{:.4f})")
figdata("fig8_d1000", L1k, "({:.2f},{:.4f})")
top1_3 = dict(L3)[0.01]; top1_1k = dict(L1k)[0.01]; top10_1k = dict(L1k)[0.1]
print("top 1%: d=3", top1_3, "d=1000", top1_1k, "top 10% d=1000", top10_1k)
rh = lambda c: 0.5 * np.abs(c - c.mean()).sum() / c.sum()
print("Robin Hood: d=3", rh(c3), "d=1000", rh(c1k))
check("fig8: 28.7 percent of points (29) hold no slot at d=1000, bottom 30 percent hold 0.13 percent", close(np.mean(c1k == 0), 0.287, 1e-3) and close(1 - dict(L1k)[0.7], 0.0013, 1e-4))
check("fig8: top 1% of points hold 1.7% of slots at d=3 and 18.0% at d=1000, top 10% hold 59.8%", close(top1_3, 0.0169, 1e-3) and close(top1_1k, 0.1797, 1e-3) and close(top10_1k, 0.5976, 1e-3))
check("Robin Hood index 0.109 at d=3 and 0.591 at d=1000", close(rh(c3), 0.109, 1e-3) and close(rh(c1k), 0.591, 1e-3))
check("Robin Hood index of N_2 example (4,3,1,2,0): 0.3", close(rh(c2), 0.3, 1e-12))
check("Robin Hood index of N_1 example (4,1,0,0,0): 0.6", close(rh(c1), 0.6, 1e-12))

# ---------------------------------------------------------------- hypothesis-discipline additions
# JL with constant 6: each tail <= n^-3, union over n(n-1)/2 pairs and two tails gives failure < 1/n
okJL = True
for n_ in [2, 10, 1000, 10 ** 6]:
    for eps_ in [0.1, 0.2, 0.5, 0.9]:
        m_ = 6 * math.log(n_) / (eps_ ** 2 / 2 - eps_ ** 3 / 3)
        tail_ = math.exp(-(m_ / 2) * (eps_ ** 2 / 2 - eps_ ** 3 / 3))
        okJL &= tail_ <= n_ ** -3 * (1 + 1e-9) and n_ * (n_ - 1) / 2 * 2 * tail_ < 1 / n_
check("JL with 6 ln n: tail <= n^-3 and failure probability < 1/n", okJL)
check("JL with 4 ln n: failure bound n(n-1)/n^2 = 1 - 1/n, so success only >= 1/n (n=1000)", close(1000 * 999 / 1000 ** 2, 1 - 1 / 1000, 1e-12))
# star-shaped requirement: an annulus not containing the origin is not mapped into itself by shrinking
# body A = {1 <= |x| <= 2} in R^2; the point (1,0) is in A, shrunk by 0.9 to (0.9,0) which is not in A
check("shrinking the annulus 1<=|x|<=2 by 0.9 maps (1,0) outside it", not (1 <= 0.9 <= 2))
# centrality is an average over queries: a central point need not be closer to every query (d=2 example)
xc = np.array([0.0, 0.0]); xf = np.array([3.0, 0.0]); qq = np.array([3.0, 0.1])
check("query (3,0.1) is closer to the far point (3,0) than to the mean point (0,0)", np.linalg.norm(qq - xf) < np.linalg.norm(qq - xc))

n_pass = sum(results)
print(f"{n_pass}/{len(results)} PASS")
sys.exit(0 if n_pass == len(results) else 1)
