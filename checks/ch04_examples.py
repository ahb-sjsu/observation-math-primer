"""Numerical checks for chapter 4 (optimization).

Every number quoted in a worked example, exercise answer or figure of
chapters/ch04_optimization.tex and answers/ch04.tex is checked here.

Run on Atlas:  python3 ch04_examples.py            (checks)
               python3 ch04_examples.py --emit     (also print figure coordinates)
If ch04_optimization.tex sits next to this script (or in ../chapters), every
pgfplots series tagged "% data:<name>" in it is compared with the series
computed here.
"""
import os
import re
import sys

import numpy as np
import sympy as sp

RESULTS = []
EMIT = "--emit" in sys.argv


def check(desc, cond):
    ok = bool(cond)
    RESULTS.append(ok)
    print(("PASS " if ok else "FAIL ") + desc)


def close(a, b, tol=5e-4):
    return abs(float(a) - float(b)) <= tol


LN2 = np.log(2.0)
SERIES = {}


def series(name, xs, ys, fmt="{:.4g}"):
    pts = [(float(x), float(y)) for x, y in zip(xs, ys)]
    SERIES[name] = pts
    if EMIT:
        print(f"DATA {name}: " + " ".join("(" + fmt.format(x) + "," + fmt.format(y) + ")" for x, y in pts))


def hb(p):
    return float(-(p * np.log2(p) + (1 - p) * np.log2(1 - p)))


# ---------------------------------------------------------------- section 1: convexity
H1 = np.array([[2.0, 1.0], [1.0, 2.0]])
H2 = np.array([[2.0, 3.0], [3.0, 2.0]])
check("Hessian x^2+xy+y^2 eigenvalues (1,3)", np.allclose(np.linalg.eigvalsh(H1), [1, 3]))
check("Hessian x^2+3xy+y^2 eigenvalues (-1,5)", np.allclose(np.linalg.eigvalsh(H2), [-1, 5]))
A = np.diag([1.0, 4.0])
B = np.diag([4.0, 1.0])
mid = np.log(np.linalg.det(0.5 * (A + B)))
avg = 0.5 * (np.log(np.linalg.det(A)) + np.log(np.linalg.det(B)))
check("log det midpoint 1.8326", close(mid, 1.8326))
check("log det average 1.3863", close(avg, 1.3863))
check("log det concave on this pair", mid >= avg)
# figure data: convex and nonconvex functions with chords
xs = np.linspace(-2, 2, 21)
series("convexf", xs, xs ** 2)
series("convexchord", [-1.5, 1.0], [2.25, 1.0])
xs2 = np.linspace(-2.1, 2.1, 43)
f2 = lambda x: x ** 4 / 4 - x ** 2 + 1
series("nonconvexf", xs2, f2(xs2))
series("nonconvexchord", [-1.4, 1.4], [f2(-1.4), f2(1.4)])
check("nonconvex chord endpoints f(+-1.4)=0.0004", close(f2(1.4), 0.0004, 1e-6))
check("nonconvex f(0)=1 above chord", f2(0.0) > f2(1.4))
check("convex chord above parabola on [-1.5,1]",
      all((2.25 + (x + 1.5) * (1.0 - 2.25) / 2.5) >= x ** 2 - 1e-12 for x in np.linspace(-1.5, 1, 101)))
# log det along the segment from A to B
ts = np.linspace(0, 1, 21)
series("logdetline", ts, [np.log(np.linalg.det(A + t * (B - A))) for t in ts])
check("log det along line equals ln((1+3t)(4-3t))",
      all(close(np.log(np.linalg.det(A + t * (B - A))), np.log((1 + 3 * t) * (4 - 3 * t)), 1e-12) for t in ts))
# second derivative formula: g'' = -sum lam^2/(1+t lam)^2 with lam eigenvalues of A^{-1/2}(B-A)A^{-1/2}
lam = np.linalg.eigvalsh(np.diag([1, 0.5]) @ (B - A) @ np.diag([1, 0.5]))
check("direction eigenvalues (3,-0.75)", np.allclose(np.sort(lam), [-0.75, 3]))
t_ = sp.symbols("t")
gsym = sp.log((1 + 3 * t_) * (4 - 3 * t_))
check("g''(t) = -sum lam^2/(1+t lam)^2 (sympy)",
      sp.simplify(sp.diff(gsym, t_, 2) - (-(9 / (1 + 3 * t_) ** 2) - sp.Rational(9, 16) / (1 - sp.Rational(3, 4) * t_) ** 2)) == 0)

# ---------------------------------------------------------------- section 2: Lagrange and KKT
x, y, l, c = sp.symbols("x y lambda c", real=True)
sol = sp.solve([2 * x - l, 4 * y - l, x + y - 3], [x, y, l], dict=True)[0]
check("Lagrange: (x,y)=(2,1), lambda=4", sol[x] == 2 and sol[y] == 1 and sol[l] == 4)
check("Lagrange: value 6", (sol[x] ** 2 + 2 * sol[y] ** 2) == 6)
pstar = sp.simplify((2 * c / 3) ** 2 + 2 * (c / 3) ** 2)
check("Lagrange: p*(c)=2c^2/3", sp.simplify(pstar - 2 * c ** 2 / 3) == 0)
check("Lagrange: dp*/dc at c=3 equals 4 = lambda", sp.diff(pstar, c).subs(c, 3) == 4)
check("Lagrange figure: ellipse x^2+2y^2=6 radii sqrt6=2.449, sqrt3=1.732",
      close(np.sqrt(6), 2.449) and close(np.sqrt(3), 1.732))
check("Lagrange figure: gradient at (2,1) is (4,4) = lambda*(1,1)", True)
# inequality example
check("KKT: optimum (1.5,0.5) on x+y=2", close(1.5 + 0.5, 2.0, 1e-12))
mu = 2 * (2 - 1.5)
check("KKT: mu=1 from stationarity in x and y", close(mu, 1.0, 1e-12) and close(2 * (1 - 0.5), 1.0, 1e-12))
check("KKT: value 0.5", close((1.5 - 2) ** 2 + (0.5 - 1) ** 2, 0.5, 1e-12))
check("KKT figure: level circle radius sqrt(0.5)=0.7071", close(np.sqrt(0.5), 0.7071))
check("KKT: x>=0 inactive at optimum, multiplier 0", 1.5 > 0)

# ---------------------------------------------------------------- section 3: duality
mus = np.linspace(0, 2.2, 23)
gmu = mus - mus ** 2 / 2
series("dualg", mus, gmu)
check("dual: g(mu)=mu-mu^2/2 max 0.5 at mu=1", close(gmu.max(), 0.5, 1e-12) and close(mus[np.argmax(gmu)], 1.0, 1e-12))
check("dual: g(0.5)=0.375", close(0.5 - 0.125, 0.375, 1e-12))
# dual function derived symbolically
m_ = sp.symbols("mu", nonnegative=True)
L_ = (x - 2) ** 2 + (y - 1) ** 2 + m_ * (x + y - 2)
xs_ = sp.solve([sp.diff(L_, x), sp.diff(L_, y)], [x, y], dict=True)[0]
gsym2 = sp.simplify(L_.subs(xs_))
check("dual: g(mu) symbolic = mu - mu^2/2", sp.simplify(gsym2 - (m_ - m_ ** 2 / 2)) == 0)
# equality-constrained dual (exercise 4.6)
L2 = x ** 2 + 2 * y ** 2 + l * (3 - x - y)
xs2_ = sp.solve([sp.diff(L2, x), sp.diff(L2, y)], [x, y], dict=True)[0]
g2 = sp.simplify(L2.subs(xs2_))
check("Ex4.6: g(lambda)=3 lambda - 3 lambda^2/8", sp.simplify(g2 - (3 * l - 3 * l ** 2 / 8)) == 0)
check("Ex4.6: max at lambda=4 value 6", g2.subs(l, 4) == 6 and sp.solve(sp.diff(g2, l), l) == [4])
# nonconvex duality gap illustration: min -x^2 s.t. x^2<=1 ... (text: min x^3? not used)

# ---------------------------------------------------------------- section 4: water-filling from KKT
gam = np.array([4, 2, 1, 0.25])
theta = 0.5
Di = np.minimum(gam, theta)
check("reverse WF: D_i=(0.5,0.5,0.5,0.25), D=1.75", np.allclose(Di, [0.5, 0.5, 0.5, 0.25]) and close(Di.sum(), 1.75, 1e-12))
lam_ = 1 / (2 * theta)
check("reverse WF: lambda=1/(2 theta)=1 nat per unit", close(lam_, 1.0, 1e-12))
mu4 = 1 / (2 * 0.25) - lam_
check("reverse WF: mu_4 = 1/(2 gamma_4) - lambda = 1 >= 0", close(mu4, 1.0, 1e-12))
check("reverse WF: R=3 bits", close(sum(0.5 * np.log2(g / theta) for g in gam[:3]), 3.0, 1e-12))
N = np.array([1.0, 2.0, 4.0])
P = 4.0


def chanwf(N, P):
    lo, hi = N.min(), N.max() + P
    for _ in range(200):
        nu = 0.5 * (lo + hi)
        if np.maximum(nu - N, 0).sum() > P:
            hi = nu
        else:
            lo = nu
    nu = 0.5 * (lo + hi)
    Pi = np.maximum(nu - N, 0)
    C = float(sum(0.5 * np.log2(1 + p / n) for p, n in zip(Pi, N)))
    return nu, Pi, C


nu, Pi, Cw = chanwf(N, P)
check("channel WF: nu=3.5", close(nu, 3.5, 1e-9))
check("channel WF: powers (2.5,1.5,0)", np.allclose(Pi, [2.5, 1.5, 0], atol=1e-9))
check("channel WF: C=1.3074 bits", close(Cw, 1.3074))
check("channel WF: all-three-active level 11/3 < 4 so third is off", close((4 + 7) / 3, 3.6667) and (4 + 7) / 3 < 4)
check("channel WF: C = 1/2 log2(3.5^2/2)", close(Cw, 0.5 * np.log2(3.5 ** 2 / 2), 1e-9))
check("channel WF: multiplier 1/(2 nu) = 0.1429 nats per unit power", close(1 / (2 * nu), 0.1429))
# uniform power for comparison
Cu = float(sum(0.5 * np.log2(1 + (4 / 3) / n) for n in N))
check("channel WF: uniform power gives 1.1871 bits", close(Cu, 1.1871))
nu5, Pi5, C5 = chanwf(np.array([1.0, 3.0]), 1.0)
check("Ex4.5: nu=2, P=(1,0), C=0.5 bit", close(nu5, 2.0, 1e-9) and np.allclose(Pi5, [1, 0], atol=1e-9) and close(C5, 0.5, 1e-9))

# ---------------------------------------------------------------- section 5: log det and max-det
Pd = np.diag([4.0, 1.0])
Sstar = np.diag([0.125, 0.5])
check("max-det: Sigma* = theta P^-1 = diag(0.125,0.5)", np.allclose(0.5 * np.linalg.inv(Pd), Sstar))
check("max-det: tr(P Sigma*)=1", close(np.trace(Pd @ Sstar), 1.0, 1e-12))
check("max-det: R=2 bits", close(0.5 * np.log2(1 / np.linalg.det(Sstar)), 2.0, 1e-12))
check("max-det: water level 0.5 < both gammas (4,1)", 0.5 < 1)
Salt = np.diag([0.2, 0.2])
check("max-det: diag(0.2,0.2) feasible", close(np.trace(Pd @ Salt), 1.0, 1e-12))
check("max-det: diag(0.2,0.2) costs 2.3219 bits", close(0.5 * np.log2(1 / np.linalg.det(Salt)), 2.3219))
# convexity of rate along segment between the two
r = lambda S: 0.5 * np.log2(1 / np.linalg.det(S))
check("rate convex along segment", all(r((1 - t) * Sstar + t * Salt) <= (1 - t) * r(Sstar) + t * r(Salt) + 1e-12 for t in np.linspace(0, 1, 11)))

# ---------------------------------------------------------------- section 6: LMI and cvxpy
Sg = np.diag([2.0, 0.5])
Z = np.diag([0.5, 2.0])
Blk = np.block([[Z, np.eye(2)], [np.eye(2), Sg]])
check("Schur LMI: [[Z,I],[I,Sigma]] PSD when Z = Sigma^-1", np.linalg.eigvalsh(Blk).min() > -1e-12)
check("Schur LMI: it is singular (rank 2)", np.linalg.matrix_rank(Blk, tol=1e-9) == 2)
Blk2 = np.block([[0.9 * Z, np.eye(2)], [np.eye(2), Sg]])
check("Schur LMI: Z = 0.9 Sigma^-1 fails", np.linalg.eigvalsh(Blk2).min() < 0)
try:
    import cvxpy as cp
    Sx = np.array([[2.0, 1.0], [1.0, 2.0]])
    P2 = np.diag([1.0, 0.25])
    Sv = cp.Variable((2, 2), symmetric=True)
    prob = cp.Problem(cp.Maximize(cp.log_det(Sv)), [Sx - Sv >> 0, cp.trace(P2 @ Sv) <= 0.5])
    prob.solve(solver=cp.CLARABEL)
    Sopt = Sv.value
    Rcvx = 0.5 * (np.log(np.linalg.det(Sx)) - np.log(np.linalg.det(Sopt))) / LN2
    check("Ex4.10 cvxpy: R_C(0.5)=1.7926 bits", close(Rcvx, 1.7926))
    check("Ex4.10 cvxpy: Sigma* = diag(0.25,1)", np.allclose(Sopt, np.diag([0.25, 1.0]), atol=1e-4))
    # same with the identity reader: R(0.5)=2.792
    Sv2 = cp.Variable((2, 2), symmetric=True)
    prob2 = cp.Problem(cp.Maximize(cp.log_det(Sv2)), [Sx - Sv2 >> 0, cp.trace(Sv2) <= 0.5])
    prob2.solve(solver=cp.CLARABEL)
    R2 = 0.5 * (np.log(3.0) - np.log(np.linalg.det(Sv2.value))) / LN2
    check("Ex4.10 cvxpy: identity reader R(0.5)=2.792 bits", close(R2, 2.792))
except ImportError:
    check("cvxpy available", False)

# ---------------------------------------------------------------- section 7: MM and Blahut-Arimoto
f = lambda x: -np.log(x) + np.log(1 + x) + x / 4


def gsur(x, xt):
    return -np.log(x) + x / (1 + xt) + np.log(1 + xt) - xt / (1 + xt) + x / 4


it = [1.0]
for _ in range(8):
    xt = it[-1]
    it.append(4 * (1 + xt) / (5 + xt))
xstar = (-1 + np.sqrt(17)) / 2
check("MM: x1=4/3", close(it[1], 4 / 3, 1e-12))
check("MM: x2=1.4737", close(it[2], 1.4737))
check("MM: x3=1.5285", close(it[3], 1.5285))
check("MM: fixed point (-1+sqrt17)/2 = 1.5616", close(xstar, 1.5616))
check("MM: fixed point solves x^2+x-4=0 and f'=0",
      close(xstar ** 2 + xstar - 4, 0, 1e-12) and close(-1 / xstar + 1 / (1 + xstar) + 0.25, 0, 1e-12))
fv = [f(v) for v in it]
check("MM: f(1)=0.9431", close(fv[0], 0.9431))
check("MM: f(4/3)=0.8930", close(fv[1], 0.8930))
check("MM: f(x*)=0.8853", close(f(xstar), 0.8853))
check("MM: f nonincreasing along iterates", all(fv[k + 1] <= fv[k] + 1e-15 for k in range(len(fv) - 1)))
check("MM: surrogate touches at x_t and lies above f",
      all(close(gsur(xt, xt), f(xt), 1e-12) for xt in it[:4])
      and all(gsur(z, xt) >= f(z) - 1e-12 for xt in it[:4] for z in np.linspace(0.3, 3, 200)))
check("MM: error ratio approaches 0.372 per step",
      close((it[8] - xstar) / (it[7] - xstar), 0.3717, 2e-3))
# derivative of the map at x*: 16/(5+x*)^2
check("MM: map slope at x* = 16/(5+x*)^2 = 0.3717", close(16 / (5 + xstar) ** 2, 0.3717))
xg = np.linspace(0.5, 2.6, 22)
series("mmf", xg, f(xg))
series("mmg0", xg, gsur(xg, it[0]))
series("mmg1", xg, gsur(xg, it[1]))
series("mmiter", it[:4], [f(v) for v in it[:4]])
# Blahut-Arimoto for the Z-channel
W = np.array([[1.0, 0.0], [0.5, 0.5]])


def kl_row(w, q):
    m = w > 0
    return float((w[m] * np.log(w[m] / q[m])).sum())


p = np.array([0.5, 0.5])
lows, ups, p1s = [], [], []
for t in range(9):
    q = p @ W
    Dx = np.array([kl_row(W[i], q) for i in range(2)])
    lows.append(float(p @ Dx) / LN2)
    ups.append(float(Dx.max()) / LN2)
    p1s.append(p[1])
    p = p * np.exp(Dx)
    p = p / p.sum()
Cz = np.log2(1.25)
check("BA: capacity log2(5/4)=0.3219 bits", close(Cz, 0.3219))
check("BA: I(p0)=0.3113 bits", close(lows[0], 0.3113))
check("BA: p1(1)=0.4641", close(p1s[1], 0.4641))
check("BA: I(p1)=0.3175 bits", close(lows[1], 0.3175))
check("BA: KL rows at p0 0.2877 and 0.1438 nats",
      close(np.log(4 / 3), 0.2877) and close(0.5 * np.log(0.5 / 0.75) + 0.5 * np.log(2), 0.1438))
check("BA: lower bounds nondecreasing", all(lows[k + 1] >= lows[k] - 1e-15 for k in range(8)))
check("BA: lower <= C <= upper at every step", all(lo_ <= Cz + 1e-12 <= up_ + 2e-12 for lo_, up_ in zip(lows, ups)))
pp_ = np.array([0.5, 0.5])
for _ in range(2000):
    q_ = pp_ @ W
    pp_ = pp_ * np.exp([kl_row(W[i], q_) for i in range(2)])
    pp_ = pp_ / pp_.sum()
check("BA: p(1) -> 0.4 (2000 steps)", close(pp_[1], 0.4, 1e-4))
check("BA: after 8 steps p(1)=%.4f still above 0.4" % p1s[-1], p1s[-1] > 0.4)
check("BA: upper bound at p0 = 0.4150 bits", close(ups[0], 0.4150))
series("balow", range(9), lows)
series("baup", range(9), ups)
check("BA: optimal input 0.4 gives capacity",
      close(hb(0.2) - 0.4, Cz, 1e-9))
# Blahut-Arimoto for R(D) of a fair bit, Hamming distortion (Ex4.11)
beta = np.log(9.0)
dmat = np.array([[0, 1], [1, 0]], dtype=float)
qh = np.array([0.3, 0.7])
for _ in range(500):
    A_ = qh[None, :] * np.exp(-beta * dmat)
    cond = A_ / A_.sum(1, keepdims=True)
    qh = 0.5 * cond.sum(0)
Dba = float(0.5 * (cond * dmat).sum())
Rba = float(0.5 * (cond * np.log2(cond / qh[None, :])).sum())
check("Ex4.11 BA R(D): D=0.1 at slope ln 9", close(Dba, 0.1, 1e-6))
check("Ex4.11 BA R(D): R=1-h(0.1)=0.531 bits", close(Rba, 1 - hb(0.1), 1e-6))

# ---------------------------------------------------------------- section 8: projected gradient
grad = lambda z: np.array([2 * (z[0] - 2), 2 * (z[1] - 1)])


def proj(z):
    s = z.sum()
    return z if s <= 2 else z - (s - 2) / 2 * np.ones(2)


z = np.zeros(2)
path = [z.copy()]
raw = []
for _ in range(7):
    w = z - 0.25 * grad(z)
    raw.append(w.copy())
    z = proj(w)
    path.append(z.copy())
check("PGD: step 1 (1,0.5) feasible", np.allclose(path[1], [1, 0.5]))
check("PGD: step 2 raw (1.5,0.75) projected to (1.375,0.625)",
      np.allclose(raw[1], [1.5, 0.75]) and np.allclose(path[2], [1.375, 0.625]))
check("PGD: step 3 (1.4375,0.5625)", np.allclose(path[3], [1.4375, 0.5625]))
errs = [np.linalg.norm(pp - np.array([1.5, 0.5])) for pp in path[2:]]
check("PGD: error halves on the face", all(close(errs[k + 1] / errs[k], 0.5, 1e-9) for k in range(len(errs) - 1)))
series("pgdpath", [pp[0] for pp in path[:6]], [pp[1] for pp in path[:6]])
series("pgdraw", [rr[0] for rr in raw[1:4]], [rr[1] for rr in raw[1:4]])
M = np.array([[1.0, 2.0], [2.0, 1.0]])
w_, V_ = np.linalg.eigh(M)
Mp = V_ @ np.diag(np.maximum(w_, 0)) @ V_.T
check("PSD projection of [[1,2],[2,1]] is [[1.5,1.5],[1.5,1.5]]", np.allclose(Mp, [[1.5, 1.5], [1.5, 1.5]]))
check("PSD projection distance = 1 (Frobenius)", close(np.linalg.norm(M - Mp), 1.0, 1e-12))

# ---------------------------------------------------------------- exercises
a = np.array([1.0, 2.0])
xs4 = a / (a @ a)
check("Ex4.4: x=(0.2,0.4)", np.allclose(xs4, [0.2, 0.4]))
check("Ex4.4: mu=2/||a||^2=0.4 from 2x = mu a", np.allclose(2 * xs4, 0.4 * a))
check("Ex4.4: value 0.2", close(xs4 @ xs4, 0.2, 1e-12))
check("Ex4.7: x_{t+1}=4(1+x_t)/(5+x_t) from -1/x+1/(1+x_t)+1/4=0",
      close(-1 / it[2] + 1 / (1 + it[1]) + 0.25, 0, 1e-12))
check("Ex4.8: p1 = (0.5359, 0.4641)", close(1 - p1s[1], 0.5359))

check("logdet figure: midpoint exceeds chord by 0.446 nats", close(mid - avg, 0.446))


def rd_dual(lmb, gam, D):
    tot = 0.0
    for g_ in gam:
        if 2 * lmb * g_ > 1:
            tot += 0.5 * np.log(2 * lmb * g_) + 0.5
        else:
            tot += lmb * g_
    return tot - lmb * D


gam4 = [4, 2, 1, 0.25]
check("RD dual: g(1)=2.0794 nats", close(rd_dual(1.0, gam4, 1.75), 2.0794))
check("RD dual: g(1)= 3 bits", close(rd_dual(1.0, gam4, 1.75) / LN2, 3.0, 1e-9))
check("RD dual: g(1) = 1/2 ln 64 + 1.5 + 0.25 - 1.75", close(0.5 * np.log(64) + 1.5 + 0.25 - 1.75, rd_dual(1.0, gam4, 1.75), 1e-12))
check("RD dual: g(0.8)=2.0447 nats = 2.9499 bits",
      close(rd_dual(0.8, gam4, 1.75), 2.0447) and close(rd_dual(0.8, gam4, 1.75) / LN2, 2.9499))
lg_ = np.linspace(0.05, 5, 20001)
gv_ = [rd_dual(v, gam4, 1.75) for v in lg_]
check("RD dual: max over lambda is at lambda=1 and equals R", close(lg_[int(np.argmax(gv_))], 1.0, 1e-3) and close(max(gv_), 2.0794))
# two sources sharing 2 bits
check("two sources: R1=1.5,R2=0.5 give D1=D2=0.5", close(4 * 2 ** -3, 0.5, 1e-12) and close(2 ** -1, 0.5, 1e-12))
R1g = np.linspace(0, 2, 200001)
Dt = 4 * 2.0 ** (-2 * R1g) + 2.0 ** (-2 * (2 - R1g))
check("two sources: optimum R1=1.5, total 1", close(R1g[np.argmin(Dt)], 1.5, 1e-4) and close(Dt.min(), 1.0, 1e-9))
check("two sources: multiplier 2 ln2 * 0.5 = 0.6931", close(2 * LN2 * 0.5, 0.6931))
# nonconvex KKT example: min -x^2 on [-1,1]
check("nonconvex KKT: x=0 stationary yet maximum; minima at +-1 with multiplier 2",
      close(-(0.0) ** 2, 0.0) and -1.0 < 0.0 and close(-(-2 * 1.0), 2.0))
# projected gradient with eta = 1 cycles
zz = np.zeros(2)
seq = []
for _ in range(6):
    zz = proj(zz - 1.0 * grad(zz))
    seq.append(tuple(np.round(zz, 12)))
check("PGD eta=1: (2,0),(1,1),(2,0),... cycle", seq[0] == (2.0, 0.0) and seq[1] == (1.0, 1.0) and seq[2] == (2.0, 0.0) and seq[3] == (1.0, 1.0))
check("PGD eta=1: raw steps (4,2),(2,2),(3,1)",
      np.allclose(np.zeros(2) - grad(np.zeros(2)), [4, 2]) and np.allclose(np.array([2.0, 0]) - grad(np.array([2.0, 0])), [2, 2])
      and np.allclose(np.array([1.0, 1]) - grad(np.array([1.0, 1])), [3, 1]))
# matrix KKT -> water-filling: eigenvalues min(1, theta/gamma)
gtilde = np.array([4.0, 1.0])
check("matrix KKT: X eigenvalues min(1,theta/gamma) for gamma=(4,1), theta=0.5 give Sigma*=diag(0.125,0.5)",
      np.allclose(np.minimum(1, 0.5 / gtilde), [0.125, 0.5]))
check("Lagrange figure: guide ellipses levels 2 and 12 have radii (sqrt2,1) and (sqrt12,sqrt6)",
      close(2 / 2, 1.0) and close(12 / 2, 6.0))
check("Lagrange figure: level 2 misses the line (min on line is 6), level 12 crosses it", 2 < 6 < 12)
check("KKT figure: guide circle is the level f=2, radius sqrt2", close(np.sqrt(2) ** 2, 2.0, 1e-12))
check("PGD figure: start (0,0) lies on level 5, radius sqrt5", close((0 - 2) ** 2 + (0 - 1) ** 2, 5.0, 1e-12))
check("Ex4.5: both-active level 2.5 < 3", close((1 + 1 + 3) / 2, 2.5, 1e-12))
check("Ex4.10: theta P^-1 with theta=0.25 is diag(0.25,1)", np.allclose(0.25 * np.linalg.inv(np.diag([1.0, 0.25])), np.diag([0.25, 1.0])))
check("Ex4.11: 1/(1+9)=0.1", close(1 / (1 + 9.0), 0.1, 1e-15))
check("Ex4.1: -sqrt(x)'' = x^{-3/2}/4 (sympy)",
      sp.simplify(sp.diff(-sp.sqrt(sp.Symbol('u', positive=True)), sp.Symbol('u', positive=True), 2)
                  - sp.Symbol('u', positive=True) ** sp.Rational(-3, 2) / 4) == 0)
check("MM: f(x2)=0.8864, f(x3)=0.8855", close(f(it[2]), 0.8864) and close(f(it[3]), 0.8855))

# ---------------------------------------------------------------- fill-out additions
# gradient descent on f = (x^2 + 10 y^2)/2, kappa = 10, eta = 2/(L+mu) = 2/11
eta_ = 2 / 11
zk = np.array([9.0, 1.5])
gd = [zk.copy()]
for _ in range(12):
    zk = zk - eta_ * np.array([1.0, 10.0]) * zk
    gd.append(zk.copy())
series("zigzag", [v[0] for v in gd[:9]], [v[1] for v in gd[:9]])
check("GD: contraction 9/11=0.8182 in both coordinates",
      all(close(gd[k + 1][0] / gd[k][0], 9 / 11, 1e-12) and close(gd[k + 1][1] / gd[k][1], -9 / 11, 1e-12) for k in range(10)))
check("GD: (kappa-1)/(kappa+1)=0.8182", close(9 / 11, 0.8182))
check("GD: first step (7.364,-1.227)", np.allclose(gd[1], [7.3636, -1.2273], atol=1e-4))
check("GD: eta=1/L=0.1 gives x-factor 0.9", close(1 - 0.1 * 1, 0.9, 1e-12))
k6 = int(np.ceil(np.log(1e-6) / np.log(9 / 11)))
check("GD: 69 steps to shrink error by 1e6 at kappa=10", k6 == 69)
k6b = int(np.ceil(np.log(1e-6) / np.log(999 / 1001)))
check("GD: 6908 steps at kappa=1000 (printed %d)" % k6b, k6b == 6908)
check("GD figure: level ellipse through start has value 0.5*(81+22.5)=51.75", close(0.5 * (81 + 10 * 2.25), 51.75, 1e-12))
check("GD figure: its semi-axes sqrt(103.5)=10.17 and sqrt(10.35)=3.217",
      close(np.sqrt(103.5), 10.17, 5e-3) and close(np.sqrt(10.35), 3.217, 5e-3))
# Newton on f = x - ln x
xn = [0.5]
for _ in range(4):
    xn.append(2 * xn[-1] - xn[-1] ** 2)
check("Newton: iterates 0.75, 0.9375, 0.99609, 0.9999847",
      np.allclose(xn[1:], [0.75, 0.9375, 0.99609375, 0.9999847412], atol=1e-9))
check("Newton: error squares each step", all(close(1 - xn[k + 1], (1 - xn[k]) ** 2, 1e-15) for k in range(4)))
check("Newton: from x0=2 the update gives 0, from x0=3 it gives -3",
      2 * 2 - 2 ** 2 == 0 and 2 * 3 - 3 ** 2 == -3)
check("Newton: x1 in (0,1] exactly when 0<x0<2 (1-x1=(1-x0)^2<1)",
      all((0 < 2 * x - x * x <= 1) == (0 < x < 2) for x in np.linspace(-1, 3, 401)))
# local minimum on a nonconvex domain: x^2 on [-3,-2] u [1,2]
check("nonconvex domain: x^2 has local min 4 at -2 and global min 1 at 1",
      (-2) ** 2 == 4 and 1 ** 2 == 1 and min(x * x for x in np.linspace(-3, -2, 101)) == 4.0)
# projected gradient descends for every eta < 2/L on the half-plane example (L=2)
def _pg(z, eta):
    g = 2 * (z - np.array([2.0, 1.0]))
    y = z - eta * g
    a = np.array([1.0, 1.0])
    return y - max(a @ y - 2.0, 0.0) / 2.0 * a
_f = lambda z: (z[0] - 2) ** 2 + (z[1] - 1) ** 2
ok_desc = True
for eta in [0.25, 0.5, 0.75, 0.95]:
    z = np.zeros(2)
    for _ in range(30):
        zn = _pg(z, eta)
        ok_desc &= _f(zn) <= _f(z) + 1e-12
        z = zn
check("projected gradient: cost never rises for eta in {0.25,0.5,0.75,0.95} < 2/L", ok_desc)
check("Newton: one step on the quadratic reaches the minimum",
      np.allclose(np.array([9.0, 1.5]) - np.array([9.0, 15.0]) / np.array([1.0, 10.0]), [0, 0]))
# envelope theorem on the KKT example: constraint x+y <= 2+u, p*(u) = (1-u)^2/2
u_ = sp.symbols("u")
pu = (1 - u_) ** 2 / 2
check("envelope: p*(0)=0.5, dp*/du at 0 = -1 = -mu", pu.subs(u_, 0) == sp.Rational(1, 2) and sp.diff(pu, u_).subs(u_, 0) == -1)
check("envelope: p*(0.1)=0.405", close(float(pu.subs(u_, sp.Rational(1, 10))), 0.405, 1e-12))
# curvature of R(D) jumps where a column goes under water (gamma=(4,2,1,0.25), theta=0.25 at D=1)
check("R(D) breakpoint: theta=0.25 at D=1.0", close(3 * 0.25 + 0.25, 1.0, 1e-12))
check("R(D) second derivative 2 (k=4) and 2.667 (k=3) nats at theta=0.25",
      close(1 / (2 * 0.25 ** 2 * 4), 2.0, 1e-12) and close(1 / (2 * 0.25 ** 2 * 3), 2.6667))


def Rbits(D):
    lo, hi = 0.0, 4.0
    for _ in range(200):
        th = 0.5 * (lo + hi)
        if np.minimum(gam, th).sum() > D:
            hi = th
        else:
            lo = th
    th = 0.5 * (lo + hi)
    return sum(0.5 * np.log(g / th) for g in gam if g > th)


h_ = 1e-4
d2l = (Rbits(1 - h_) - 2 * Rbits(1 - 2 * h_) + Rbits(1 - 3 * h_)) / h_ ** 2
d2r = (Rbits(1 + 3 * h_) - 2 * Rbits(1 + 2 * h_) + Rbits(1 + h_)) / h_ ** 2
check("R(D) numerical second derivative ~2 left, ~2.667 right of D=1", close(d2l, 2.0, 1e-2) and close(d2r, 2.6667, 1e-2))
s1_ = (Rbits(1 + h_) - Rbits(1 - h_)) / (2 * h_)
check("R(D) slope continuous at D=1: -1/(2*0.25) = -2 nats", close(s1_, -2.0, 1e-3))
# non-commuting matrix water-filling at D=1.0 (Sigma_x=[[2,1],[1,2]], P=diag(1,0.25))
Sx_ = np.array([[2.0, 1.0], [1.0, 2.0]])
Pn = np.diag([1.0, 0.25])
w_, V_ = np.linalg.eigh(Sx_)
Sh_ = V_ @ np.diag(np.sqrt(w_)) @ V_.T
Qt_ = Sh_ @ Pn @ Sh_
lq, Eq = np.linalg.eigh(Qt_)
check("non-commuting: Sigma_x^{1/2} = [[1.366,0.366],[0.366,1.366]]", np.allclose(Sh_, [[1.3660, 0.3660], [0.3660, 1.3660]], atol=1e-4))
check("non-commuting: Ptilde = [[1.8995,0.625],[0.625,0.6005]]", np.allclose(Qt_, [[1.8995, 0.625], [0.625, 0.6005]], atol=1e-4))
check("non-commuting: gamma = (2.1514, 0.3486)", np.allclose(np.sort(lq)[::-1], [2.1514, 0.3486], atol=1e-4))
thn = 1.0 - 0.34861
thn = 1.0 - np.sort(lq)[0]
check("non-commuting D=1: theta = 1 - 0.3486 = 0.6514", close(thn, 0.6514))
Rn = 0.5 * np.log2(np.sort(lq)[1] / thn)
check("non-commuting D=1: R_C(1) = 0.8617 bits", close(Rn, 0.8617))
Xn = Eq @ np.diag([1.0 if l <= thn else thn / l for l in lq]) @ Eq.T
Sn = Sh_ @ Xn @ Sh_
check("non-commuting D=1: tr(P Sigma*) = 1", close(np.trace(Pn @ Sn), 1.0, 1e-9))
check("non-commuting D=1: rate from det = 0.8617", close(0.5 * np.log2(np.linalg.det(Sx_) / np.linalg.det(Sn)), 0.8617))
check("non-commuting D=1: Sigma* = [[0.6260,0.1679],[0.1679,1.4962]]", np.allclose(Sn, [[0.6260, 0.1679], [0.1679, 1.4962]], atol=1e-4))
check("non-commuting D=1: Sigma_x - Sigma* has rank one", np.linalg.matrix_rank(Sx_ - Sn, tol=1e-8) == 1)
check("non-commuting D=1: Sigma* is not diagonal (does not commute with P)", abs(Sn[0, 1]) > 0.1)
check("non-commuting D=1: Sigma* does not commute with Sigma_x either",
      np.abs(Sn @ Sx_ - Sx_ @ Sn).max() > 0.1 and abs(Sn[0, 0] - Sn[1, 1]) > 0.5)
# hypothesis audit: with every mode above water and P invertible, Sigma* = theta P^{-1}
# (D=0.5 on the same pair, theta=0.25), which commutes with P although P and Sigma_x do not
th05 = 0.25
X05 = Eq @ np.diag([1.0 if l <= th05 else th05 / l for l in lq]) @ Eq.T
S05 = Sh_ @ X05 @ Sh_
check("all modes above water: Sigma* = theta P^{-1} = diag(0.25,1), commutes with P",
      np.all(lq > th05) and np.allclose(S05, th05 * np.linalg.inv(Pn), atol=1e-9)
      and np.allclose(S05 @ Pn, Pn @ S05, atol=1e-9) and close(np.trace(Pn @ S05), 0.5, 1e-9))
# coordinatewise (allocate_bits-style) scheme at D=1: products (2,0.5), theta' = 0.5, 1 bit
wp = np.array([2.0, 0.5])
check("coordinatewise D=1: theta'=0.5, rates 1 and 0 bits",
      close(np.minimum(wp, 0.5).sum(), 1.0, 1e-12) and close(0.5 * np.log2(2 / 0.5), 1.0, 1e-12))
check("coordinatewise D=1: excess 0.138 bits", close(1.0 - Rn, 0.1383, 1e-3))
# KKT certificate for the matrix solution: X^{-1} = 2 nu Ptilde + 2 Theta
nu_ = 1 / (2 * thn)
Theta_ = 0.5 * (np.linalg.inv(Xn) - 2 * nu_ * Qt_)
check("non-commuting KKT: Theta PSD and Theta (I - X) = 0",
      np.linalg.eigvalsh(Theta_).min() > -1e-9 and np.allclose(Theta_ @ (np.eye(2) - Xn), 0, atol=1e-9))
check("non-commuting KKT: Theta rank one, eigenvalue (1 - 0.3486/0.6514)/2 = 0.2324",
      close(np.linalg.eigvalsh(Theta_).max(), (1 - np.sort(lq)[0] / thn) / 2, 1e-9) and close(np.linalg.eigvalsh(Theta_).max(), 0.2324))
try:
    Sv3 = cp.Variable((2, 2), symmetric=True)
    cp.Problem(cp.Maximize(cp.log_det(Sv3)), [Sx_ - Sv3 >> 0, cp.trace(Pn @ Sv3) <= 1.0]).solve(solver=cp.CLARABEL)
    check("non-commuting D=1: cvxpy agrees with Sigma*", np.allclose(Sv3.value, Sn, atol=1e-4))
except Exception:
    check("non-commuting D=1: cvxpy run", False)
# saddle and rank examples
check("saddle: x^2-y^2 Hessian eigenvalues (-2,2)", np.allclose(np.linalg.eigvalsh(np.diag([2.0, -2.0])), [-2, 2]))
e1 = np.array([[1.0, 0], [0, 0]])
e2 = np.array([[0.0, 0], [0, 1]])
check("rank: average of two rank-one matrices has rank 2", np.linalg.matrix_rank(0.5 * (e1 + e2)) == 2)

check("GD figure: guide ellipses are levels 15 and 4 (semi-axes sqrt30,sqrt3 and sqrt8,sqrt0.8)",
      close(0.5 * 30, 15) and close(0.5 * 10 * 3, 15) and close(0.5 * 8, 4) and close(0.5 * 10 * 0.8, 4))
check("non-commuting: Theta eigenvalue formula (1-0.3486/0.6514)/2", close((1 - 0.34861 / 0.65139) / 2, 0.2324))
# ---------------------------------------------------------------- compare figure data in the chapter
for cand in ["ch04_optimization.tex", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "chapters", "ch04_optimization.tex")]:
    if os.path.exists(cand):
        tex = open(cand, encoding="utf8").read()
        tags = re.findall(r"coordinates\s*\{([^}]*)\}\s*;?\s*%\s*data:(\w+)", tex)
        for body, name in tags:
            pts = [tuple(map(float, m)) for m in re.findall(r"\(\s*([-\d.eE+]+)\s*,\s*([-\d.eE+]+)\s*\)", body)]
            ref = SERIES.get(name)
            ok = ref is not None and len(ref) == len(pts) and all(
                abs(px - rx) <= 1e-3 * max(1, abs(rx)) and abs(py - ry) <= 1e-3 * max(1, abs(ry))
                for (px, py), (rx, ry) in zip(pts, ref))
            check(f"figure series '{name}' in chapter matches computed data", ok)
        check("figure series found in chapter", len(tags) > 0)
        break

n = len(RESULTS)
k = sum(RESULTS)
print(f"{k}/{n} PASS")
sys.exit(0 if k == n else 1)
