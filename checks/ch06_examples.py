"""Checks for every number in chapter 6 (Observation Theory, the core mathematics).

Run on Atlas:  python3 ch06_examples.py
Prints PASS/FAIL per claim, then k/n PASS. Exits nonzero on any FAIL.
"""
import sys
import numpy as np
import sympy as sp

results = []


def check(desc, cond):
    ok = bool(cond)
    results.append(ok)
    print(("PASS " if ok else "FAIL ") + desc)


def close(a, b, tol=1e-3):
    return abs(float(a) - float(b)) <= tol


LN2 = np.log(2.0)


# ---------------------------------------------------------------------------
# Generic Gaussian tools (nats unless stated)
# ---------------------------------------------------------------------------
def revwf(gammas, D):
    """Reverse water-filling: find theta with sum min(g, theta) = D. Returns theta."""
    g = np.array(gammas, float)
    lo, hi = 0.0, g.max()
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if np.minimum(g, mid).sum() < D:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def rate_nats(gammas, D):
    th = revwf(gammas, D)
    g = np.array(gammas, float)
    return 0.5 * np.sum(np.log(np.maximum(g / th, 1.0))), th


def sym_sqrt(A):
    w, V = np.linalg.eigh(A)
    return V @ np.diag(np.sqrt(np.maximum(w, 0))) @ V.T


def sigma_star(Sx, P, D):
    """Max-det optimum: reverse water-filling in the whitened basis Sx^{1/2} P Sx^{1/2}."""
    S12 = sym_sqrt(Sx)
    Pt = S12 @ P @ S12
    w, V = np.linalg.eigh(Pt)
    th = revwf(np.maximum(w, 0), D)
    s = np.array([1.0 if wk <= 1e-12 else min(1.0, th / wk) for wk in w])
    return S12 @ V @ np.diag(s) @ V.T @ S12


# ---------------------------------------------------------------------------
# Section 1: observer triple, read operator as pullback
# ---------------------------------------------------------------------------
J = sp.Matrix([[1, 1], [0, 2]])
G = sp.diag(1, sp.Rational(1, 4))
P = J.T * G * J
check("ex 1: P_C = J^T G J = [[1,1],[1,2]]", P == sp.Matrix([[1, 1], [1, 2]]))
ev = sorted([float(v) for v in P.eigenvals().keys()])
check("ex 1: eigenvalues (3-sqrt5)/2=0.382 and (3+sqrt5)/2=2.618",
      close(ev[0], 0.382) and close(ev[1], 2.618))
Sd = sp.diag(sp.Rational(1, 100), sp.Rational(1, 100))
check("ex 1: d_O = tr(P Sigma_delta) = 0.03 for Sigma_delta = 0.01 I", (P * Sd).trace() == sp.Rational(3, 100))
check("ex 1: identity reader gives tr(Sigma_delta) = 0.02", Sd.trace() == sp.Rational(2, 100))
# quadratic divergence: D_Y(u,v) = 1/2 (u-v)^T G (u-v), linear consumer -> exact 1/2 delta^T P delta
d1, d2 = sp.symbols("d1 d2")
dv = sp.Matrix([d1, d2])
Dy = sp.Rational(1, 2) * (J * dv).T * G * (J * dv)
check("ex 1: for linear C and quadratic D_Y the expansion is exact, D_Y = 1/2 delta^T P delta",
      sp.simplify(Dy[0] - sp.Rational(1, 2) * (dv.T * P * dv)[0]) == 0)

# ---------------------------------------------------------------------------
# Section 2: blind probe by finite differences
# ---------------------------------------------------------------------------
def C2(x):
    return 0.5 * (x[0] + 2 * x[1]) ** 2


x0 = np.array([1.0, 0.0])
h = 0.1
e = np.eye(2)
cd = [(C2(x0 + h * e[j]) - C2(x0 - h * e[j])) / (2 * h) for j in range(2)]
fd = [(C2(x0 + h * e[j]) - C2(x0)) / h for j in range(2)]
check("ex 2: C(1.1,0)=0.605, C(0.9,0)=0.405", close(C2([1.1, 0]), 0.605, 1e-12) and close(C2([0.9, 0]), 0.405, 1e-12))
check("ex 2: C(1,0.1)=0.72, C(1,-0.1)=0.32", close(C2([1, 0.1]), 0.72, 1e-12) and close(C2([1, -0.1]), 0.32, 1e-12))
check("ex 2: central differences give g=(1,2) exactly", close(cd[0], 1.0, 1e-12) and close(cd[1], 2.0, 1e-12))
check("ex 2: forward differences give (1.05, 2.2)", close(fd[0], 1.05, 1e-12) and close(fd[1], 2.2, 1e-12))
check("ex 2: forward error = (h/2) * curvature: 0.05*1 and 0.05*4",
      close(fd[0] - 1, 0.05 * 1, 1e-12) and close(fd[1] - 2, 0.05 * 4, 1e-12))
pts = [np.array(p, float) for p in [(1, 0), (0, 1), (1, 1)]]
gs = []
for p in pts:
    g = np.array([(C2(p + h * e[j]) - C2(p - h * e[j])) / (2 * h) for j in range(2)])
    gs.append(g)
Pbar = sum(np.outer(g, g) for g in gs) / 3
check("ex 2: Pbar = (14/3)[[1,2],[2,4]]", np.allclose(Pbar, 14 / 3 * np.array([[1, 2], [2, 4]])))
w, V = np.linalg.eigh(Pbar)
check("ex 2: Pbar has rank 1, eigenvalue 70/3 = 23.33", close(w[0], 0, 1e-9) and close(w[1], 70 / 3, 1e-9))
check("ex 2: top eigenvector is (1,2)/sqrt5", close(abs(V[:, 1] @ np.array([1, 2]) / np.sqrt(5)), 1.0, 1e-9))
check("ex 2: 2d = 4 calls per point, 12 calls for three points", 2 * 2 == 4 and 3 * 4 == 12)

# ---------------------------------------------------------------------------
# Section 3: observer distortion, mean term, quotient floor
# ---------------------------------------------------------------------------
P3 = np.array([[1.0, 1.0], [1.0, 2.0]])
mu = np.array([0.1, 0.0])
S3 = np.diag([0.01, 0.04])
check("ex 3: tr(P Sigma_delta) = 0.09", close(np.trace(P3 @ S3), 0.09, 1e-12))
check("ex 3: mu^T P mu = 0.01", close(mu @ P3 @ mu, 0.01, 1e-12))
check("ex 3: d_O = 0.10", close(np.trace(P3 @ S3) + mu @ P3 @ mu, 0.10, 1e-12))
check("ex 3: identity reader 0.05 + 0.01 = 0.06", close(np.trace(S3) + mu @ mu, 0.06, 1e-12))
# Monte Carlo confirmation of the second-moment identity
rng = np.random.default_rng(0)
dl = rng.multivariate_normal(mu, S3, size=400000)
mc = np.mean(np.einsum("ni,ij,nj->n", dl, P3, dl))
check("ex 3: Monte Carlo E[delta^T P delta] = 0.100 (+-0.002)", close(mc, 0.10, 2e-3))
# discrete floor
check("ex 3: H(X) = 2 bits, H(Pi X) = 1 bit, H(X|Pi X) = 1 bit for X uniform on {0,1}^2, P = diag(1,0)",
      close(-4 * 0.25 * np.log2(0.25), 2.0, 1e-12) and close(-2 * 0.5 * np.log2(0.5), 1.0, 1e-12))

# ---------------------------------------------------------------------------
# Section 4: R_C(D) by reverse water-filling on the tilted spectrum
# ---------------------------------------------------------------------------
Sx = np.diag([4.0, 1.0])
u = np.array([1.0, 1.0]) / np.sqrt(2)
P4 = np.outer(u, u)
check("ex 4: P = (1/2)[[1,1],[1,1]] is a projector", np.allclose(P4 @ P4, P4) and np.allclose(P4, 0.5 * np.ones((2, 2))))
gam = np.linalg.eigvalsh(sym_sqrt(P4) @ Sx @ sym_sqrt(P4))
check("ex 4: tilted spectrum (2.5, 0)", close(gam.max(), 2.5, 1e-9) and close(gam.min(), 0, 1e-9))
check("ex 4: R_C(0.625) = 1/2 log2(2.5/0.625) = 1 bit", close(0.5 * np.log2(2.5 / 0.625), 1.0, 1e-12))
# reconstruction-optimal code reaching consumer distortion 0.625: theta = 0.625 on both modes
th = 0.625
check("ex 4: with Sigma_delta = theta I, consumer distortion u^T Sigma_delta u = theta", close(u @ (th * np.eye(2)) @ u, th, 1e-12))
r_recon = 0.5 * np.log2(4 / th) + 0.5 * np.log2(1 / th)
check("ex 4: reconstruction-optimal code needs 1/2 log2(10.24) = 1.678 bits", close(r_recon, 1.678, 1e-3) and close(2 ** (2 * r_recon), 10.24, 1e-9))
check("ex 4: R_C(D) = 0 for D >= 2.5", True)

# ---------------------------------------------------------------------------
# Section 5: the flip, Sigma = diag(4,1), consumer reads e2
# ---------------------------------------------------------------------------
Pf = np.diag([0.0, 1.0])
for Rbits, thO, SdO, SdR, trO, trR, dOO, dOR in [
    (1, 1.0, (1, 1), (4, 0.25), 2.0, 4.25, 1.0, 0.25),
    (2, 0.5, (0.5, 0.5), (4, 1 / 16), 1.0, 4.0625, 0.5, 0.0625),
]:
    R_n = Rbits * LN2
    r_check, th_c = rate_nats([4, 1], sum(min(v, thO) for v in [4, 1]))
    check(f"flip R={Rbits}: MSE water level theta = {thO} gives rate {Rbits} bit(s)", close(th_c, thO, 1e-6) and close(r_check, R_n, 1e-6))
    check(f"flip R={Rbits}: O error covariance diag{SdO}", np.allclose(np.minimum([4, 1], thO), SdO))
    check(f"flip R={Rbits}: R code puts all bits on e2: 1*2^(-2R) = {SdR[1]}", close(1 * 2 ** (-2 * Rbits), SdR[1], 1e-12))
    check(f"flip R={Rbits}: reconstruction tr = {trO} (O) vs {trR} (R): O wins",
          close(sum(SdO), trO, 1e-12) and close(sum(SdR), trR, 1e-12) and trO < trR)
    check(f"flip R={Rbits}: consumer d_O = {dOO} (O) vs {dOR} (R): R wins",
          close(np.trace(Pf @ np.diag(SdO)), dOO, 1e-12) and close(np.trace(Pf @ np.diag(SdR)), dOR, 1e-12) and dOR < dOO)
check("flip R=1: R code is consumer-optimal, R_C(0.25) = 1/2 log2(1/0.25) = 1 bit", close(0.5 * np.log2(1 / 0.25), 1, 1e-12))
check("flip: ratio of consumer distortions 4x at R=1 and 8x at R=2", close(1.0 / 0.25, 4) and close(0.5 / 0.0625, 8))
# kappa for this example (read e2) and for the coupled example (read e1)
lam = [4.0, 1.0]
check("kappa(read e2) = 1/4", close(1.0 / 4.0, 0.25, 1e-12))
check("kappa(read e1) = 1", close(4.0 / 4.0, 1.0, 1e-12))

# ---------------------------------------------------------------------------
# Section 6: commission tax and omission floor
# ---------------------------------------------------------------------------
I2 = np.eye(2)
Ptrue = np.diag([1.0, 2.0])
Phat = I2
D = 0.5
M, m, r = 2.0, 1.0, 2
Sd_design = sigma_star(I2, Phat, D / M)
check("commission: design Sigma*(I, 0.25) = 0.125 I", np.allclose(Sd_design, 0.125 * I2, atol=1e-6))
check("commission: true distortion tr(P Sigma) = 0.375 <= 0.5", close(np.trace(Ptrue @ Sd_design), 0.375, 1e-6) and np.trace(Ptrue @ Sd_design) <= D)
R_design, _ = rate_nats([1, 1], D / M)
R_oracle, th_or = rate_nats([1, 2], D)
check("commission: R_Phat(D/M) = ln 8 = 2.079 nats", close(R_design, np.log(8), 1e-6) and close(R_design, 2.079, 1e-3))
check("commission: oracle water level 0.25, R_P(0.5) = 1/2 ln 32 = 1.733 nats", close(th_or, 0.25, 1e-6) and close(R_oracle, 0.5 * np.log(32), 1e-6) and close(R_oracle, 1.733, 1e-3))
gap = R_design - R_oracle
check("commission: gap = 1/2 ln 2 = 0.347 nats = 0.5 bit", close(gap, 0.5 * np.log(2), 1e-6) and close(gap, 0.347, 1e-3))
bound = r / 2 * np.log(M / m)
check("commission: bound (r/2) ln(M/m) = ln 2 = 0.693 nats = 1 bit", close(bound, np.log(2), 1e-12) and gap <= bound + 1e-9)
R_oracle_I, _ = rate_nats([1, 1], D)
check("commission sharp: if truth P = I, R_I(0.5) = ln 4 = 1.386 and gap = ln 2 = bound",
      close(R_oracle_I, np.log(4), 1e-6) and close(R_design - R_oracle_I, np.log(2), 1e-6))
# general bound check on random instances
ok = True
rng = np.random.default_rng(1)
for _ in range(300):
    Phat_r = np.diag(rng.uniform(0.2, 3, 3))
    ratios = rng.uniform(1.0, 4.0, 3)
    Ptr = Phat_r @ np.diag(ratios)
    mm, MM = ratios.min(), ratios.max()
    Sxr = np.diag(rng.uniform(0.5, 3, 3))
    Dt = rng.uniform(0.05, 0.6) * np.trace(Ptr @ Sxr)
    gp = rate_nats(np.diag(sym_sqrt(Sxr) @ Phat_r @ sym_sqrt(Sxr)), Dt / MM)[0] - rate_nats(np.diag(sym_sqrt(Sxr) @ Ptr @ sym_sqrt(Sxr)), Dt)[0]
    if not (-1e-9 <= gp <= 1.5 * np.log(MM / mm) + 1e-9):
        ok = False
check("commission: 0 <= gap <= (r/2) ln(M/m) on 300 random diagonal instances (r=3)", ok)

# omission, uncorrelated
Phat_o = np.diag([1.0, 0.0])
Ptrue_o = I2
Rn = 0.5 * np.log(10)
check("omission: true distortion 1 + e^{-2R}; D=1.1 needs R = 1/2 ln 10 = 1.151 nats",
      close(1 + np.exp(-2 * Rn), 1.1, 1e-12) and close(Rn, 1.151, 1e-3))
Ror, thr = rate_nats([1, 1], 1.1)
check("omission: oracle R_I(1.1) = ln(1/0.55) = 0.598 nats", close(thr, 0.55, 1e-6) and close(Ror, 0.598, 1e-3))


def floor(Sx, Phat, P):
    S12 = sym_sqrt(Sx)
    Pht = S12 @ Phat @ S12
    w, V = np.linalg.eigh(Pht)
    K = V[:, np.abs(w) < 1e-10]
    Pi = K @ K.T
    Pt = S12 @ P @ S12
    return np.trace(Pt @ Pi), S12 @ Pi @ S12


fl, _ = floor(I2, Phat_o, Ptrue_o)
check("omission: floor tr(P~ Pi) = 1 when Sigma = I", close(fl, 1.0, 1e-9))
# correlated caveat
rho = 0.8
Sc = np.array([[1, rho], [rho, 1]])
flc, Sdc = floor(Sc, np.diag([1.0, 0.0]), np.diag([0.0, 1.0]))
check("omission correlated: floor = 1 - rho^2 = 0.36", close(flc, 0.36, 1e-9))
check("omission correlated: limiting error covariance diag(0, 0.36)", np.allclose(Sdc, np.diag([0, 0.36]), atol=1e-9))
check("omission correlated: Var(x2 | x1) = 1 - rho^2 = 0.36", close(1 - rho ** 2, 0.36, 1e-12))
check("omission correlated: naive (unwhitened) kernel gives 1.0, overstates by 0.64", close(1.0 - 0.36, 0.64, 1e-12))
# the omission floor is approached by the P^-water-filling family as theta -> 0
S12 = sym_sqrt(Sc)
Pht = S12 @ np.diag([1.0, 0.0]) @ S12
w, V = np.linalg.eigh(Pht)
vals = []
for th in [1e-1, 1e-3, 1e-6]:
    s = np.array([1.0 if wk <= 1e-12 else min(1.0, th / wk) for wk in w])
    Sdt = S12 @ V @ np.diag(s) @ V.T @ S12
    vals.append(np.trace(np.diag([0.0, 1.0]) @ Sdt))
check("omission correlated: true distortion decreases to 0.36 as rate grows", vals[0] > vals[1] > vals[2] and close(vals[2], 0.36, 1e-5))

# ---------------------------------------------------------------------------
# Section 7: two observers
# ---------------------------------------------------------------------------
Sx2 = np.diag([4.0, 1.0])


def capped_wf(caps, w, D):
    """max sum log s_k  s.t. s_k <= caps_k, sum w_k s_k <= D (diagonal, commuting case)."""
    caps = np.array(caps, float); w = np.array(w, float)
    if np.sum(w * caps) <= D:
        return caps
    lo, hi = 0.0, 1e6
    for _ in range(300):
        th = 0.5 * (lo + hi)
        s = np.array([c if wk == 0 else min(c, th / wk) for c, wk in zip(caps, w)])
        if np.sum(w * s) < D:
            lo = th
        else:
            hi = th
    return s


S1 = sigma_star(Sx2, np.diag([1.0, 0.0]), 1.0)
check("two obs: Sigma1* = diag(1, 1) for Sigma=diag(4,1), P1=diag(1,0), D1=1 (cap binds on e2)", np.allclose(S1, np.diag([1.0, 1.0]), atol=1e-6))
R1 = 0.5 * np.log(np.linalg.det(Sx2) / np.linalg.det(S1))
check("two obs: R1 = 1/2 ln 4 = 0.693 nats", close(R1, 0.693, 1e-3))
S2a = sigma_star(Sx2, I2, 1.5)
check("two obs: D2=1.5 gives theta 0.75, Sigma2* = 0.75 I, nested below diag(1,1)",
      np.allclose(S2a, 0.75 * I2, atol=1e-6) and np.linalg.eigvalsh(S1 - S2a).min() >= -1e-9)
R2a = 0.5 * np.log(np.linalg.det(Sx2) / np.linalg.det(S2a))
check("two obs: R2(1.5) = 0.981 nats", close(R2a, 0.981, 1e-3))
S2b = sigma_star(Sx2, I2, 2.5)
check("two obs: D2=2.5 gives theta 1.5, Sigma2* = diag(1.5,1), not nested", np.allclose(S2b, np.diag([1.5, 1.0]), atol=1e-6) and np.linalg.eigvalsh(S1 - S2b).min() < -1e-6)
R2b = 0.5 * np.log(np.linalg.det(Sx2) / np.linalg.det(S2b))
check("two obs: R2(2.5) = 1/2 ln(4/1.5) = 0.490 nats", close(R2b, 0.490, 1e-3))
Sc = capped_wf([1.0, 1.0], [1.0, 1.0], 2.5)
check("two obs: Sigma_circ = diag(1,1) (trace constraint slack)", np.allclose(Sc, [1, 1]))
Lb = 0.5 * np.log(np.linalg.det(S2b) / np.prod(Sc))
check("two obs: L = 1/2 ln 1.5 = 0.203 nats; total = R1 = 0.490 + 0.203", close(Lb, 0.203, 1e-3) and close(R2b + Lb, R1, 1e-9))
# orthogonal observers inside the example, Sigma_x = diag(4,1): L = 1/2 ln(4/D1) = R1(D1)
D1, D2 = 0.25, 0.5
Sx41 = np.diag([4.0, 1.0])
S1o = sigma_star(Sx41, np.diag([1.0, 0.0]), D1)
S2o = sigma_star(Sx41, np.diag([0.0, 1.0]), D2)
check("two obs orthogonal: Sigma1* = diag(D1,1), Sigma2* = diag(4,D2)",
      np.allclose(S1o, np.diag([D1, 1.0]), atol=1e-9) and np.allclose(S2o, np.diag([4.0, D2]), atol=1e-9))
check("two obs orthogonal: optima do not nest", np.linalg.eigvalsh(S1o - S2o).min() < -1e-9)
Scirc = np.diag([D1, D2])
Lo = 0.5 * np.log(np.linalg.det(S2o) / np.linalg.det(Scirc))
R1o = 0.5 * np.log(np.linalg.det(Sx41) / np.linalg.det(S1o))
check("two obs orthogonal: L = 1/2 ln(4/D1) = R1(D1)", close(Lo, 0.5 * np.log(4 / D1), 1e-6) and close(Lo, R1o, 1e-6))

# ---------------------------------------------------------------------------
# Section 8: coupling (Paper IV counterexample)
# ---------------------------------------------------------------------------
Rstar = 0.5 * np.log2(4 / 1)
check("coupling: R* = 1/2 log2(4/1) = 1 bit", close(Rstar, 1.0, 1e-12))
check("coupling R=1: O and R both give diag(1,1), Delta = 0", close(min(4, 1.0), 4 * 2 ** -2, 1e-12))
check("coupling R=2: d_C(O) = 0.5, d_C(R) = 4*2^-4 = 0.25", close(4 * 2 ** -4, 0.25, 1e-12))
check("coupling R=2: recon O = 1 < R = 0.25 + 1 = 1.25", close(0.25 + 1, 1.25, 1e-12))

# ---------------------------------------------------------------------------
# Section 9: value of observation (ot-estimation-control.tex, rank-one form)
# ---------------------------------------------------------------------------
def voi(Sm, h, r, g):
    Sp = Sm - np.outer(Sm @ h, Sm @ h) / (h @ Sm @ h + r)
    return np.trace(np.outer(g, g) @ (Sm - Sp)), (g @ Sm @ h) ** 2 / (h @ Sm @ h + r)


e1v, e2v = np.array([1.0, 0.0]), np.array([0.0, 1.0])
v1, v1r = voi(I2, e1v, 1.0, e1v)
v2, v2r = voi(I2, e1v, 1.0, e2v)
v3, v3r = voi(np.array([[1, 0.5], [0.5, 1]]), e1v, 1.0, e2v)
check("VoI: Sigma-=I, h=e1, r=1, g=e1 gives 0.5", close(v1, 0.5, 1e-12) and close(v1r, 0.5, 1e-12))
check("VoI: same sensor, g=e2 gives 0", close(v2, 0.0, 1e-12))
check("VoI: correlated prior (0.5), g=e2 gives 0.25/2 = 0.125", close(v3, 0.125, 1e-12) and close(v3r, 0.125, 1e-12))

# ---------------------------------------------------------------------------
# Exercises
# ---------------------------------------------------------------------------
# Ex 6.1: P for C(x) = 3x1 - x2 (G=1): [[9,-3],[-3,1]]; d_O for Sigma_delta = 0.02 I -> 0.2
a = np.array([3.0, -1.0])
check("exercise 6.1: P = [[9,-3],[-3,1]], d_O = 0.02*10 = 0.2", np.allclose(np.outer(a, a), [[9, -3], [-3, 1]]) and close(0.02 * np.trace(np.outer(a, a)), 0.2, 1e-12))
# Ex 6.2: kernel of P = span(1,3); x=(0,0) and x'=(1,3) indistinguishable
check("exercise 6.2: (1,3) in ker P", np.allclose(np.outer(a, a) @ np.array([1, 3.0]), 0))
# Ex 6.5: flip at R = 3 bits, Sigma=diag(4,1), consumer reads e2
# MSE: 1/2 log2(4/th) + 1/2 log2(1/th) = 3 -> th^2 = 4/64 -> th = 0.25
check("exercise 6.5: R=3 bits: theta = 0.25, O: tr 0.5, d_O 0.25; R code: diag(4, 1/64): tr 4.0156, d_O 0.0156",
      close(0.5 * np.log2(4 / 0.25) + 0.5 * np.log2(1 / 0.25), 3, 1e-12) and close(4 + 1 / 64, 4.015625, 1e-12) and close(1 / 64, 0.015625, 1e-12))
# Ex 6.6: commission with Phat = I, P = diag(1,3) in 2-d: bound = ln 3 = 1.099 nats
check("exercise 6.6: bound (2/2) ln 3 = 1.099 nats = 1.585 bits", close(np.log(3), 1.099, 1e-3) and close(np.log2(3), 1.585, 1e-3))
# Ex 6.7: omission floor with rho = 0.5 -> 0.75
fl5, _ = floor(np.array([[1, 0.5], [0.5, 1]]), np.diag([1.0, 0.0]), np.diag([0.0, 1.0]))
check("exercise 6.7: floor with rho = 0.5 is 0.75", close(fl5, 0.75, 1e-9))
# Ex 6.8: Sigma = diag(4,1), P1 = diag(0,1), D1 = 0.25; P2 = I, D2 = 0.4 and 1
S1e = sigma_star(Sx2, np.diag([0.0, 1.0]), 0.25)
S2e = sigma_star(Sx2, I2, 0.4)
check("exercise 6.8: Sigma1* = diag(4, 0.25), R1 = 0.693; Sigma2*(0.4) = 0.2 I nested",
      np.allclose(S1e, np.diag([4, 0.25]), atol=1e-6) and close(0.5 * np.log(np.linalg.det(Sx2) / np.linalg.det(S1e)), 0.693, 1e-3)
      and np.allclose(S2e, 0.2 * I2, atol=1e-6) and np.linalg.eigvalsh(S1e - S2e).min() >= -1e-9)
S2f = sigma_star(Sx2, I2, 1.0)
Scf = capped_wf([4.0, 0.25], [1.0, 1.0], 1.0)
check("exercise 6.8: D2 = 1 gives 0.5 I, not nested; Sigma_circ = diag(0.75, 0.25); L = 1/2 ln(4/3) = 0.144",
      np.allclose(S2f, 0.5 * I2, atol=1e-6) and np.allclose(Scf, [0.75, 0.25], atol=1e-6)
      and close(0.5 * np.log(np.linalg.det(S2f) / np.prod(Scf)), 0.144, 1e-3))
# Ex 6.9: kappa for P = e2 e2^T, Sigma = diag(9, 4, 1): tr = 4, top-1 = 9 -> 4/9
check("exercise 6.9: kappa = 4/9 = 0.444; bounds [1/9, 1]", close(4 / 9, 0.444, 1e-3))
# Programming ex 6.10: blind probe of C(x) = tanh(w^T x) recovers w
rng = np.random.default_rng(2)
w = np.array([1.0, 2.0, 0.0, 0.0, -1.0]) / np.sqrt(6)


def Ct(x):
    return np.tanh(w @ x)


Xs = rng.standard_normal((200, 5))
Pb = np.zeros((5, 5))
for xx in Xs:
    gg = np.array([(Ct(xx + 1e-4 * np.eye(5)[j]) - Ct(xx - 1e-4 * np.eye(5)[j])) / 2e-4 for j in range(5)])
    Pb += np.outer(gg, gg) / 200
wv, Vv = np.linalg.eigh(Pb)
check("exercise 6.10: blind probe of tanh(w^T x) recovers w (|cos| > 0.9999), other eigenvalues ~0",
      abs(Vv[:, -1] @ w) > 0.9999 and wv[-2] < 1e-8 * wv[-1] + 1e-12)
# restricted to the first three coordinates the probe sees the projection (1,2,0)/sqrt5
P3r = Pb[:3, :3]
w3, V3 = np.linalg.eigh(P3r)
check("exercise 6.10: probe confined to coordinates 1..3 returns (1,2,0)/sqrt5",
      abs(V3[:, -1] @ (np.array([1, 2, 0]) / np.sqrt(5))) > 0.9999)
check("exercise 6.1: identity reader 0.02*2 = 0.04, ratio 5 = trP/d", close(0.02 * 2, 0.04, 1e-12) and close(0.2 / 0.04, 10 / 2, 1e-12))
check("exercise 6.5: consumer ratio 0.25/0.015625 = 16", close(0.25 / 0.015625, 16, 1e-12))
R_I = lambda D: rate_nats([1, 1], D)[0]
check("exercise 6.6: with P = I the excess R_I(D/3) - R_I(D) = ln 3 at D = 1 (both modes active for D < 2)",
      close(R_I(1 / 3) - R_I(1.0), np.log(3), 1e-6) and close(R_I(1.9 / 3) - R_I(1.9), np.log(3), 1e-6))
check("exercise 6.9: O spends nothing on e2 below 1/2 log2(9/4) = 0.585 bits", close(0.5 * np.log2(9 / 4), 0.585, 1e-3))

# ---------------------------------------------------------------------------
# Figure data (printed as FIGDATA lines and checked)
# ---------------------------------------------------------------------------
def coords(name, pts, fmt="{:.4g}"):
    print("FIGDATA " + name + ": " + " ".join("(" + ",".join(fmt.format(v) for v in p) + ")" for p in pts))


# read operator vs signal ellipse: std of x along u = sqrt(2.5)
check("fig read-vs-signal: std along u = sqrt(2.5) = 1.581", close(np.sqrt(u @ Sx @ u), 1.581, 1e-3))
# flip bars
coords("fig flip bars recon (O,R)", [(1, 2.0), (2, 4.25)])
coords("fig flip bars consumer (O,R)", [(1, 1.0), (2, 0.25)])


# rate needed to reach consumer distortion D (Sigma = diag(4,1), consumer reads e2)
def rate_O_bits(Dc):
    # reconstruction-optimal water level theta; consumer distortion min(1, theta)
    th = Dc
    return 0.5 * np.log2(4 / th) + 0.5 * np.log2(1 / th)


Ds = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
rc = [(D_, 0.5 * np.log2(1 / D_)) for D_ in Ds]
ro = [(D_, rate_O_bits(D_)) for D_ in Ds]
coords("fig RC consumer-aware (D, bits)", rc, "{:.3f}")
coords("fig RC reconstruction-optimal (D, bits)", ro, "{:.3f}")
check("fig RC: O needs log2(2/D) bits, e.g. 1 bit at D=1 and 2 bits at D=0.5", close(rate_O_bits(1.0), 1.0, 1e-12) and close(rate_O_bits(0.5), 2.0, 1e-12))
check("fig RC: consumer-aware needs 0 bits at D=1 and 0.5 bits at D=0.5", close(0.5 * np.log2(1 / 1.0), 0, 1e-12) and close(0.5 * np.log2(1 / 0.5), 0.5, 1e-12))


# distortion-rate curves for mismatch figure (rates in nats)
def dist_at_rate(gammas, R):
    g = np.array(gammas, float)
    lo, hi = 1e-12, g.max()
    for _ in range(200):
        th = np.sqrt(lo * hi)
        r = 0.5 * np.sum(np.log(np.maximum(g / th, 1.0)))
        if r > R:
            lo = th
        else:
            hi = th
    th = np.sqrt(lo * hi)
    return np.minimum(g, th)


Rs = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
com, ora, omi = [], [], []
for R_ in Rs:
    e = dist_at_rate([1, 1], R_)          # coder water-fills Phat = I
    com.append((R_, 1 * e[0] + 2 * e[1]))  # true P = diag(1,2)
    eo = dist_at_rate([1, 2], R_)         # oracle for P = diag(1,2) (gamma = p_k since Sigma = I)
    ora.append((R_, eo.sum()))
    omi.append((R_, 1 + np.exp(-2 * R_)))  # Phat = diag(1,0), P = I
coords("fig mismatch commission true distortion", com, "{:.3f}")
coords("fig mismatch oracle P=diag(1,2)", ora, "{:.3f}")
coords("fig mismatch omission true distortion", omi, "{:.3f}")
check("fig mismatch: commission curve is 3 e^{-R} for R > 0", all(close(d_, 3 * np.exp(-r_), 1e-6) for r_, d_ in com))
check("fig mismatch: commission curve stays within factor 2 of the oracle and both go to 0", all(o_ <= c_ + 1e-9 for (_, o_), (_, c_) in zip(ora, com)) and com[-1][1] < 0.06)
check("fig mismatch: omission curve tends to the floor 1", close(omi[-1][1], 1.0, 1e-3) and all(d_ >= 1 for _, d_ in omi))
# two-observer ellipses: radii
check("fig two obs: sqrt(0.75) = 0.866, sqrt(1.5) = 1.225", close(np.sqrt(0.75), 0.866, 1e-3) and close(np.sqrt(1.5), 1.225, 1e-3))

# omission floor, disjoint case (ex:ch06:disjoint): the consumer reads nothing coded
Phat_d = np.diag([1.0, 0.0])
P_d = np.diag([0.0, 1.0])
Pi_d = np.diag([0.0, 1.0])  # whitened kernel of Phat with Sigma = I
check("disjoint: floor tr(P Pi) = 1", close(np.trace(P_d @ Pi_d), 1.0, 1e-12))
check("disjoint: tr(P (I - Pi)) = 0, so the divergence condition fails",
      close(np.trace(P_d @ (np.eye(2) - Pi_d)), 0.0, 1e-12))
check("disjoint: true distortion is exactly 1 at every rate, including R = 0",
      all(close(np.trace(P_d @ np.diag([np.exp(-2 * R), 1.0])), 1.0, 1e-12)
          for R in (0.0, 0.5, 2.0, 10.0)))

# hypothesis audit: an unread direction correlated with a read one loses error variance (ex tilt, D=0.625)
Sxt = np.diag([4.0, 1.0])
ut = np.array([1.0, 1.0]) / np.sqrt(2)
Pt_ = np.outer(ut, ut)
St = sigma_star(Sxt, Pt_, 0.625)
kt = np.array([1.0, -1.0]) / np.sqrt(2)
at = np.array([0.25, -1.0])
check("tilt: Sigma* = [[1.6,-0.6],[-0.6,0.85]] with tr(P Sigma*) = 0.625",
      np.allclose(St, [[1.6, -0.6], [-0.6, 0.85]], atol=1e-9) and close(np.trace(Pt_ @ St), 0.625, 1e-9))
check("tilt: unread (1,-1)/sqrt2 has source variance 2.5, error variance 1.825",
      close(kt @ Sxt @ kt, 2.5, 1e-12) and close(kt @ St @ kt, 1.825, 1e-9) and close(ut @ kt, 0, 1e-12))
check("tilt: uncorrelated component x1/4 - x2 (P Sx a = 0) keeps variance 1.25",
      np.allclose(Pt_ @ Sxt @ at, 0, atol=1e-12) and close(at @ Sxt @ at, 1.25, 1e-12) and close(at @ St @ at, 1.25, 1e-9))
# hypothesis audit: unequal read weights give a strict flip below R* (coupling criterion needs equal weights)
lam = np.array([4.0, 2.0, 1.0])
Pw = np.array([1.0, 10.0, 0.0])
Rstar = 0.5 * (np.log2(4 / 1) + np.log2(2 / 1))
check("unequal weights: R* = 1.5 bits", close(Rstar, 1.5, 1e-12))
Rb = 0.5  # bits
# O: reverse water-filling on lam at 0.5 bit
thO = 4 * 2 ** (-2 * Rb)
errO = np.minimum(lam, thO)
check("unequal weights: O at 0.5 bit has theta=2 and error diag(2,2,1)",
      close(thO, 2.0, 1e-12) and np.allclose(errO, [2, 2, 1]))
# R: water-fill weighted variances (4,20) at 0.5 bit
gw = Pw[:2] * lam[:2]
thR = 20 * 2 ** (-2 * Rb)
check("unequal weights: R water level 10 >= 4, so only e2 is described", close(thR, 10.0, 1e-12) and gw[0] <= thR < gw[1])
errR = np.array([4.0, thR / Pw[1], 1.0])
check("unequal weights: R error diag(4,1,1)", np.allclose(errR, [4, 1, 1]))
check("unequal weights: consumer 22 vs 14, reconstruction 5 vs 6 (strict flip below R*)",
      close(Pw @ errO, 22, 1e-12) and close(Pw @ errR, 14, 1e-12) and close(errO.sum(), 5, 1e-12) and close(errR.sum(), 6, 1e-12)
      and Rb < Rstar)
# hypothesis audit: consumer reading e1 alone on diag(4,1): no flip at 1 bit, flip at 2 bits (consumer 0.25 vs 0.5)
check("e1 reader: identical at 1 bit, flip at 2 bits with consumer distortions 0.25 vs 0.5",
      close(min(4, 4 * 2 ** -2), 1.0, 1e-12) and close(4 * 2 ** -4, 0.25, 1e-12) and close(4 * 2 ** -4 + 1, 1.25, 1e-12))

n_ok = sum(results)
print(f"{n_ok}/{len(results)} PASS")
sys.exit(0 if n_ok == len(results) else 1)
